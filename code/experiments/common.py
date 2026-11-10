"""E0–E9 共用的路径约定、场景配对、Omega 构造与表格落盘。"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from qcert.pfa_eval import (boxes_to_letterbox, build_omega, calibrate_lambda,
                            cell_centers, measure_pfa)

CODE = Path(__file__).resolve().parents[1]      # .../code
ROOT = CODE.parent                              # .../dinghuibianyuan
RESULTS = CODE / "results"


# ------------------------------------------------------------------ 文件
def ensure_dir(p) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def json_dump(obj, path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def json_load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _cell(v):
    if v is None:
        return ""
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        return f"{v:.6g}"
    return str(v)


def save_table(rows, out_dir, name: str, title: str = ""):
    """把 list[dict] 落成 csv / md / json 三份，返回路径字典。"""
    out_dir = ensure_dir(out_dir)
    rows = list(rows)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)

    paths = {}
    p_csv = out_dir / f"{name}.csv"
    with p_csv.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys or ["note"], extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: _cell(r.get(k)) for k in keys})
    paths["csv"] = p_csv

    p_json = out_dir / f"{name}.json"
    json_dump(rows, p_json)
    paths["json"] = p_json

    lines = []
    if title:
        lines.append(f"### {title}\n")
    if keys:
        lines.append("| " + " | ".join(keys) + " |")
        lines.append("|" + "|".join(["---"] * len(keys)) + "|")
        for r in rows:
            lines.append("| " + " | ".join(_cell(r.get(k)) for k in keys) + " |")
    else:
        lines.append("_（空表：尚无数据，跑完对应实验后回填）_")
    p_md = out_dir / f"{name}.md"
    p_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    paths["md"] = p_md
    return paths


def print_table(rows, limit: int = 40):
    for r in rows[:limit]:
        print("  " + "  ".join(f"{k}={_cell(v)}" for k, v in r.items()))
    if len(rows) > limit:
        print(f"  ... 共 {len(rows)} 行")


# ------------------------------------------------------------------ GT
def read_yolo_xyxy(label_path, img_w: float, img_h: float) -> np.ndarray:
    """YOLO 归一化标签 -> 原图 xyxy。文件不存在返回空数组。"""
    p = Path(label_path)
    if not p.exists():
        return np.zeros((0, 4), dtype=np.float32)
    rows = []
    for line in p.read_text(encoding="utf-8").strip().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        _, xc, yc, w, h = parts[:5]
        xc, yc, w, h = float(xc) * img_w, float(yc) * img_h, float(w) * img_w, float(h) * img_h
        rows.append([xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2])
    return np.asarray(rows, dtype=np.float32).reshape(-1, 4)


def gt_for(stem, label_dir, meta: dict):
    """按 dump 时记录的 iw/ih 读 GT；没有标签目录或缺少尺寸则返回 None。"""
    if not label_dir:
        return None
    iw, ih = float(meta.get("iw", 0) or 0), float(meta.get("ih", 0) or 0)
    if iw <= 0 or ih <= 0:
        raise ValueError(f"{stem}: meta 缺 iw/ih，无法把 GT 投到 letterbox 坐标"
                         "（dump 时应写入 iw/ih）")
    return read_yolo_xyxy(Path(label_dir) / f"{stem}.txt", iw, ih)


# ------------------------------------------------------------------ 配对
def load_pairs(ref_dir, test_dir, verbose=True):
    """两个 dump 目录按 stem 配对 -> [(stem, ref_field, test_field)]。"""
    from qcert.scores import load_dir

    a = load_dir(ref_dir)
    b = load_dir(test_dir)
    stems = sorted(set(a) & set(b))
    if not stems:
        raise ValueError(f"{ref_dir} 与 {test_dir} 没有同名场景")
    if verbose:
        print(f"[pair] {len(stems)} 个共同场景；"
              f"仅 ref {len(set(a) - set(b))}，仅 test {len(set(b) - set(a))}")
    return [(s, a[s], b[s]) for s in stems]


# ------------------------------------------------------------------ Omega
def omega_for(field, boxes=None, guard_px: float = 8.0, block_px: float = 64.0,
              keep: float = 0.5, seed: int = 0):
    """一个分数场的 (Omega_cal, Omega_meas, info) 与网格中心。

    位置单位是**检测器网格中心**（letterbox 坐标系），两条链路的 Omega 天然对齐。
    """
    if field.head_sizes is None:
        raise ValueError(f"{field.stem}: 缺 head_sizes")
    imgsz = int(field.meta.get("imgsz", 640))
    centers = cell_centers(field.head_sizes, imgsz)
    lb = boxes_to_letterbox(boxes, field.meta) if boxes is not None and len(boxes) else None
    mask_cal, mask_meas, info = build_omega(
        centers, lb, guard_px=guard_px, block_px=block_px, keep=keep, seed=seed)
    return (mask_cal, mask_meas, info), centers


def scene_pfa_pair(ref, test, boxes=None, alpha: float = 1e-3,
                   guard_px: float = 8.0, block_px: float = 64.0,
                   keep: float = 0.5, seed: int = 0):
    """同一场景：各自标定 lambda、各自测 P_fa，返回一行记录。

    注意两条链路用的是**同一批网格位置**（Omega 由 centers 决定，与分数无关），
    因此 R 是真正的"逐位置配对"比值，而不是两次独立采样之比。
    """
    (mc, mm, info), centers = omega_for(ref, boxes, guard_px, block_px, keep, seed)

    def one(field):
        z = field.z()
        lam = calibrate_lambda(z, mc, alpha)
        return lam, measure_pfa(z, mm, lam)

    lam_r, pfa_r = one(ref)
    lam_t, pfa_t = one(test)
    return {
        "stem": ref.stem,
        "n_cal": info["n_cal"],
        "n_meas": info["n_meas"],
        "n_blocks": info["n_blocks"],
        "n_gt_cells": info["n_gt_cells"],
        "lambda_ref": lam_r,
        "lambda_test": lam_t,
        "pfa_ref": pfa_r,
        "pfa_test": pfa_t,
        "R": (pfa_t / pfa_r) if pfa_r > 0 else (float("inf") if pfa_t > 0 else 1.0),
        "impl_ref": ref.meta.get("impl", "?"),
        "impl_test": test.meta.get("impl", "?"),
    }


def group_of(field, key: str = "sea"):
    """场景分组键：meta 里优先 sea_state / sea，其次 contrast 档。"""
    m = field.meta
    if key == "sea":
        return str(m.get("sea_state", m.get("sea", "all")))
    if key == "contrast":
        return str(m.get("contrast", 1.0))
    return str(m.get(key, "all"))


def aggregate(rows, key: str = "R"):
    v = np.asarray([r[key] for r in rows], dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "n": 0}
    return {"mean": float(v.mean()), "median": float(np.median(v)),
            "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90)),
            "n": int(v.size)}
