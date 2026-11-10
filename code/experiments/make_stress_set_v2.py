# -*- coding: utf-8 -*-
"""E2 压测集 v2 构造（本地，生成后打包推板）· 2026-11-04。

v1 失效复盘（data/stress/ 保留作审计痕迹）：
  纯增益+clip 破坏图像结构（直方图削顶），板上 Neck int8 sat 全零。
  本地三重排除证据 —— ① probe_stress_tf.py：11 种图像域变换超限 0/12；
  ② probe_copy_paste.py：强目标 patch（源 p5≈39）贴低激活接收图，med 仍
  ~7.1，超限 0/12（激活是场景级全局属性）；③ 自然分布 P5 med=11.70 /
  P90=26.64 / P99=35.19 / max=40.03，超标定上限 38.61 仅 4/1067 ——
  AMCT 标定（calib256 ← train 分层抽样）刚好贴住自然分布绝对上限。

  → v2 定案：按 P5 自然激活分档选图（数据驱动）。c1/c2/c3 = 低/中/高档
  各 100 张（p5 降序 rank 带：底部 100 / 中位带 100 / 顶部 100），剂量
  单调由构造保证，c3 含全部 P5 自然超限图。若 c3 饱和率仍不上升，即为
  "量化 range 对结构级激活鲁棒"的阴性证据（§13.2 → 转 E4 分母主线）。

档位由文件名前缀 c{N}_ 承载（沿用 v1 dump 合并目录约定）；板上跑
acl_infer 不加 --contrast（meta.contrast 如实保持 1.0：无图像变换），
run_e2 用 --dose-from stem 按前缀取剂量。

选图依据：asc-sar/results/act_scan_ssdd.csv（probe_act_scan.py 产出，
SSDD 全集 1067 张 fp ONNX Neck SiLU act_max，板上同一 letterbox 预处理）。

产物（--out，默认 asc-sar/data/stress_v2/）：
    c{1,2,3}_{stem}.jpg      原图字节级拷贝（split 记录在 manifest）
    labels/c{N}_{stem}.txt   YOLO GT 重命名拷贝（缺失则跳过并记录）
    manifest.json            分档协议 + 逐图 p3/p4/p5 + 偏差全程记录
    ../stress_v2.tgz         tarfile 打包（跨平台）
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import tarfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent / "asc-sar"
CSV_IN = ROOT / "results" / "act_scan_ssdd.csv"
IMAGES = ROOT / "datasets" / "ssdd" / "images"
LABELS = ROOT / "datasets" / "ssdd" / "labels"
CAL_MAX = {"p3": 12.08, "p4": 17.43, "p5": 38.61}   # AMCT 标定上限(SiLU 输出)


def read_scan(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as fp:
        for r in csv.DictReader(fp):
            rows.append({"split": r["split"], "stem": r["stem"],
                         "p3": float(r["p3"]), "p4": float(r["p4"]),
                         "p5": float(r["p5"])})
    if not rows:
        raise SystemExit(f"[err] 空扫描表: {path}")
    return rows


def bands_of(n, per):
    """p5 降序 rank 带（含头不含尾）：c1=底部，c2=中位带，c3=顶部。"""
    c = n // 2
    return {"c1": (n - per, n),
            "c2": (c - per // 2, c - per // 2 + per),
            "c3": (0, per)}


def stats(vals):
    v = np.asarray(vals, dtype=np.float64)
    return {"min": round(float(v.min()), 2), "med": round(float(np.median(v)), 2),
            "max": round(float(v.max()), 2)}


def over_layers(row):
    return [t for t in ("p3", "p4", "p5") if row[t] > CAL_MAX[t]]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=str(CSV_IN))
    ap.add_argument("--out", default=str(ROOT / "data" / "stress_v2"))
    ap.add_argument("--n", type=int, default=100, help="每档张数（协议 100）")
    args = ap.parse_args()

    rows = read_scan(args.csv)
    rows.sort(key=lambda r: (-r["p5"], -r["p4"], -r["p3"], r["stem"]))  # 确定性
    n = len(rows)
    bands = bands_of(n, args.n)
    if bands["c2"][1] > bands["c1"][0]:
        raise SystemExit(f"[err] 语料不足：n={n} 撑不起不相交三档×{args.n}")

    p5_over = [i for i, r in enumerate(rows) if r["p5"] > CAL_MAX["p5"]]
    if len(p5_over) > args.n:
        raise SystemExit(f"[err] P5 超限图 {len(p5_over)} 张 > 每档 {args.n}，c3 装不下")
    if any(i >= args.n for i in p5_over):
        raise SystemExit("[err] 存在 P5 超限图未落入 c3 顶部档（排序异常）")

    out = Path(args.out)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    manifest = {
        "protocol": ("paper/04 §5 · E2 压测集 v2（2026-11-04）：按 P5 自然激活"
                     "分档选图，替代 v1 人工增益（×1/×2/×3）"),
        "deviation": [
            "v1（data/stress/，保留作审计痕迹）纯增益+clip 破坏图像结构，"
            "板上 Neck int8 sat 全零（激活不升反降）",
            "probe_stress_tf.py：11 种图像域变换（增益/gamma/CLAHE/杂波注入等）"
            "推过标定上限 0/12 —— 激活是场景级全局属性",
            "probe_copy_paste.py：强目标 patch（源 p5≈39）贴低激活接收图，"
            "med 仍 ~7.1，超限 0/12",
            "自然分布（act_scan_ssdd.csv，1067 张）：P5 med=11.70 P90=26.64 "
            "P99=35.19 max=40.03，超 38.61 仅 4/1067 —— AMCT 标定"
            "（calib256 ← train 分层抽样）贴住自然分布绝对上限",
            "结论：唯一有效压测形态为自然激活分档；若 c3 饱和率仍不上升，即为"
            "量化 range 对结构级激活鲁棒的阴性证据（§13.2 → 转 E4 分母主线）",
        ],
        "dose": {"var": "p5_act_max (fp ONNX Neck SiLU 输出)",
                 "carrier": "文件名前缀 c{1,2,3}_（c1 < c2 < c3，P5 递增）",
                 "meta_contrast": 1.0,
                 "run_e2": "--dose-from stem"},
        "cal_max": CAL_MAX,
        "n_per_tier": args.n,
        "corpus": {"csv": "results/act_scan_ssdd.csv",
                   "producer": "probe_act_scan.py（fp ONNX best.onnx，"
                               "板上同一 letterbox 预处理）",
                   "n": n,
                   **{t: stats([r[t] for r in rows]) for t in ("p3", "p4", "p5")}},
        "over_limit_all": {t: [f"{r['split']}/{r['stem']}={r[t]:.2f}"
                               for r in rows if r[t] > CAL_MAX[t]]
                           for t in ("p3", "p4", "p5")},
        "tiers": {},
        "labels_missing": [],
    }

    for tier, (lo, hi) in bands.items():
        sel = rows[lo:hi]
        imgs = []
        for r in sel:
            src = IMAGES / r["split"] / f"{r['stem']}.jpg"
            if not src.exists():
                raise SystemExit(f"[err] 原图缺失: {src}")
            shutil.copy2(src, out / f"{tier}_{r['stem']}.jpg")   # 字节级，不重编码
            lab = LABELS / r["split"] / f"{r['stem']}.txt"
            if lab.exists():
                shutil.copy2(lab, out / "labels" / f"{tier}_{r['stem']}.txt")
            else:
                manifest["labels_missing"].append(f"{tier}_{r['stem']}")
            imgs.append({**{k: r[k] for k in ("split", "stem", "p3", "p4", "p5")},
                         "over_cal": over_layers(r)})
        manifest["tiers"][tier] = {
            "rank_band_p5_desc": [lo, hi],
            "n": len(sel),
            "p5": stats([r["p5"] for r in sel]),
            "over_p5_cal": sum(1 for r in sel if r["p5"] > CAL_MAX["p5"]),
            "images": imgs,
        }

    with open(out / "manifest.json", "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, ensure_ascii=False, indent=1)

    tgz = out.parent / (out.name + ".tgz")
    with tarfile.open(tgz, "w:gz") as tar:
        tar.add(out, arcname=out.name)

    print(f"[ok] 三档各 {args.n} 张 -> {out}  +  {tgz.name} "
          f"({tgz.stat().st_size / 1e6:.1f} MB)")
    for tier in ("c1", "c2", "c3"):
        t = manifest["tiers"][tier]
        s = t["p5"]
        print(f"  {tier}  rank{t['rank_band_p5_desc']}  p5 {s['min']:.1f} / "
              f"{s['med']:.1f} / {s['max']:.1f}   超P5限 {t['over_p5_cal']}")
    print(f"  P5 超限图（>{CAL_MAX['p5']}）共 {len(p5_over)} 张，全部在 c3："
          + " ".join(manifest["over_limit_all"]["p5"]))
    if manifest["labels_missing"]:
        print(f"  [warn] 缺标签 {len(manifest['labels_missing'])} 张："
              f"{manifest['labels_missing'][:5]}")
    else:
        print("  labels 全部齐备")


if __name__ == "__main__":
    main()
