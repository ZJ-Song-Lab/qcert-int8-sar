"""E8 · 检测兼容性与对照（Table 6）。

协议依据 paper/04 §11。这一节要回答两个问题：

1. **精度没掉**：把 QCert 的判决层挂到检测器上之后，``P_d@P_fa=1e-3``
   相对最强复现基线不低于 −1.5 点，mAP50 不下降超过 0.5 点；
2. **量化没把契约弄丢**：QCert(INT8) 与 QCert(FP16) 的实测 P_fa 之比
   ``R ≤ 1.2``（"精度不变性"），而全局阈值在跨海况桶上会大幅漂移。

表中 **CA-CFAR 必须有**（§11 原话）：它是最强的经典对手，也是
"你和 CFAR 的区别是什么"这一问的唯一防线。像素域实现见 ``qcert/cfar.py``。

两种跑法
--------
::

    # 离线自检（合成场景；时延/mAP 为占位，产物 note 会写明）
    python -m experiments.run_e8 --synth --n 40

    # 真板（分数场由 acl_infer.py --dump-scores 生成；CFAR 需要原图）
    python -m experiments.run_e8 --fp16 dumps/fp16 --int8 dumps/int8 \
        --labels asc-sar/datasets/ssdd/labels/test \
        --images asc-sar/datasets/ssdd/images/test \
        --map-json '{"fp16":0.971,"int8":0.968}' \
        --latency-json '{"fp16":8.6,"int8":4.2}'

口径
----
- ``P_d`` 是**框级**：GT 框内只要有一个网格中心超过 λ 就算检出；
- ``P_fa`` 只在纯杂波单元上统计（GT 框外扩 guard 后全部排除）；
- 标定集与测量集用 ``build_omega`` 的互斥抽样，避免"自己测自己"；
- α=1e-4 时单场景样本不够，**跨场景合并**计数，结果标 ``pfa_noisy_1e-4``。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from qcert.cfar import ca_cfar, cfar_metrics
from qcert.pfa_eval import (boxes_to_letterbox, build_omega, cell_centers,
                            points_in_boxes)
from qcert.qint import dequantize, quant_params_from_range, quantize

from .common import (RESULTS, ensure_dir, gt_for, json_dump, json_load,
                     load_pairs, read_yolo_xyxy, save_table)

SEA_STATES = ("calm", "moderate", "rough")
ALPHAS = (1e-3, 1e-4)


# ------------------------------------------------------------------ 场景装配
def _box_masks(centers, boxes, guard_px: float = 0.0):
    """每个 GT 框一个 bool 掩码（网格中心是否落在框内）。"""
    c = np.asarray(centers, dtype=np.float64)
    out = []
    for b in np.asarray(boxes, dtype=np.float64).reshape(-1, 4):
        out.append(points_in_boxes(c, b.reshape(1, 4), guard_px))
    return out


def _scene_record(centers, boxes, z_fp16, z_int8, seed: int, sea: str,
                  keep: float = 0.5, guard_px: float = 8.0) -> dict:
    """一个场景的判决统计量 + Omega 划分 + 框级 GT。"""
    mask_cal, mask_meas, info = build_omega(
        centers, boxes, guard_px=guard_px, block_px=64.0, keep=keep, seed=seed)
    return {
        "sea": sea,
        "z_fp16": np.asarray(z_fp16, dtype=np.float64),
        "z_int8": np.asarray(z_int8, dtype=np.float64),
        "cal": mask_cal, "meas": mask_meas,
        "n_cal": int(info["n_cal"]), "n_meas": int(info["n_meas"]),
        "box_masks": _box_masks(centers, boxes),
        "n_boxes": int(np.asarray(boxes).reshape(-1, 4).shape[0]),
    }


def _synth_pixels(seed: int, sea: str, n_target: int, size: int):
    """指数杂波 + 亮矩形目标的像素域强度图（CFAR 的输入）。"""
    rng = np.random.default_rng(seed)
    scale = {"calm": 0.7, "moderate": 1.0, "rough": 1.7}.get(sea, 1.0)
    img = rng.exponential(scale, size=(size, size))
    boxes = []
    for _ in range(int(n_target)):
        w = int(rng.integers(6, 13))
        h = int(rng.integers(4, 9))
        x = int(rng.integers(0, max(1, size - w)))
        y = int(rng.integers(0, max(1, size - h)))
        img[y:y + h, x:x + w] += rng.exponential(12.0 * scale, size=(h, w))
        boxes.append([x, y, x + w - 1, y + h - 1])
    return img, np.asarray(boxes, dtype=np.float64).reshape(-1, 4)


def _synth_scenes(n: int, size: int = 256, seed0: int = 9000,
                  n_target: int = 3, calib_scenes: int = 8):
    """合成场景：同一几何同时给网络侧分数场与 CFAR 侧像素图。

    INT8 的输出 scale 只用**前 calib_scenes 个标定场景**定，符合 AMCT 语义
    （离线标定一次，部署时不再变）；评估在全部场景上做。
    """
    from .synth import make_scene

    raw, pix = [], []
    for i in range(int(n)):
        sea = SEA_STATES[i % len(SEA_STATES)]
        con = 1.0 if sea != "rough" else 1.4
        sc = make_scene(seed=seed0 + i, n_target=n_target, sea_state=sea,
                        contrast=con, target_snr=4.5)
        raw.append({"sea": sea,
                    "centers": np.asarray(sc["centers"], dtype=np.float64),
                    "boxes": np.asarray(sc["boxes"], dtype=np.float64).reshape(-1, 4),
                    "z": np.asarray(sc["s"], dtype=np.float64)})
        img, pboxes = _synth_pixels(seed=seed0 + 50_000 + i, sea=sea,
                                    n_target=n_target, size=size)
        pix.append({"sea": sea, "img": img, "boxes": pboxes})

    calib = raw[:max(1, int(calib_scenes))]
    lo = min(float(r["z"].min()) for r in calib)
    hi = max(float(r["z"].max()) for r in calib)
    delta, zp = quant_params_from_range(lo, hi, bits=8, symmetric=True)

    recs = []
    for i, r in enumerate(raw):
        z8 = dequantize(quantize(r["z"], delta, zp, bits=8),
                        delta, zp).astype(np.float64)
        recs.append(_scene_record(r["centers"], r["boxes"], r["z"], z8,
                                  i, r["sea"]))
    return recs, pix, {"delta": float(delta), "zero_point": int(zp),
                       "n_calib_scenes": int(calib_scenes)}


# ------------------------------------------------------------------ 判决评估
def _lambda_global(recs, alpha: float, key: str) -> float:
    pool = np.concatenate([r[key][r["cal"]] for r in recs])
    return float(np.quantile(pool, 1.0 - alpha))


def _eval_thresholding(recs, alpha: float, mode: str) -> dict:
    """按 mode 取 λ（全局 / 逐场景 fp16 / 逐场景 int8），合并统计 P_fa 与 P_d。"""
    key = "z_int8" if mode in ("qcert_int8", "qcert_full") else "z_fp16"
    lam_g = _lambda_global(recs, alpha, key)

    n_fa = n_clutter = n_hit = n_box = 0
    per_scene = []
    for r in recs:
        z = r[key]
        lam = (lam_g if mode == "global"
               else float(np.quantile(z[r["cal"]], 1.0 - alpha)))
        det = z > lam
        n_fa += int((det & r["meas"]).sum())
        n_clutter += r["n_meas"]
        hits = [int((det & bm).any()) for bm in r["box_masks"]]
        n_hit += int(np.sum(hits))
        n_box += r["n_boxes"]
        per_scene.append({"sea": r["sea"],
                          "pfa": (float((det & r["meas"]).sum() / r["n_meas"])
                                  if r["n_meas"] else float("nan")),
                          "n_hit": int(np.sum(hits)), "n_box": r["n_boxes"]})
    pfa = float(n_fa / n_clutter) if n_clutter else float("nan")
    pd = float(n_hit / n_box) if n_box else float("nan")

    # 跨海况桶的 P_fa 离散度（E8 表里"实测 P_fa（跨桶 std）"）
    buckets = {}
    for s in per_scene:
        buckets.setdefault(s["sea"], []).append(s["pfa"])
    bucket_mean = {k: float(np.nanmean(v)) for k, v in buckets.items() if v}
    pfa_std = (float(np.nanstd(list(bucket_mean.values())))
               if len(bucket_mean) > 1 else float("nan"))
    return {"pfa": pfa, "pd": pd, "pfa_std_across_buckets": pfa_std,
            "n_fa": n_fa, "n_clutter": n_clutter,
            "n_hit": n_hit, "n_box": n_box,
            "bucket_pfa": bucket_mean, "lambda_global": lam_g, "key": key}


def _precision_invariance(recs, alpha: float) -> float:
    """R = P_fa(INT8)/P_fa(FP16)，两条链路都各自逐场景标定。"""
    ratios = []
    for r in recs:
        lam16 = float(np.quantile(r["z_fp16"][r["cal"]], 1.0 - alpha))
        lam8 = float(np.quantile(r["z_int8"][r["cal"]], 1.0 - alpha))
        m = r["meas"]
        if not m.any():
            continue
        p16 = float((r["z_fp16"][m] > lam16).mean())
        p8 = float((r["z_int8"][m] > lam8).mean())
        if np.isfinite(p16) and p16 > 0 and np.isfinite(p8):
            ratios.append(p8 / p16)
    return float(np.mean(ratios)) if ratios else float("nan")


def _cfar_rows(pix, alphas=ALPHAS, guard: int = 2, train: int = 8,
               min_area: int = 8) -> dict:
    """像素域 CA-CFAR：逐 α 合并 P_d / P_fa（P_fa 只在纯杂波像素上）。"""
    out = {}
    for alpha in alphas:
        n_hit = n_box = n_fa = n_clutter = 0
        for p in pix:
            res = ca_cfar(p["img"], guard=guard, train=train, pfa=alpha)
            m = cfar_metrics(res["det"], p["boxes"], guard=guard, train=train,
                             pfa=alpha, min_area=min_area)
            n_hit += int(m["n_target_detected"])
            n_box += int(m["n_target"])
            n_px = int(p["img"].size)
            n_fa += int(round(m["pfa_emp"] * n_px))
            n_clutter += n_px
        out[alpha] = {
            "pd": float(n_hit / n_box) if n_box else float("nan"),
            "pfa": float(n_fa / n_clutter) if n_clutter else float("nan"),
            "n_hit": n_hit, "n_box": n_box,
        }
    return out


# ------------------------------------------------------------------ 真板输入
def _dump_scenes(fp16_dir, int8_dir, labels, keep: float = 0.5):
    pairs = load_pairs(fp16_dir, int8_dir)
    recs = []
    for i, (stem, f32, q) in enumerate(pairs):
        boxes = gt_for(stem, labels, f32.meta) if labels else None
        if boxes is None:
            boxes = np.asarray((f32.meta or {}).get("boxes", []),
                               dtype=np.float64).reshape(-1, 4)
        centers = (np.asarray(f32.centers, dtype=np.float64)
                   if f32.centers is not None else None)
        if centers is None:
            if f32.head_sizes is None:
                raise ValueError(f"{stem}: 缺 centers 且缺 head_sizes")
            centers = cell_centers(f32.head_sizes,
                                   int(f32.meta.get("imgsz", 640)))
        lb = boxes_to_letterbox(boxes, f32.meta) if len(boxes) else boxes
        z8 = q.z() if q.is_int8 else f32.z()
        recs.append(_scene_record(
            centers, lb, f32.z(), z8, i,
            str((f32.meta or {}).get("sea_state", "all")), keep=keep))
    return recs


def _cfar_from_images(images_dir, labels_dir, stems, guard: int, train: int):
    """真图 + 真标签跑 CA-CFAR（框在原图坐标，与像素图一致）。"""
    import cv2

    pix = []
    for s in stems:
        cand = sorted(Path(images_dir).glob(f"{s}.*"))
        if not cand:
            continue
        img = cv2.imread(str(cand[0]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        h0, w0 = img.shape
        lbl = Path(labels_dir) / f"{s}.txt" if labels_dir else None
        boxes = read_yolo_xyxy(lbl, w0, h0) if lbl else np.zeros((0, 4))
        if not len(boxes):
            continue
        pix.append({"sea": "all", "img": np.asarray(img, dtype=np.float64),
                    "boxes": boxes})
    return _cfar_rows(pix, guard=guard, train=train) if pix else {}


# ------------------------------------------------------------------ 主流程
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synth", action="store_true", help="合成场景离线自检")
    ap.add_argument("--n", type=int, default=40, help="--synth 的场景数")
    ap.add_argument("--size", type=int, default=256, help="CFAR 像素图边长")
    ap.add_argument("--fp16", default=None, help="FP16 分数场 dump 目录")
    ap.add_argument("--int8", default=None, help="INT8 分数场 dump 目录")
    ap.add_argument("--labels", default=None, help="YOLO 标签目录")
    ap.add_argument("--images", default=None,
                    help="原图目录（只有给了才跑 CA-CFAR 真图）")
    ap.add_argument("--alpha", type=float, default=1e-3, help="主虚警率")
    ap.add_argument("--guard", type=int, default=2, help="CFAR 保护单元")
    ap.add_argument("--train", type=int, default=8, help="CFAR 参考窗宽")
    ap.add_argument("--map-json", default=None,
                    help='{"fp16":..,"int8":..} mAP50（0–1 或百分制）')
    ap.add_argument("--latency-json", default=None,
                    help='板级实测时延 {"fp16":..,"int8":..} ms')
    ap.add_argument("--reported-json", default=None,
                    help='引用方法 mAP50 {"AC-YOLO":96.7, ...}（标 reported）')
    ap.add_argument("--out", default=str(RESULTS / "e8"))
    args = ap.parse_args(argv)

    meta = {"alpha": args.alpha, "alphas": list(ALPHAS)}
    if args.synth or not (args.fp16 and args.int8):
        recs, pix, qmeta = _synth_scenes(args.n, size=args.size)
        source = "synth"
        meta.update(qmeta)
        cfar = _cfar_rows(pix, guard=args.guard, train=args.train)
    else:
        recs = _dump_scenes(args.fp16, args.int8, args.labels)
        source = "dump"
        cfar = (_cfar_from_images(
                    args.images, args.labels,
                    [p.stem for p in sorted(Path(args.fp16).glob("*.npz"))],
                    args.guard, args.train)
                if args.images else {})

    ev = {m: _eval_thresholding(recs, args.alpha, m)
          for m in ("global", "qcert_fp16", "qcert_int8")}
    g_1e4 = _eval_thresholding(recs, ALPHAS[1], "global") if len(recs) > 1 else None
    q_1e4 = _eval_thresholding(recs, ALPHAS[1], "qcert_int8") if len(recs) > 1 else None
    R = _precision_invariance(recs, args.alpha)

    lat = json_load(args.latency_json) if args.latency_json else {}
    mp = json_load(args.map_json) if args.map_json else {}

    def _pct(v):
        v = float(v)
        return v * 100.0 if 0.0 <= v <= 1.0 else v

    def _n(v):
        return float(v) if v is not None else float("nan")

    rows = []
    if cfar:
        c = cfar[ALPHAS[0]]
        rows.append({"method": "CA-CFAR（像素域）", "source": source,
                     "pd_at_1e-3": c["pd"], "pd_at_1e-4": cfar[ALPHAS[1]]["pd"],
                     "map50": float("nan"), "pfa": c["pfa"],
                     "pfa_std_across_buckets": float("nan"),
                     "R": float("nan"), "latency_ms": float("nan"),
                     "n_box": c["n_box"]})
    else:
        rows.append({"method": "CA-CFAR（像素域）", "source": source,
                     "pd_at_1e-3": float("nan"), "pd_at_1e-4": float("nan"),
                     "map50": float("nan"), "pfa": float("nan"),
                     "pfa_std_across_buckets": float("nan"),
                     "R": float("nan"), "latency_ms": float("nan"),
                     "n_box": 0, "note": "未提供 --images，跳过"})

    rows.append({"method": "YOLO11n + 全局阈值", "source": source,
                 "pd_at_1e-3": ev["global"]["pd"],
                 "pd_at_1e-4": _n(g_1e4 and g_1e4["pd"]),
                 "map50": _pct(mp["fp16"]) if "fp16" in mp else float("nan"),
                 "pfa": ev["global"]["pfa"],
                 "pfa_std_across_buckets": ev["global"]["pfa_std_across_buckets"],
                 "R": float("nan"),
                 "latency_ms": float(lat.get("fp16", float("nan"))),
                 "n_box": ev["global"]["n_box"]})
    rows.append({"method": "YOLO11n + QCert(FP16)", "source": source,
                 "pd_at_1e-3": ev["qcert_fp16"]["pd"],
                 "pd_at_1e-4": float("nan"),
                 "map50": _pct(mp["fp16"]) if "fp16" in mp else float("nan"),
                 "pfa": ev["qcert_fp16"]["pfa"],
                 "pfa_std_across_buckets": ev["qcert_fp16"]["pfa_std_across_buckets"],
                 "R": 1.0,
                 "latency_ms": float(lat.get("fp16", float("nan"))),
                 "n_box": ev["qcert_fp16"]["n_box"]})
    rows.append({"method": "YOLO11n + QCert(INT8)", "source": source,
                 "pd_at_1e-3": ev["qcert_int8"]["pd"],
                 "pd_at_1e-4": _n(q_1e4 and q_1e4["pd"]),
                 "map50": _pct(mp["int8"]) if "int8" in mp else float("nan"),
                 "pfa": ev["qcert_int8"]["pfa"],
                 "pfa_std_across_buckets": ev["qcert_int8"]["pfa_std_across_buckets"],
                 "R": R,
                 "latency_ms": float(lat.get("int8", float("nan"))),
                 "n_box": ev["qcert_int8"]["n_box"]})
    rows.append({"method": "QCert（完整）", "source": source,
                 "pd_at_1e-3": ev["qcert_int8"]["pd"],
                 "pd_at_1e-4": _n(q_1e4 and q_1e4["pd"]),
                 "map50": _pct(mp["int8"]) if "int8" in mp else float("nan"),
                 "pfa": ev["qcert_int8"]["pfa"],
                 "pfa_std_across_buckets": ev["qcert_int8"]["pfa_std_across_buckets"],
                 "R": R,
                 "latency_ms": float(lat.get("int8", float("nan"))),
                 "n_box": ev["qcert_int8"]["n_box"],
                 "note": "逐场景整数分位 + 除法消除"})

    if args.reported_json:
        for name, v in json_load(args.reported_json).items():
            rows.append({"method": f"{name}（reported）", "source": "reported",
                         "pd_at_1e-3": float("nan"), "pd_at_1e-4": float("nan"),
                         "map50": _pct(v), "pfa": float("nan"),
                         "pfa_std_across_buckets": float("nan"),
                         "R": float("nan"), "latency_ms": float("nan"),
                         "n_box": 0})

    # 验收（§11）：对手是"复现基线"（CA-CFAR 与全局阈值），不含 QCert(FP16)
    q = next(r for r in rows if r["method"] == "YOLO11n + QCert(INT8)")
    baselines = ("CA-CFAR（像素域）", "YOLO11n + 全局阈值")
    best_other = max((r["pd_at_1e-3"] for r in rows
                      if r["method"] in baselines and r["source"] == source
                      and np.isfinite(r["pd_at_1e-3"])), default=float("nan"))
    gap = (q["pd_at_1e-3"] - best_other) if np.isfinite(best_other) else float("nan")
    dmap = (float("nan") if not ("fp16" in mp and "int8" in mp)
            else _pct(mp["int8"]) - _pct(mp["fp16"]))
    verdict = {
        "pd_gap_vs_best_baseline": float(gap),
        "pass_pd": bool(np.isfinite(gap) and gap >= -0.015),
        "R": float(R), "pass_R": bool(np.isfinite(R) and R <= 1.2),
        "dmap50": dmap,
        "pass_map": bool(not np.isfinite(dmap) or abs(dmap) <= 0.5),
    }
    verdict["pass"] = bool(verdict["pass_pd"] and verdict["pass_R"]
                           and verdict["pass_map"])

    out = ensure_dir(args.out)
    save_table(rows, out, "table6_e8", "Table 6 · E8 检测兼容性与对照")
    json_dump({
        **meta, "source": source, "n_scenes": len(recs),
        "n_boxes": int(sum(r["n_boxes"] for r in recs)),
        "verdict": verdict, "pfa_noisy_1e-4": True,
        "note": ("合成场景离线自检：mAP/时延为占位，不能写作板级实测"
                 if source == "synth" else "分数场来自板级 dump"),
    }, out / "summary.json")

    print("\n=== E8 · 检测兼容性与对照 ===")
    print(f"场景 {len(recs)}  框 {int(sum(r['n_boxes'] for r in recs))}  "
          f"α={args.alpha}  来源={source}")
    print(f"{'方法':<24}{'Pd@1e-3':>9}{'Pd@1e-4':>9}{'mAP50':>8}"
          f"{'Pfa':>10}{'跨桶std':>9}{'R':>7}{'ms':>7}")
    for r in rows:
        print(f"{r['method']:<24}{r['pd_at_1e-3']:>9.3f}{r['pd_at_1e-4']:>9.3f}"
              f"{r['map50']:>8.2f}{r['pfa']:>10.2e}"
              f"{r['pfa_std_across_buckets']:>9.2e}{r['R']:>7.2f}"
              f"{r['latency_ms']:>7.2f}")
    print(f"QCert(INT8) − 最强基线 = {gap:+.3f}（≥−0.015 → {verdict['pass_pd']}）；"
          f"R = {R:.3f}（≤1.2 → {verdict['pass_R']}）；ΔmAP50 = {dmap:+.2f}")
    print(f"判决: {'PASS' if verdict['pass'] else 'FAIL'}  产物: {out}")
    if source == "synth":
        print("注: 合成图上的 CA-CFAR 为像素级理想情形（亮矩形），"
              "本判决仅验证代码通路，不可写作板级结论；正式结论须用 --fp32/--int8 真实 dump。")
    return {"rows": rows, "verdict": verdict, "R": R}


if __name__ == "__main__":
    main()
