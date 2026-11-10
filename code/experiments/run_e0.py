"""E0 存在性实验：R = P_fa(INT8) / P_fa(FP32) ≥ 1.5 且 |ΔmAP50| < 1 点。

这是论文的**硬门**（paper/05 §3 M1）：R 上不去，v3 主线不成立。

输入是两条链路在同一批场景上的稠密分数场 dump：

    python -m experiments.run_e0 \
        --fp32  dumps/fp32 --int8 dumps/int8 \
        --labels asc-sar/datasets/ssdd/labels/test \
        --alpha 1e-3 --map-json results/e0/metrics.json

``--map-json`` 可选，格式 ``{"fp32": 0.973, "int8": 0.967}``（来自 acl_infer 的 mAP，
单位是百分制点数或 0–1 都可以，脚本按 0–1 与百分制自动区分）。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from qcert.pfa_eval import bootstrap_ratio_ci, deviation_from_alpha, pfa_ratio

from .common import (RESULTS, aggregate, ensure_dir, group_of, gt_for,
                     json_dump, json_load, load_pairs, save_table,
                     scene_pfa_pair)


def _as_pct(v):
    return float(v) * 100.0 if 0.0 <= float(v) <= 1.0 else float(v)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fp32", required=True, help="FP32 链路的分数场 dump 目录")
    ap.add_argument("--int8", required=True, help="INT8 链路的分数场 dump 目录")
    ap.add_argument("--labels", default=None, help="YOLO 标签目录（排除 GT 区域）")
    ap.add_argument("--alpha", type=float, default=1e-3)
    ap.add_argument("--guard-px", type=float, default=8.0)
    ap.add_argument("--block-px", type=float, default=64.0)
    ap.add_argument("--keep", type=float, default=0.5,
                    help="块抽样比例。alpha=1e-3 需要 n_meas >= 1e4；"
                         "E9 item3 实测 keep=1/16 就有 0.44 decade 的 α 误差")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--map-json", default=None, help='{"fp32":..,"int8":..}')
    ap.add_argument("--out", default=str(RESULTS / "e0"))
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args(argv)

    pairs = load_pairs(args.fp32, args.int8)
    rows = []
    for stem, ref, test in pairs:
        boxes = gt_for(stem, args.labels, ref.meta) if args.labels else None
        r = scene_pfa_pair(ref, test, boxes, args.alpha, args.guard_px,
                           args.block_px, args.keep, args.seed)
        r["sea"] = group_of(ref, "sea")
        r["dev_ref"] = deviation_from_alpha(r["pfa_ref"], args.alpha)
        r["dev_test"] = deviation_from_alpha(r["pfa_test"], args.alpha)
        rows.append(r)

    pfa_r = [r["pfa_ref"] for r in rows]
    pfa_t = [r["pfa_test"] for r in rows]
    lo, hi, point = bootstrap_ratio_ci(pfa_t, pfa_r, args.n_boot, args.seed)

    agg_r = aggregate(rows, "pfa_ref")
    agg_t = aggregate(rows, "pfa_test")
    R_mean = pfa_ratio(agg_t["mean"], agg_r["mean"])

    dmap = float("nan")
    map_note = "未提供 --map-json"
    if args.map_json:
        m = json_load(args.map_json)
        if "fp32" in m and "int8" in m:
            dmap = _as_pct(m["int8"]) - _as_pct(m["fp32"])
            map_note = f"mAP50 {_as_pct(m['fp32']):.2f} -> {_as_pct(m['int8']):.2f}"
        else:
            map_note = "map-json 缺少 fp32/int8 键"

    summary = {
        "n_scenes": len(rows),
        "alpha": args.alpha,
        # 采样口径必须随结果一起落盘：keep 变了 R 会变（见 code/README.md）
        "keep": args.keep, "guard_px": args.guard_px, "block_px": args.block_px,
        "n_meas": int(sum(r["n_meas"] for r in rows)),
        "insufficient": bool(sum(r["n_meas"] for r in rows) * args.alpha < 10.0),
        "source": f"{args.fp32} | {args.int8}",
        "pfa_fp32": agg_r["mean"],
        "pfa_int8": agg_t["mean"],
        "R": R_mean,
        "R_median": aggregate(rows, "R")["median"],
        "R_ci": [lo, hi],
        "R_ci_point": point,
        "dev_from_alpha_fp32": aggregate(rows, "dev_ref")["mean"],
        "dev_from_alpha_int8": aggregate(rows, "dev_test")["mean"],
        "dmap50": dmap,
        "map_note": map_note,
        "pass_R": bool(np.isfinite(R_mean) and R_mean >= 1.5),
        "pass_map": bool(not np.isfinite(dmap) or abs(dmap) < 1.0),
        "verdict": ("GO" if np.isfinite(R_mean) and R_mean >= 1.5
                    and (not np.isfinite(dmap) or abs(dmap) < 1.0) else "NO-GO"),
    }

    out = ensure_dir(args.out)
    save_table(rows, out, "e0_per_scene", "E0 逐场景 P_fa 与 R")
    table1 = [{
        "n_scenes": summary["n_scenes"], "alpha": args.alpha,
        "pfa_fp32": agg_r["mean"], "pfa_int8": agg_t["mean"],
        "R": R_mean, "R_lo": lo, "R_hi": hi,
        "R_median": summary["R_median"],
        "dmap50": dmap, "pass_R": summary["pass_R"],
        "pass_map": summary["pass_map"], "verdict": summary["verdict"],
    }]
    save_table(table1, out, "table1_e0", "Table 1 · E0 存在性")
    json_dump(summary, out / "summary.json")

    print("\n=== E0 ===")
    print(f"场景数 {len(rows)}  alpha={args.alpha}  keep={args.keep}  "
          f"n_meas={summary['n_meas']}")
    if summary["insufficient"]:
        print("[warn] n_meas*alpha < 10：P_fa 卡在 1/n_meas 分辨率上，"
              "R 只能是粗估（加大 --keep 或换更小的 alpha）")
    print(f"P_fa FP32 = {agg_r['mean']:.3e}   INT8 = {agg_t['mean']:.3e}")
    print(f"R = {R_mean:.3f}  95%CI [{lo:.3f}, {hi:.3f}]  中位数 {summary['R_median']:.3f}")
    print(map_note + (f"  Δ={dmap:+.2f} 点" if np.isfinite(dmap) else ""))
    print(f"判决: {summary['verdict']}（R≥1.5 → {summary['pass_R']}；"
          f"|ΔmAP|<1 → {summary['pass_map']}）")
    print(f"产物: {out}")
    return summary


if __name__ == "__main__":
    main()
