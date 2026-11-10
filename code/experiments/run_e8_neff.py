"""E8 补充实验：标定/测量池的有效独立样本数 n_eff（F2 定量交代）。

F2 指出：α 的可测性受 n_eff 约束（α ≥ 1/n_eff）。E8 在 α=1e-4 下只能跨场景
合并报数（summary 已标 pfa_noisy_1e-4），但缺少数值交代。本脚本按 E8 原协议
（build_omega, keep=0.5, block_px=64, guard=8, seed=逐场景序号）重建每个场景
的 Omega_cal / Omega_meas，用 qcert.pfa_eval.n_eff 统计有效独立块数：

- 每场景 n_eff(cal)、n_eff(meas)、n_cal、n_meas；
- 分布 median / p10 / p90 / min；
- 可测性判据：α=1e-3 需要 n_eff ≳ 1e3，α=1e-4 需要 n_eff ≳ 1e4，
  报告满足比例（按块抽样独立性口径）。

    python -m experiments.run_e8_neff
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from qcert.pfa_eval import n_eff

from .common import RESULTS, ensure_dir, json_dump

FP16_DIR = Path(__file__).resolve().parents[2] / "asc-sar" / "results" / "scores_ssdd_fp16"
INT8_DIR = Path(__file__).resolve().parents[2] / "asc-sar" / "results" / "scores_ssdd_int8"


def main():
    out = ensure_dir(RESULTS / "e8")
    # 与 run_e8._dump_scenes 相同的 Omega 重建几何（keep=0.5, block=64, guard=8）
    from qcert.pfa_eval import boxes_to_letterbox, build_omega, cell_centers
    from qcert.scores import load_dir

    d16 = load_dir(FP16_DIR)
    stems = sorted(d16.keys())
    rows = []
    for i, s in enumerate(stems):
        f = d16[s]
        boxes = np.asarray((f.meta or {}).get("boxes", []),
                           dtype=np.float64).reshape(-1, 4)
        centers = (np.asarray(f.centers, dtype=np.float64)
                   if f.centers is not None
                   else cell_centers(f.head_sizes, int(f.meta.get("imgsz", 640))))
        lb = boxes_to_letterbox(boxes, f.meta) if len(boxes) else boxes
        mask_cal, mask_meas, info = build_omega(
            centers, lb, guard_px=8.0, block_px=64.0, keep=0.5, seed=i)
        rows.append({
            "stem": s,
            "n_cal": int(mask_cal.sum()), "n_meas": int(mask_meas.sum()),
            "n_eff_cal": n_eff(mask_cal, centers, block_px=64.0),
            "n_eff_meas": n_eff(mask_meas, centers, block_px=64.0),
            "n_blocks": info["n_blocks"],
        })

    with open(out / "e8_neff_per_scene.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    neff_cal = np.array([r["n_eff_cal"] for r in rows], dtype=float)
    neff_meas = np.array([r["n_eff_meas"] for r in rows], dtype=float)
    nmeas = np.array([r["n_meas"] for r in rows], dtype=float)

    def _dist(v):
        return {"min": float(v.min()), "p10": float(np.percentile(v, 10)),
                "median": float(np.median(v)), "p90": float(np.percentile(v, 90)),
                "max": float(v.max())}

    summary = {
        "n_scenes": len(rows),
        "block_px": 64.0, "keep": 0.5, "guard_px": 8.0,
        "n_eff_cal": _dist(neff_cal), "n_eff_meas": _dist(neff_meas),
        "n_meas": _dist(nmeas),
        "measurability": {
            "alpha_1e-3_needs_1e3": {
                "frac_scenes_neff_meas_ge_1000": float(np.mean(neff_meas >= 1000)),
            },
            "alpha_1e-4_needs_1e4": {
                "frac_scenes_neff_meas_ge_10000": float(np.mean(neff_meas >= 10000)),
            },
            "pfa_granularity_per_scene": float(np.median(1.0 / nmeas)),
        },
        "note": "n_eff=被选中的独立块数（块内样本强相关，不独立计数）；"
                "逐场景 P_fa 的量化步长为 1/n_meas",
    }
    json_dump(summary, out / "e8_neff_summary.json")

    print("=== E8 · n_eff 统计 ===")
    print(f"场景数 {len(rows)}")
    for k in ("n_eff_cal", "n_eff_meas", "n_meas"):
        d = summary[k]
        print(f"{k:>12}: min={d['min']:.0f} p10={d['p10']:.0f} "
              f"median={d['median']:.0f} p90={d['p90']:.0f} max={d['max']:.0f}")
    m = summary["measurability"]
    print(f"α=1e-3 可测(n_eff≥1e3)场景比例: {m['alpha_1e-3_needs_1e3']['frac_scenes_neff_meas_ge_1000']:.3f}")
    print(f"α=1e-4 可测(n_eff≥1e4)场景比例: {m['alpha_1e-4_needs_1e4']['frac_scenes_neff_meas_ge_10000']:.3f}")
    print(f"逐场景 P_fa 量化步长中位数: {m['pfa_granularity_per_scene']:.5f}")
    print(f"产物: {out}")


if __name__ == "__main__":
    main()
