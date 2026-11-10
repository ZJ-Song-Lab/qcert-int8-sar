"""E6/E4 补充实验（S3/M2）：

E6 部分 —— P99 基线方差与双运行归属：
  两份 e6_windows.csv（code/results/e6 = INT8 复跑；asc-sar/code/results/e6
  = FP16 板级带传感器）逐窗重算 p99_drift_rel = (p99 - t_model)/t_model，
  与各自 summary 对照确认归属；并给出 t_model 跨窗标准差（热稳定基线）。

E4 部分 —— 标度指数按 |Δln| 重拟合（M2）：
  原拟合（results/e4/summary.json 的 dln_scaling_exponent）把 linear_tensor
  的 ε=0.02 点（Δln=-3.2e-4，符号为负）剔除后只用 2 点定指数；本脚本：
  - 用全部 3 点 + |Δln| 做 log-log 最小二乘；
  - 与"仅信息点"（|Δln| > fp16 对照噪声底 0.004877）拟合对照；
  - 给出各配置两种口径的指数，供论文如实标注。

    python -m experiments.run_e6_e4_supp
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .common import RESULTS, ensure_dir, json_dump

ROOT = Path(__file__).resolve().parents[2]
E6_CSV = {
    "int8_rerun": ROOT / "code" / "results" / "e6" / "e6_windows.csv",
    "fp16_board": ROOT / "asc-sar" / "code" / "results" / "e6" / "e6_windows.csv",
}
E4_CSV = ROOT / "code" / "results" / "e4" / "table3_e4.csv"
FP16_FLOOR = 0.004877          # fp16 对照在三个 ε 下的 Δln（噪声底，std≈0.0059）
CONFIGS = ("linear_tensor", "log_tensor", "channel_linear", "channel_log")


def _read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def fit_exponent(eps, dln):
    """log-log 最小二乘：ln|Δln| = p·ln ε + b。"""
    x = np.log(np.asarray(eps, dtype=float))
    y = np.log(np.abs(np.asarray(dln, dtype=float)))
    p, b = np.polyfit(x, y, 1)
    yhat = p * x + b
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(p), float(b), r2


def main():
    out = ensure_dir(RESULTS / "e6_e4_supp")

    # ---------- E6 ----------
    e6 = {}
    for tag, path in E6_CSV.items():
        rows = _read_csv(path)
        p99 = np.array([float(r["p99_ms"]) for r in rows])
        tmod = np.array([float(r["t_model_ms"]) for r in rows])
        drift = (p99 - tmod) / tmod
        e6[tag] = {
            "n_windows": len(rows),
            "p99_drift_rel_per_window": [float(x) for x in drift],
            "p99_drift_rel_mean": float(drift.mean()),
            "p99_drift_rel_max": float(drift.max()),
            "t_model_mean_ms": float(tmod.mean()),
            "t_model_std_ms": float(tmod.std()),
            "t_model_cv": float(tmod.std() / tmod.mean()),
            "p99_ms_range": [float(p99.min()), float(p99.max())],
        }
        if rows[0].get("temp_c") not in (None, "", "nan"):
            temp = np.array([float(r["temp_c"]) for r in rows])
            e6[tag]["temp_c"] = [float(temp.min()), float(temp.max())]
            pow_ = np.array([float(r["power_w"]) for r in rows])
            e6[tag]["power_w"] = [float(pow_.min()), float(pow_.max())]

    # ---------- E4 ----------
    rows = _read_csv(E4_CSV)
    by_cfg = {c: {"eps": [], "dln": []} for c in CONFIGS}
    for r in rows:
        c = r["config"]
        if c in by_cfg:
            by_cfg[c]["eps"].append(float(r["eps_sigma"]))
            by_cfg[c]["dln"].append(float(r["dln_meas"]))

    e4 = {}
    eps_all = by_cfg["linear_tensor"]["eps"]
    for c in CONFIGS:
        eps = np.array(by_cfg[c]["eps"])
        dln = np.array(by_cfg[c]["dln"])
        p_all, b_all, r2_all = fit_exponent(eps, dln)
        keep = np.abs(dln) > FP16_FLOOR
        p_inf, b_inf, r2_inf = (fit_exponent(eps[keep], dln[keep])
                                if keep.sum() >= 2 else (float("nan"),) * 3)
        e4[c] = {
            "exponent_abs_all3": p_all, "r2_all": r2_all,
            "exponent_informative_only": p_inf, "r2_informative": r2_inf,
            "n_informative": int(keep.sum()),
            "dln": [float(x) for x in dln],
        }

    json_dump({"e6": e6, "e4": e4, "fp16_floor": FP16_FLOOR,
               "note": "exponent_abs_all3=|Δln| 全 3 点 log-log 拟合；"
                       "exponent_informative_only=剔除 |Δln|≤fp16 噪声底的点"},
              out / "e6_e4_supp_summary.json")

    print("=== E6 · P99 漂移逐窗 ===")
    for tag, d in e6.items():
        print(f"[{tag}] n={d['n_windows']} "
              f"p99_drift mean={d['p99_drift_rel_mean']:.5f} max={d['p99_drift_rel_max']:.5f} "
              f"t_model={d['t_model_mean_ms']:.2f}±{d['t_model_std_ms']:.3f}ms "
              f"(CV={d['t_model_cv']*100:.2f}%)")
        if "temp_c" in d:
            print(f"    temp {d['temp_c'][0]:.1f}–{d['temp_c'][1]:.1f}°C, "
                  f"power {d['power_w'][0]:.2f}–{d['power_w'][1]:.2f}W")

    print("\n=== E4 · 标度指数重拟合 ===")
    print(f"{'config':>16}{'p(|Δ|全3点)':>14}{'R²':>8}{'p(信息点)':>12}"
          f"{'R²':>8}{'信息点数':>9}")
    for c in CONFIGS:
        d = e4[c]
        print(f"{c:>16}{d['exponent_abs_all3']:>14.3f}{d['r2_all']:>8.3f}"
              f"{d['exponent_informative_only']:>12.3f}{d['r2_informative']:>8.3f}"
              f"{d['n_informative']:>9}")
    print(f"产物: {out}")


if __name__ == "__main__":
    main()
