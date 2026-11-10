"""E4 · σ 量化粒度四配置，验证预注册预言 P1。

协议依据 paper/04 §7。P1（可证伪）：把 σ 改成**对数域 + 逐通道 scale** 后，
``Δln P_fa`` 相对"线性 + 逐张量"应当下降 **≥ 5 倍**。

为什么这会成立：``Δln P_fa ≈ z²·ε_σ``，而 ``ε_σ(u) = Δ_σ / (2 σ(u))``。
逐张量线性量化时 ``Δ_σ ∝ max σ``（即 median σ），低 σ 区域的相对误差被放大；
对数域让相对误差与 σ 无关，逐通道再把块间动态范围拆开。

**统计功效**（R12 的教训）：α=1e-3 时 P_fa 由尾部计数决定，n=6e4 只有 ~30 个计数，
``dln`` 自身的抖动就有 ±0.15，比要测的效应还大。因此本脚本
1) 标定样本与测量样本**独立抽样**（各 ``--n`` 个），互不泄漏；
2) 量化器步长只在标定样本上定一次（AMCT 语义），再作用到测量样本；
3) 多个 seed 的 ``dln`` 取**有符号均值**（独立噪声相互抵消），并报告 std。

    python -m experiments.run_e4 --n 1000000 --reps 3 --eps-sigma 0.05
    python -m experiments.run_e4 --dist t3 --p1-factor 5
"""
from __future__ import annotations

import argparse

import numpy as np

from qcert.qint import dequantize, quantize

from .common import RESULTS, ensure_dir, json_dump, save_table
from .synth import (actual_eps_sigma, apply_sigma_steps, clutter_sample,
                    sigma_steps)

# 配置 -> (mode, block_px)；block_px=None 即"逐张量一条 scale"
CONFIGS = {
    "linear_tensor": ("linear", None),
    "log_tensor": ("log", None),
    "channel_linear": ("linear", 64.0),
    "channel_log": ("log", 64.0),
    "fp16": ("fp16", None),
}
# P1 的"配置 2/3"（04 §13.1-5）：对数域 / 逐通道
P1_CONFIGS = ("log_tensor", "channel_linear", "channel_log")


def _grid_centers(n: int, step_px: float = 8.0):
    w = int(np.ceil(np.sqrt(n)))
    idx = np.arange(n)
    xy = np.stack([(idx % w) * step_px, (idx // w) * step_px], axis=1)
    return xy.astype(np.float64)


def _roundtrip(x, delta, bits):
    return x if delta <= 0 else dequantize(quantize(x, delta, bits=bits), delta)


def run_once(n_cal, n_meas, alpha, eps_sigma, eps_s, eps_mu, bits, seed, dist, cfg):
    mode, block_px = CONFIGS[cfg]

    # 两批独立样本：标定用 cal，测量用 meas
    s_c, mu_c, sig_c = clutter_sample(n_cal, dist=dist, seed=seed)
    s_m, mu_m, sig_m = clutter_sample(n_meas, dist=dist, seed=seed + 50_007)
    cen_c, cen_m = _grid_centers(n_cal), _grid_centers(n_meas)

    # 量化器参数只在标定样本上定（AMCT：离线标定一次，部署不再变）
    ref = float(np.median(sig_c))
    ds, dmu = 2.0 * eps_s * ref, 2.0 * eps_mu * ref
    steps = sigma_steps(sig_c, (None if block_px is None else cen_c),
                        eps_sigma, block_px or 64.0, mode)
    sig_cq = apply_sigma_steps(sig_c, steps, cen_c, block_px or 64.0)
    sig_mq = apply_sigma_steps(sig_m, steps, cen_m, block_px or 64.0)

    # λ 在 FP32 标定侧取（部署路径：阈值由浮点标定，算术在定点执行）
    z_c = (s_c - mu_c) / np.maximum(sig_c, 1e-6)
    lam_fp = float(np.quantile(z_c, 1.0 - alpha))

    z_fp = (s_m - mu_m) / np.maximum(sig_m, 1e-6)
    z_q = ((_roundtrip(s_m, ds, bits) - _roundtrip(mu_m, dmu, bits))
           / np.maximum(sig_mq, 1e-6))

    pfa_fp = float(np.mean(z_fp > lam_fp))
    pfa_q = float(np.mean(z_q > lam_fp))
    dln = (float(np.log(pfa_q) - np.log(pfa_fp))
           if pfa_q > 0 and pfa_fp > 0 else float("nan"))
    eps_u = actual_eps_sigma(sig_m, sig_mq)
    return {
        "config": cfg, "eps_sigma": float(eps_sigma),
        "eps_sigma_median": float(np.nanmedian(eps_u)),
        "eps_sigma_p95": float(np.nanpercentile(eps_u, 95)),
        "pfa_fp32": pfa_fp, "pfa_int_route": pfa_q,
        "ratio": (pfa_q / pfa_fp) if pfa_fp > 0 else float("inf"),
        "dln_meas": dln,
        "dln_pred": float(lam_fp ** 2 * float(eps_sigma)),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=1_000_000,
                    help="标定/测量各自的样本数（两批独立抽样）")
    ap.add_argument("--alpha", type=float, default=1e-3)
    ap.add_argument("--eps-sigma", type=float, nargs="+", default=[0.02, 0.05, 0.10])
    ap.add_argument("--eps-s", type=float, default=0.02)
    ap.add_argument("--eps-mu", type=float, default=0.02)
    ap.add_argument("--dist", default="gauss", choices=["gauss", "t3"])
    ap.add_argument("--bits", type=int, default=16)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--p1-factor", type=float, default=5.0)
    ap.add_argument("--out", default=str(RESULTS / "e4"))
    args = ap.parse_args(argv)

    rows = []
    for eps in args.eps_sigma:
        for cfg in CONFIGS:
            reps = [run_once(args.n, args.n, args.alpha, eps, args.eps_s,
                             args.eps_mu, args.bits, 1000 * s, args.dist, cfg)
                    for s in range(args.reps)]
            dlns = np.asarray([r["dln_meas"] for r in reps], dtype=np.float64)
            dlns = dlns[np.isfinite(dlns)]
            rows.append({
                "eps_sigma": float(eps), "config": cfg,
                "eps_sigma_median": float(np.nanmean([r["eps_sigma_median"] for r in reps])),
                "eps_sigma_p95": float(np.nanmean([r["eps_sigma_p95"] for r in reps])),
                "pfa_fp32": float(np.nanmean([r["pfa_fp32"] for r in reps])),
                "pfa_int_route": float(np.nanmean([r["pfa_int_route"] for r in reps])),
                # 有符号均值：不同 seed 的尾部计数噪声相互抵消
                "dln_meas": float(dlns.mean()) if dlns.size else float("nan"),
                "dln_std": float(dlns.std(ddof=1)) if dlns.size > 1 else float("nan"),
                "n_reps": int(dlns.size),
                "dln_pred": float(np.nanmean([r["dln_pred"] for r in reps])),
            })

    # P1：配置 2/3 里最好的一个，相对"线性·逐张量"的下降倍数
    p1 = []
    for eps in args.eps_sigma:
        d = {r["config"]: r["dln_meas"] for r in rows if r["eps_sigma"] == eps}
        e = {r["config"]: r["eps_sigma_median"] for r in rows if r["eps_sigma"] == eps}
        base = abs(d.get("linear_tensor", float("nan")))
        cand = {k: abs(d[k]) for k in P1_CONFIGS if k in d}
        best_name = min(cand, key=lambda k: cand[k]) if cand else ""
        best = cand.get(best_name, float("nan"))
        drop = (base / best) if (np.isfinite(base) and best > 0) else float("inf")
        p1.append({
            "eps_sigma": float(eps),
            "dln_linear_tensor": d.get("linear_tensor", float("nan")),
            "best_config": best_name,
            "dln_best": d.get(best_name, float("nan")),
            "eps_med_linear": e.get("linear_tensor", float("nan")),
            "eps_med_best": e.get(best_name, float("nan")),
            "drop_factor": float(drop),
            # 基线本身落在噪声里（|dln| 与 dln_std 同量级）时，倍数没有意义
            "informative": bool(np.isfinite(base) and base >= 0.05),
            "P1_holds": bool(np.isfinite(drop) and drop >= args.p1_factor),
        })

    # 诊断：|Δln P_fa| 对 ε_σ 的伸缩指数。≈1 说明一阶（系统偏置）主导，
    # ≈2 说明对称舍入的一阶项相消、只剩二阶 —— 直接决定 Prop.2 的适用口径。
    scaling = {}
    for cfg in CONFIGS:
        pts = [(np.log(r["eps_sigma"]), np.log(abs(r["dln_meas"])))
               for r in rows
               if r["config"] == cfg and abs(r["dln_meas"]) >= 1e-3]
        if len(pts) >= 2:
            x, y = np.asarray(pts).T
            scaling[cfg] = float(np.polyfit(x, y, 1)[0])

    out = ensure_dir(args.out)
    save_table(rows, out, "table3_e4", "Table 3 上 · E4 σ 量化粒度")
    save_table(p1, out, "e4_p1", "E4 · P1 可证伪预言检验")
    summary = {
        "alpha": args.alpha, "n_cal": args.n, "n_meas": args.n,
        "dist": args.dist, "bits": args.bits, "reps": args.reps,
        "p1_factor": args.p1_factor, "p1": p1,
        "dln_scaling_exponent": scaling,
        "P1_all": bool(all(r["P1_holds"] for r in p1)),
        "P1_all_informative": bool(all(r["P1_holds"] for r in p1
                                       if r["informative"])),
    }
    json_dump(summary, out / "summary.json")

    print("\n=== E4 · σ 粒度 ===")
    print(f"{'eps':>6}{'config':>16}{'eps_med':>9}{'eps_p95':>9}"
          f"{'ratio':>8}{'dln_meas':>10}{'dln_std':>9}{'dln_pred':>10}")
    for r in rows:
        print(f"{r['eps_sigma']:>6.3f}{r['config']:>16}"
              f"{r['eps_sigma_median']:>9.4f}{r['eps_sigma_p95']:>9.4f}"
              f"{(r['pfa_int_route'] / r['pfa_fp32']):>8.3f}"
              f"{r['dln_meas']:>10.4f}{r['dln_std']:>9.4f}{r['dln_pred']:>10.4f}")
    for r in p1:
        print(f"  P1 @eps={r['eps_sigma']:.2f}: |{r['dln_linear_tensor']:.4f}| / "
              f"|{r['dln_best']:.4f}|({r['best_config']}) = "
              f"{r['drop_factor']:.2f}× → {'PASS' if r['P1_holds'] else 'FAIL'}"
              f"   eps_med {r['eps_med_linear']:.4f}→{r['eps_med_best']:.4f}")
    if scaling:
        print("  |Δln P_fa| ∝ ε_σ^p : " +
              "  ".join(f"{k}={v:.2f}" for k, v in scaling.items()))
    print(f"产物: {out}")
    return summary


if __name__ == "__main__":
    main()
