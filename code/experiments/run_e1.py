"""E1 · Prop.2 数值验证：ε_σ 扫描下的界 / 实测 / 一阶近似 / AUROC。

协议依据 paper/04 §4.1。合成杂波**逐位置**给出 ``sigma(u)``（块相关 + 白噪声），
这一点是硬要求：全局常数 sigma 会让 ``Delta_sigma`` 退化成一次确定性缩放，
自标定把它完全抵消，曲线是平的（R11）。

三个实测口径要分清：

``dln_recal``
    两条链路各自在 Omega_cal 上重标定 lambda —— 这是"每台设备都重新标"的理想情况，
    理论上应当 ≈ 0（只余有限样本噪声）。
``dln_fixed``
    把 FP32 标定出来的 lambda **直接用在量化后的分数场**上 —— 这是真实部署路径
    （阈值在 FP32 标定、模型在 INT8 跑），也是 Prop.2 要约束的量。
``dln_pred``
    一阶近似 ``z^2 * eps_sigma``；``dln_pred2`` 是零均值舍入下由凸性给出的
    ``z^4 * eps^2 / 6``（见 README「为什么实测常小于一阶近似」）。

    python -m experiments.run_e1 --n 200000 --reps 5 --bits 16
    python -m experiments.run_e1 --dist t3 --mode log --eps-sigma 0.02 0.05
"""
from __future__ import annotations

import argparse

import numpy as np

from qcert.pfa_eval import prop2_bound_pfa, prop2_delta
from qcert.qint import dequantize, quantize

from .common import RESULTS, ensure_dir, json_dump, save_table
from .synth import actual_eps_sigma, clutter_sample, quantize_sigma

SIGNAL_AMP = 4.0              # H1 相对 H0 的平移（以 sigma 为单位）


def _auc(z0, z1):
    """Mann–Whitney 形式 AUC（不依赖 scipy），并列取平均秩。"""
    z0 = np.asarray(z0, dtype=np.float64)
    z1 = np.asarray(z1, dtype=np.float64)
    allv = np.concatenate([z0, z1])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(allv.size, dtype=np.float64)
    ranks[order] = np.arange(1, allv.size + 1, dtype=np.float64)
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    ranks = (np.bincount(inv, weights=ranks) / cnt)[inv]
    n0, n1 = z0.size, z1.size
    return float((ranks[n0:].sum() - n1 * (n1 + 1) / 2.0) / (n0 * n1))


def _roundtrip(x, delta, bits):
    """delta<=0 表示该路不量化（eps=0 的对照点）。"""
    x = np.asarray(x, dtype=np.float64)
    if delta <= 0:
        return x
    return dequantize(quantize(x, delta, bits=bits), delta)


def run_once(n, alpha, eps_sigma, eps_s, eps_mu, bits, seed,
             dist="gauss", mode="linear"):
    s, mu, sigma = clutter_sample(n, dist=dist, seed=seed)
    sig_ref = float(np.median(sigma))
    ds = 2.0 * eps_s * sig_ref
    dmu = 2.0 * eps_mu * sig_ref

    s_q = _roundtrip(s, ds, bits)
    mu_q = _roundtrip(mu, dmu, bits)
    sig_q = quantize_sigma(sigma, eps_sigma, sig_ref, bits=bits, mode=mode)
    eps_u = actual_eps_sigma(sigma, sig_q)

    z_fp = (s - mu) / sigma
    z_q = (s_q - mu_q) / np.maximum(sig_q, 1e-6)

    # H1：同一位置的 sigma/mu 不变，只是把 z 平移 SIGNAL_AMP
    rng = np.random.default_rng(seed + 7717)
    z1 = rng.standard_normal(n if n <= 200_000 else 20_000)
    idx1 = rng.integers(0, n, z1.size)
    s1 = mu[idx1] + sigma[idx1] * (z1 + SIGNAL_AMP)
    s1_q = _roundtrip(s1, ds, bits)
    z1_fp = (s1 - mu[idx1]) / sigma[idx1]
    z1_q = (s1_q - mu_q[idx1]) / np.maximum(sig_q[idx1], 1e-6)

    perm = rng.permutation(n)
    half = n // 2
    cal, meas = perm[:half], perm[half:]

    lam_fp = float(np.quantile(z_fp[cal], 1.0 - alpha))
    lam_q = float(np.quantile(z_q[cal], 1.0 - alpha))
    pfa_fp = float(np.mean(z_fp[meas] > lam_fp))
    pfa_recal = float(np.mean(z_q[meas] > lam_q))
    pfa_fixed = float(np.mean(z_q[meas] > lam_fp))

    pd_fp = float(np.mean(z1_fp > lam_fp))
    pd_fixed = float(np.mean(z1_q > lam_fp))

    def dln(a, b):
        return float(np.log(a) - np.log(b)) if (a > 0 and b > 0) else float("nan")

    b = prop2_bound_pfa(z_fp[cal], lam_fp, sig_ref, ds, dmu,
                        2.0 * eps_sigma * sig_ref)
    zc = lam_fp
    eps_arr = eps_u[np.isfinite(eps_u)]
    eps_med = float(np.median(eps_arr)) if eps_arr.size else float("nan")
    eps_mean = float(np.mean(eps_arr)) if eps_arr.size else float("nan")

    return {
        "eps_sigma": float(eps_sigma),
        "eps_sigma_median": eps_med,
        "eps_sigma_mean": eps_mean,
        "eps_s": float(eps_s),
        "eps_mu": float(eps_mu),
        "dist": dist,
        "mode": mode,
        "lambda_fp32": lam_fp,
        "lambda_int8": lam_q,
        "pfa_fp32": pfa_fp,
        "pfa_int8": pfa_recal,
        "pfa_fixed_lambda": pfa_fixed,
        "ratio": (pfa_fixed / pfa_fp) if pfa_fp > 0 else float("inf"),
        "ratio_recal": (pfa_recal / pfa_fp) if pfa_fp > 0 else float("inf"),
        "dln_recal": dln(pfa_recal, pfa_fp),
        "dln_meas": dln(pfa_fixed, pfa_fp),
        "dln_pred": float(zc ** 2 * float(eps_sigma)),
        "dln_pred2": float(zc ** 4 * float(eps_sigma) ** 2 / 6.0),
        "abs_dpfa": abs(pfa_fixed - pfa_fp),
        "bound_L_delta": float(b["bound"]),
        "L": float(b["L"]),
        "prop2_delta": float(prop2_delta(lam_fp, sig_ref, ds, dmu,
                                         2.0 * eps_sigma * sig_ref)),
        "bound_holds": bool(abs(pfa_fixed - pfa_fp) <= b["bound"] * 1.05 + 1e-12),
        "z_lambda": float(zc),
        "pd_fp32": pd_fp,
        "pd_int8": pd_fixed,
        "auc_fp32": _auc(z_fp[meas], z1_fp),
        "auc_int8": _auc(z_q[meas], z1_q),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--alpha", type=float, default=1e-3)
    ap.add_argument("--eps-sigma", type=float, nargs="+",
                    default=[0.0, 0.01, 0.02, 0.05, 0.10, 0.20])
    ap.add_argument("--eps-s", type=float, default=0.02)
    ap.add_argument("--eps-mu", type=float, default=0.02)
    ap.add_argument("--dist", default="gauss", choices=["gauss", "t3"])
    ap.add_argument("--mode", default="linear", choices=["linear", "log", "fp16"])
    ap.add_argument("--bits", type=int, default=16,
                    help="量化位宽；E1 只验证界的形式，饱和留给 E2")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", default=str(RESULTS / "e1"))
    args = ap.parse_args(argv)

    rows, raw = [], []
    for eps in args.eps_sigma:
        reps = [run_once(args.n, args.alpha, eps, args.eps_s, args.eps_mu,
                         args.bits, seed=s, dist=args.dist, mode=args.mode)
                for s in range(args.reps)]
        raw.extend([dict(r, rep=i) for i, r in enumerate(reps)])
        keys = ("pfa_fp32", "pfa_int8", "pfa_fixed_lambda", "ratio", "ratio_recal",
                "dln_recal", "dln_meas", "dln_pred", "dln_pred2",
                "bound_L_delta", "prop2_delta", "pd_fp32", "pd_int8",
                "auc_fp32", "auc_int8", "eps_sigma_median")
        avg = {k: float(np.nanmean([r[k] for r in reps])) for k in keys}
        avg.update({
            "eps_sigma": float(eps),
            "dist": args.dist,
            "mode": args.mode,
            "lambda_fp32": float(np.mean([r["lambda_fp32"] for r in reps])),
            "bound_holds": bool(all(r["bound_holds"] for r in reps)),
            "abs_dpfa": float(np.mean([r["abs_dpfa"] for r in reps])),
            "dauroc": float(avg["auc_int8"] - avg["auc_fp32"]),
            "pred_over_meas": (float(avg["dln_pred"] / avg["dln_meas"])
                               if abs(avg["dln_meas"]) > 1e-9 else float("nan")),
        })
        rows.append(avg)

    out = ensure_dir(args.out)
    save_table(rows, out, "e1_eps_sweep", "E1 · ε_σ 扫描（Prop.2 界 / 实测 / 一阶预言）")
    save_table(raw, out, "e1_raw", "E1 原始重复")

    fig = [{"eps_sigma": r["eps_sigma"], "eps_sigma_median": r["eps_sigma_median"],
            "meas": r["dln_meas"], "pred": r["dln_pred"],
            "pred2": r["dln_pred2"], "bound": r["bound_L_delta"],
            "dauroc": r["dauroc"]} for r in rows]
    save_table(fig, out, "fig4_data", "Fig.4 · Δln P_fa 对比")
    json_dump({"alpha": args.alpha, "n": args.n, "dist": args.dist,
               "mode": args.mode, "rows": rows}, out / "summary.json")

    print("\n=== E1 · Prop.2 ===")
    print(f"alpha={args.alpha}  n={args.n}  bits={args.bits}  "
          f"dist={args.dist}  mode={args.mode}")
    print(f"{'eps_sigma':>10}{'eps_u_med':>10}{'pfa_fp32':>12}{'pfa_fix':>12}"
          f"{'ratio':>8}{'dln_meas':>10}{'dln_pred':>10}{'bound':>10}{'holds':>7}")
    for r in rows:
        print(f"{r['eps_sigma']:>10.3f}{r['eps_sigma_median']:>10.4f}"
              f"{r['pfa_fp32']:>12.3e}{r['pfa_fixed_lambda']:>12.3e}"
              f"{r['ratio']:>8.3f}{r['dln_meas']:>10.4f}"
              f"{r['dln_pred']:>10.4f}{r['bound_L_delta']:>10.4f}"
              f"{str(r['bound_holds']):>7}")
    print(f"产物: {out}")
    return rows


if __name__ == "__main__":
    main()
