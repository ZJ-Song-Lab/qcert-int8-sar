# -*- coding: utf-8 -*-
"""Generate paper figures Fig.1-5, Fig.7 from code/results data.

Outputs to paper/figs/ (png + pdf). Data sources: results/e0,e1,e2,e3,e7.
Fig.6 (failure-case image crops) is NOT generated here: raw SAR images are not
in the repo; only score-field npz dumps. See paper/01_论文大纲.md.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIGS = ROOT.parent / "paper" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

# colorblind-safe palette
C_FP16, C_INT8, C_ACC, C_RED, C_GRAY = "#0072B2", "#E69E00", "#009E73", "#D55E00", "#7F7F7F"
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.bottom": False, "axes.spines.left": False,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "DejaVu Sans",
})


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"{name}.{ext}")
    plt.close(fig)
    print("saved", name)


# ---------------------------------------------------------------- Fig.1 (E0)
def fig1():
    df = pd.read_csv(RES / "e0" / "e0_per_scene.csv")
    rng = np.random.default_rng(0)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.8, 3.4), gridspec_kw={"width_ratios": [2.1, 1]})

    for x, col, lab, c in [(0.0, "pfa_ref", "FP16 OM", C_FP16), (1.0, "pfa_test", "INT8 OM", C_INT8)]:
        y = df[col].to_numpy()
        a1.scatter(x + rng.uniform(-0.13, 0.13, len(y)), y, s=7, alpha=0.45, color=c, label=lab, zorder=2)
        bp = a1.boxplot(y, positions=[x], widths=0.22, patch_artist=True, showfliers=False, zorder=3)
        for p in bp["boxes"]:
            p.set(facecolor="white", edgecolor=c, alpha=0.9)
        for m in bp["medians"]:
            m.set(color=c, lw=1.6)
    a1.axhline(1e-3, color=C_RED, ls="--", lw=1.0, zorder=1)
    a1.text(1.42, 1.06e-3, r"nominal $\alpha=10^{-3}$", color=C_RED, fontsize=7.5, ha="right")
    a1.set_yscale("log"); a1.set_ylim(8e-5, 5e-2)
    a1.set_xticks([0, 1]); a1.set_xticklabels(["FP16 OM", "INT8 OM"])
    a1.set_ylabel(r"per-scene $P_{fa}$ (log)")
    a1.set_title("(a) Same model, same test set: per-scene $P_{fa}$")
    a1.text(0.5, 3.1e-2, "$R=0.969$, 95% CI $[0.901,1.033]$   |   median $=1.0$, $n=232$ scenes",
            ha="center", fontsize=7.5, color="black")

    m = [98.27, 97.98]
    bars = a2.bar(["FP16 OM", "INT8 OM"], m, width=0.55, color=[C_FP16, C_INT8], alpha=0.85)
    a2.set_ylim(96.5, 99.0)
    a2.set_ylabel("mAP50 (%)")
    a2.set_title("(b) mAP50 (zoomed axis)")
    for b, v in zip(bars, m):
        a2.text(b.get_x() + b.get_width() / 2, v + 0.06, f"{v:.2f}", ha="center", fontsize=8)
    a2.annotate(r"$\Delta=-0.29$ pt (statistically indistinguishable)",
                xy=(1, 97.98), xytext=(0.42, 97.2), fontsize=7.5,
                arrowprops=dict(arrowstyle="->", lw=0.8, color=C_GRAY))
    fig.tight_layout()
    save(fig, "fig1_existence")


# --------------------------------------------------------------- Fig.2 (pipeline)
def box(ax, xy, w, h, text, fc, ec="#333333", fs=7.5, lw=1.0, style="round,pad=0.02", tc="black"):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle=style, fc=fc, ec=ec, lw=lw))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fs, color=tc)


def arrow(ax, p1, p2, color="#333333", lw=1.2, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=11,
                                 color=color, lw=lw, linestyle=ls, shrinkA=1, shrinkB=1))


def fig2():
    fig, ax = plt.subplots(figsize=(7.0, 3.1))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")
    ax.text(0.1, 4.78, "online inference loop (NPU, host-free)", fontsize=8, color=C_INT8, weight="bold")

    box(ax, (0.1, 3.9), 1.25, 0.62, "SAR scene $x$\n$640{\\times}640$", "#EAEAEA")
    box(ax, (1.95, 3.55), 2.35, 1.3,
        "single static .om (int8)\ndetector $f_\\theta$ + null model\n$\\rightarrow s_q,\\ \\mu_q,\\ \\sigma_q$ (int codes)",
        "#D6E9F8", fs=7.2)
    box(ax, (5.0, 3.55), 2.5, 1.3,
        "integer decision layer (Eq. 21)\n$r\\,s_q-b\\,\\mu_q > c\\,\\sigma_q$\nno div / no float / int64",
        "#D9F0E5", fs=7.2)
    box(ax, (8.2, 3.9), 1.5, 0.62, "$\\hat{y}$ bitmap\n$P_{fa}\\approx\\alpha$", "#F6E3C5")

    arrow(ax, (1.35, 4.21), (1.95, 4.21))
    arrow(ax, (4.30, 4.21), (5.00, 4.21))
    arrow(ax, (7.50, 4.21), (8.20, 4.21))
    arrow(ax, (8.95, 3.90), (8.95, 3.10), color=C_GRAY)
    box(ax, (7.9, 2.45), 2.1, 0.62, "NMS / connected comps\n(host-free)", "#EFEFEF", fs=7.2)
    arrow(ax, (8.95, 2.45), (8.95, 1.75), color=C_GRAY)
    box(ax, (7.9, 1.10), 2.1, 0.62, "ship reports +\nmeasured $P_{fa}\\leq\\alpha$", "#FFFFFF", fs=7.2)

    ax.text(0.1, 2.95, "one-time offline steps (host, outside the loop)", fontsize=8, color=C_GRAY, weight="bold")
    box(ax, (0.1, 1.9), 2.6, 0.75, "$\\Omega_{clutter}$: GT guard band\n+ target-free chips, block=64px", "#F2F2F2", fs=7.0, ec=C_GRAY, lw=0.8)
    box(ax, (3.3, 1.9), 2.9, 0.75, "integer quantile $\\lambda_{int}$\n(Eq. 22, histogram $B{=}4096$)", "#F2F2F2", fs=7.0, ec=C_GRAY, lw=0.8)
    box(ax, (6.7, 1.9), 2.6, 0.75, "or transport $\\lambda_{int}$\nfrom two scale tables (Eq. 26)", "#F2F2F2", fs=7.0, ec=C_GRAY, lw=0.8)
    arrow(ax, (2.7, 2.27), (3.3, 2.27), color=C_GRAY, lw=0.9)
    arrow(ax, (6.2, 2.27), (6.7, 2.27), color=C_GRAY, lw=0.9)
    arrow(ax, (4.75, 2.65), (4.75, 3.55), color=C_GRAY, lw=0.9, ls="--")
    ax.text(4.9, 3.02, "compiled-in constants", fontsize=6.8, color=C_GRAY, rotation=90, va="center")

    ax.text(0.1, 0.55, "measured on Atlas 200I DK A2 (Ascend 310B): decision layer 0.024 ms ($\\ll$ 3 ms budget), "
                       "P99/mean $=1.0001$, bit-exact reproducibility $=1.0$ (E3/E7)",
            fontsize=7.0, color="#444444")
    fig.tight_layout()
    save(fig, "fig2_pipeline")


# ---------------------------------------------------------------- Fig.3 (E2)
def fig3():
    df = pd.read_csv(RES / "e2" / "e2_audit.csv")
    g = df.groupby("contrast")[["sat", "tie_ratio", "uniq_frac"]].mean().reset_index()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.8))

    a1.bar(g["contrast"].astype(str), g["sat"], width=0.5, color=C_FP16, alpha=0.85)
    a1.set_ylim(0, 0.004)
    a1.set_xlabel("synthetic contrast level"); a1.set_ylabel(r"score-layer saturation rate")
    a1.set_title("(a) Saturation rate at the decision score layer")
    a1.text(1, 0.0022, "identically 0 across all 300 scenes\n(int32 accumulator grid —\nno int8 requantization\nbefore sigmoid)",
            ha="center", fontsize=7.3, color=C_RED)

    a2.plot(g["contrast"], g["tie_ratio"], "o-", color=C_INT8, label="tie ratio (same integer code)")
    a2.plot(g["contrast"], g["uniq_frac"], "s--", color=C_FP16, label="unique-value fraction")
    a2.set_xlabel("synthetic contrast level"); a2.set_ylim(0, 0.30)
    a2.set_title("(b) Order-collapse proxies at the score layer")
    a2.legend(frameon=False, loc="center right")
    fig.tight_layout()
    save(fig, "fig3_saturation_audit")


# ---------------------------------------------------------------- Fig.4 (E1)
def fig4():
    d = pd.read_csv(RES / "e1" / "fig4_data.csv")
    s = pd.read_csv(RES / "e1" / "e1_eps_sweep.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.8, 3.5))

    a1.plot(d["eps_sigma"], d["meas"], "o-", color=C_INT8, lw=1.4, label=r"measured $\Delta\ln P_{fa}$")
    a1.plot(d["eps_sigma"], d["pred"], "-", color=C_GRAY, lw=1.1, label=r"first-order $z^2\varepsilon_\sigma$")
    a1.plot(d["eps_sigma"], d["pred2"], "--", color=C_ACC, lw=1.1, label="second-order correction")
    a1.set_yscale("log"); a1.set_xscale("symlog", linthresh=1e-3)
    a1.set_xlabel(r"quantization error on null-model scale $\varepsilon_\sigma$")
    a1.set_ylabel(r"$\Delta\ln P_{fa}$ (log)")
    a1.set_title("(a) Tail drift vs. denominator error (gauss, $n{=}10^6$)")
    axr = a1.twinx()
    axr.spines.right.set_visible(True)
    axr.plot(d["eps_sigma"], np.abs(d["dauroc"]), "v:", color=C_RED, lw=1.0, label=r"$|\Delta$AUROC$|$")
    axr.set_ylabel(r"$|\Delta$AUROC$|$ (mAP proxy)", color=C_RED, fontsize=8)
    axr.tick_params(axis="y", colors=C_RED, labelsize=7.5)
    axr.set_ylim(0, 2.6e-3)
    axr.text(0.012, 2.15e-3, "mAP stays blind:\n$<$2 mAP points moved", color=C_RED, fontsize=7)
    h1, l1 = a1.get_legend_handles_labels()
    h2, l2 = axr.get_legend_handles_labels()
    a1.legend(h1 + h2, l1 + l2, frameon=False, loc="lower right", fontsize=7)

    ok = s[s["dist"] == "gauss"]
    a2.plot(ok["bound_L_delta"], ok["abs_dpfa"], "o", color=C_FP16, label="gauss (6 levels)")
    t3 = pd.read_csv(RES / "e1_t3" / "e1_eps_sweep.csv")
    a2.plot(t3["bound_L_delta"], t3["abs_dpfa"], "^", color=C_INT8, label="t3 heavy-tail (6 levels)")
    lim = [1e-6, 5e-3]
    a2.plot(lim, lim, "--", color=C_RED, lw=1.0)
    a2.text(2.8e-6, 1.6e-6, "y = x (bound active)", color=C_RED, fontsize=7, rotation=24)
    a2.text(4.7e-3, 4.6e-3, "all 12 levels below the line\nbound_holds = True",
            fontsize=7, ha="right", va="top", color=C_GRAY)
    a2.set_xscale("log"); a2.set_yscale("log"); a2.set_xlim(lim); a2.set_ylim(lim)
    a2.set_xlabel(r"Prop.2 bound $L\delta$ (from scale table)")
    a2.set_ylabel(r"measured $|\Delta P_{fa}|$")
    a2.set_title("(b) Bound covers measured drift")
    a2.legend(frameon=False, loc="upper left", fontsize=7)
    fig.tight_layout()
    save(fig, "fig4_bound")


# ---------------------------------------------------------------- Fig.5 (E7)
def fig5():
    d = pd.read_csv(RES / "e7" / "table5a_e7.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.8, 3.3))
    labels = ["FP16\nglobal", "FP16\nQCert", "INT8\nglobal", "INT8\nQCert"]
    colors = [C_FP16, "#6BAED6", C_INT8, "#F2C57C"]
    x = np.arange(4)
    a1.bar(x - 0.18, d["t_e2e_mean_ms"], width=0.36, color=colors, label="mean")
    a1.bar(x + 0.18, d["t_e2e_p99_ms"], width=0.36, color=colors, alpha=0.45, hatch="//", edgecolor="white", label="P99")
    for i, r in d.iterrows():
        a1.text(i, max(r["t_e2e_mean_ms"], r["t_e2e_p99_ms"]) + 0.9,
                f"P99/mean {r['p99_over_mean']:.5f}", ha="center", fontsize=6.5)
    a1.set_xticks(x); a1.set_xticklabels(labels, fontsize=7.3)
    a1.set_ylabel("end-to-end latency (ms)"); a1.set_ylim(0, 66)
    a1.set_title("(a) Latency is tail-free (P99/mean $\\approx$ 1)")
    a1.legend(frameon=False, loc="upper right", fontsize=7, ncol=2)
    a1.text(1.5, 57.0, "INT8 vs FP16: 1.31$\\times$ faster (34.7 ms vs 45.5 ms)",
            fontsize=7.2, ha="center")

    w = 0.34
    a2.bar([0 - w / 2, 1 - w / 2], d[d["thr"] == "global"]["cand_p90_rough"], width=w, color=C_GRAY, label="global threshold")
    a2.bar([0 + w / 2, 1 + w / 2], d[d["thr"] == "qcert"]["cand_p90_rough"], width=w, color=C_ACC, label="QCert")
    a2.set_xticks([0, 1]); a2.set_xticklabels(["FP16 OM", "INT8 OM"])
    for xi, (gv, qv) in enumerate(zip(d[d["thr"] == "global"]["cand_p90_rough"],
                                      d[d["thr"] == "qcert"]["cand_p90_rough"])):
        a2.text(xi, max(gv, qv) + 0.7, f"{gv:.0f} $\\rightarrow$ {qv:.0f}\n({gv / qv:.2f}$\\times$ fewer)",
                ha="center", fontsize=7.3)
    a2.set_ylabel("candidate count P90 (rough seas)")
    a2.set_title("(b) Constant false alarm $\\Rightarrow$ constant load")
    a2.legend(frameon=False, loc="upper left", fontsize=7, ncol=2)
    a2.set_ylim(0, 50)
    fig.tight_layout()
    save(fig, "fig5_system")


# ---------------------------------------------------------------- Fig.7 (E3)
def fig7():
    df = pd.read_csv(RES / "e3" / "e3_per_scene.csv")
    rng = np.random.default_rng(1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.8), gridspec_kw={"width_ratios": [1.35, 1]})

    for x, col, lab, c in [(0.0, "disagree_int8_board", "board .om repeat", C_FP16),
                           (1.0, "disagree_int8_ort", "ORT-INT8 (x86)", C_INT8)]:
        y = df[col].to_numpy()
        y = y[y > 0]
        a1.scatter(x + rng.uniform(-0.15, 0.15, len(y)), y, s=8, alpha=0.5, color=c, label=lab, zorder=2)
        a1.hlines(y.mean(), x - 0.28, x + 0.28, color=c, lw=1.8, zorder=3)
        a1.text(x, y.mean() * 2.1, f"mean {y.mean():.2e}", ha="center", fontsize=7.2, color=c)
    a1.set_yscale("log"); a1.set_ylim(8e-5, 3e-2)
    a1.set_xticks([0, 1]); a1.set_xticklabels(["board .om", "ORT-INT8"])
    a1.set_ylabel("per-position decision\ndisagreement rate (log)")
    a1.set_title("(a) Cross-implementation decision divergence")
    a1.text(0.5, 1.3e-2, "zero-disagreement scenes omitted\n(bit-exact rate = 1.0 within same .om)",
            ha="center", fontsize=7.0, color=C_GRAY)

    for x, col, lab, c in [(0.0, "R_board", "board", C_FP16), (1.0, "R_ort", "ORT", C_INT8)]:
        y = df[col].to_numpy()
        a2.scatter(x + rng.uniform(-0.15, 0.15, len(y)), y, s=8, alpha=0.5, color=c, zorder=2)
        a2.hlines(y.mean(), x - 0.28, x + 0.28, color=c, lw=1.8, zorder=3)
    a2.axhline(1.0, color=C_RED, ls="--", lw=1.0)
    a2.text(1.42, 1.002, r"$R=1$", color=C_RED, fontsize=7.5, ha="right")
    a2.set_ylim(0.9, 1.1)
    a2.set_xticks([0, 1]); a2.set_xticklabels(["board .om", "ORT-INT8"])
    a2.set_ylabel(r"invariance ratio $R$")
    a2.set_title("(b) $R$ stays at 1 per scene")
    a2.text(0.5, 1.075, "mean $R_{board}=1.00086$\nmean $R_{ort}=1.00259$", ha="center", fontsize=7.2)
    fig.tight_layout()
    save(fig, "fig7_cross_impl")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5(); fig7()
    print("all figures ->", FIGS)
