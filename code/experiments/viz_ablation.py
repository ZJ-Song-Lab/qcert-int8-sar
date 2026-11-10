# -*- coding: utf-8 -*-
"""E9 消融实验可视化样例 -> paper/figs/samples/

a1 块抽样 keep 扫描 (真实)      a2 标定样本量 n (真实)
a3 floor vs round 舍入标度 (合成) a4 有/无 h0 + 校准集分层 (合成)
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"D:\dinghuibianyuan"
R9 = os.path.join(ROOT, "neural networks", "code", "results", "e9")
R9R = os.path.join(ROOT, "neural networks", "code", "results", "e9_real")
OUT = os.path.join(ROOT, "neural networks", "paper", "figs", "samples")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
    "figure.dpi": 200, "savefig.bbox": "tight",
})

real = json.load(open(os.path.join(R9R, "summary.json"), encoding="utf-8"))
syn = json.load(open(os.path.join(R9, "table3_e9.json"), encoding="utf-8"))


def pick(rows, item, setting, metric):
    for r in rows:
        if r["item"] == item and r["setting"] == setting and r["metric"] == metric:
            return r["value"]
    return None


# -------------------------------------------------- a1 块抽样 keep（真实）
it3 = real["conclusions"]["item3"]
keeps = [1.0, 0.25, 0.125, 0.0625]
KE = {1.0: "keep_1", 0.25: "keep_0.25", 0.125: "keep_0.125", 0.0625: "keep_0.0625"}
errs = [it3[KE[k]]["err"] for k in keeps]
neff = [it3[KE[k]]["n_eff"] for k in keeps]

fig, ax1 = plt.subplots(figsize=(3.7, 3.0))
ax2 = ax1.twinx()
l1, = ax1.plot(range(4), errs, "o-", color="#d62728", label=r"err $=|\log_{10}(P_{fa}/\alpha)|$")
l2, = ax2.plot(range(4), neff, "s--", color="#1f77b4", label=r"$n_{\rm eff}$")
ax1.set_xticks(range(4))
ax1.set_xticklabels([str(k) for k in keeps])
ax1.set_xlabel("block-sampling keep ratio")
ax1.set_ylabel("err (dex)", color="#d62728")
ax2.set_ylabel(r"$n_{\rm eff}$", color="#1f77b4")
ax2.grid(False)
ax1.legend(handles=[l1, l2], loc="upper left", fontsize=8, frameon=False)
fig.savefig(os.path.join(OUT, "a1_keep_sweep.png"))
plt.close(fig)

# -------------------------------------------------- a2 标定样本量（真实）
it4 = real["conclusions"]["item4"]
ns = ["n16", "n64", "n128", "n200"]
means = [it4[k]["mean"] for k in ns]
stds = [it4[k]["std"] for k in ns]
fig, ax = plt.subplots(figsize=(3.7, 3.0))
x = [16, 64, 128, 200]
ax.errorbar(x, means, yerr=stds, fmt="o-", capsize=3, color="#1f77b4")
ax.set_xscale("log")
ax.set_xlabel("calibration scenes $n_{\\rm cal}$")
ax.set_ylabel(r"$\alpha$ err $|\log_{10}|$ (dex)")
ax.annotate("chosen\n$n$=128", xy=(128, means[2]), xytext=(150, 0.115),
            fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.8))
fig.savefig(os.path.join(OUT, "a2_calib_n.png"))
plt.close(fig)

# -------------------------------------------------- a3 floor vs round（合成）
eps = [0.02, 0.05, 0.10, 0.20]
dl_floor = [pick(syn, "8 floor vs round", f"floor @ ε_σ={e}", "Δln P_fa（汇总）") for e in eps]
dl_round = [pick(syn, "8 floor vs round", f"round @ ε_σ={e}", "Δln P_fa（汇总）") for e in eps]
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
ax = axes[0]
ax.plot(eps, dl_floor, "o-", color="#d62728", label="floor (biased)")
ax.plot(eps, dl_round, "s-", color="#1f77b4", label="round (symmetric)")
ref = np.array(eps)
ax.plot(ref, 9.30 * ref, ":", color="#d62728", lw=1, label=r"$z_\lambda^2\,\varepsilon_\sigma$ (Prop.2, $p$=1)")
ax.plot(ref, dl_round[0] * (ref / ref[0]) ** 2, ":", color="#1f77b4", lw=1, label=r"$\propto\varepsilon_\sigma^2$ ($p$=2)")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"$\varepsilon_\sigma$")
ax.set_ylabel(r"$|\Delta\ln P_{\rm fa}|$")
ax.legend(fontsize=7, frameon=False)
ax.set_title("(a) rounding-mode scaling")
# 局部指数
segs = [("0.02→0.05", "round"), ("0.05→0.10", "round"), ("0.10→0.20", "round"),
        ("0.02→0.05", "floor"), ("0.05→0.10", "floor"), ("0.10→0.20", "floor")]
pvals = [pick(syn, "8 floor vs round", f"{m} 局部 {s}", "指数 p") for s, m in segs]
ax = axes[1]
xx = np.arange(6)
colors = ["#1f77b4"] * 3 + ["#d62728"] * 3
ax.bar(xx, pvals, color=colors, width=0.6)
ax.axhline(2, ls="--", lw=0.8, color="#1f77b4")
ax.axhline(1, ls="--", lw=0.8, color="#d62728")
ax.set_xticks(xx)
ax.set_xticklabels([s for s, _ in segs], rotation=30, fontsize=7)
ax.set_ylabel("local exponent $p$")
ax.set_title("(b) local exponent")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "a3_round_floor_scaling.png"))
plt.close(fig)

# -------------------------------------------------- a4 h0 / 校准分层（合成）
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
ax = axes[0]
r_h0 = pick(syn, "6 有/无 h0", "z = (s−μ)/σ", "R（INT8/FP32，汇总）")
r_no = pick(syn, "6 有/无 h0", "s（无 h0，直接取阈值）", "R（INT8/FP32，汇总）")
pd_h0 = pick(syn, "6 有/无 h0", "z = (s−μ)/σ", "P_d@1e-3（INT8，汇总）")
pd_no = pick(syn, "6 有/无 h0", "s（无 h0，直接取阈值）", "P_d@1e-3（INT8，汇总）")
xb = np.arange(2)
ax.bar(xb - 0.18, [r_h0, r_no], width=0.32, color="#1f77b4", label="R (INT8/FP32)")
ax.bar(xb + 0.18, [pd_h0, pd_no], width=0.32, color="#ff7f0e", label="$P_d$@1e-3")
ax.axhline(1.0, ls="--", lw=0.8, color="k")
ax.set_xticks(xb)
ax.set_xticklabels(["with $h_0$: $z=(s-\\mu)/\\sigma$", "no $h_0$: raw $s$"])
for x, v in zip(xb - 0.18, [r_h0, r_no]):
    ax.text(x, v + 0.02, f"{v:.2f}", ha="center", fontsize=7)
for x, v in zip(xb + 0.18, [pd_h0, pd_no]):
    ax.text(x, v + 0.02, f"{v:.2f}", ha="center", fontsize=7)
ax.legend(fontsize=8, frameon=False)
ax.set_title("(a) clutter normalization $h_0$")
ax = axes[1]
sat_str = pick(syn, "7 校准集分层", "分层（三海况）", "s 上饱和率（压测集均值）")
sat_cal = pick(syn, "7 校准集分层", "只取平静", "s 上饱和率（压测集均值）")
rr_str = pick(syn, "7 校准集分层", "分层（三海况）", "R（压测集，INT8/FP32）")
rr_cal = pick(syn, "7 校准集分层", "只取平静", "R（压测集，INT8/FP32）")
xb = np.arange(2)
ax.bar(xb - 0.18, [sat_str, sat_cal], width=0.32, color="#9467bd", label="saturation rate (stress)")
ax.bar(xb + 0.18, [rr_str, rr_cal], width=0.32, color="#2ca02c", label="R (INT8/FP32)")
ax.set_xticks(xb)
ax.set_xticklabels(["stratified (3 sea states)", "calm-only calibration"])
for x, v in zip(xb - 0.18, [sat_str, sat_cal]):
    ax.text(x, v + 0.01, f"{v*100:.1f}%", ha="center", fontsize=7)
for x, v in zip(xb + 0.18, [rr_str, rr_cal]):
    ax.text(x, v + 0.01, f"{v:.2f}", ha="center", fontsize=7)
ax.legend(fontsize=8, frameon=False)
ax.set_title("(b) calibration-set stratification")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "a4_h0_and_stratify.png"))
plt.close(fig)

print("ablation samples ->", OUT)
