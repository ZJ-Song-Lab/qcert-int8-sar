# -*- coding: utf-8 -*-
"""新增正式图 Fig.31-35 -> paper/figs/

Fig.31 近岸/远洋 P_fa ECDF + 候选负载 ECDF + coast_score 分布（场景对比定量）
Fig.32 E9 消融对比汇总（item6 h0 / item7 分层 / item5 全局vs逐场景）
Fig.33 与 SOTA 对比（SSDD mAP50 + Table 0 覆盖矩阵 + P_d 对照）
Fig.34 近岸/远洋并排检测 gallery（calmA/roughB 双行，同一图内直接对比）
Fig.35 P3 objectness 热力图 2x2 gallery（近岸x2 vs 远洋x2，共享 colorbar）
"""
import json
import os

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

ROOT = r"D:\dinghuibianyuan"
CODE = os.path.join(ROOT, "neural networks", "code")
R7 = os.path.join(CODE, "results", "e7")
R9 = os.path.join(CODE, "results", "e9")
R9R = os.path.join(CODE, "results", "e9_real")
SSDD = os.path.join(ROOT, "neural networks", "asc-sar", "datasets", "ssdd")
DUMPS = os.path.join(CODE, "dumps", "e5s1")
FIGS = os.path.join(ROOT, "neural networks", "paper", "figs")
ALPHA = 1e-3
IMGSZ = 640
C_TYPE = {"nearshore": "#d62728", "offshore": "#1f77b4"}

plt.rcParams.update({
    "font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
    "figure.dpi": 200, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.bottom": False, "axes.spines.left": False,
})


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGS, f"{name}.{ext}"))
    plt.close(fig)
    print("saved:", name)


def load_csv(path):
    with open(path, encoding="utf-8-sig") as f:
        import csv
        return list(csv.DictReader(f))


pf = load_csv(os.path.join(R7, "e7_per_frame_pfa.csv"))
stt = {r["stem"]: r["scene_type"] for r in load_csv(os.path.join(R7, "e7_scene_type.csv"))}
cs = {r["stem"]: float(r["coast_score"]) for r in load_csv(os.path.join(R7, "e7_scene_type.csv"))}

# ================================================================= Fig.31
fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.6))

# (a) ECDF per-frame P_fa/alpha, INT8 global
ax = axes[0, 0]
for t, c in C_TYPE.items():
    v = np.sort(np.array([float(r["pfa_global_int8"]) for r in pf
                          if stt[r["stem"]] == t]) / ALPHA)
    y = np.arange(1, len(v) + 1) / len(v)
    n2 = int((v > 2).sum())
    ax.step(np.concatenate([[v[0]], v]), np.concatenate([[0], y]),
            where="post", color=c, lw=1.5,
            label=f"{t} (n={len(v)}, >2α {n2/len(v)*100:.1f}%)")
ax.axvline(1, color="k", ls="--", lw=0.8)
ax.axvline(2, color="#d62728", ls=":", lw=0.9)
ax.set_xscale("log")
ax.set_xlim(0.05, 30)
ax.set_xlabel(r"per-frame $P_{\rm fa}/\alpha$ (INT8 global)")
ax.set_ylabel("ECDF")
ax.legend(fontsize=8, frameon=False, loc="upper left")
ax.set_title("(a) INT8-global: nearshore tail vs offshore", fontsize=9)

# (b) ECDF INT8-QCert
ax = axes[0, 1]
for t, c in C_TYPE.items():
    v = np.sort(np.array([float(r["pfa_qcert_int8"]) for r in pf
                          if stt[r["stem"]] == t]) / ALPHA)
    y = np.arange(1, len(v) + 1) / len(v)
    ax.step(np.concatenate([[v[0]], v]), np.concatenate([[0], y]),
            where="post", color=c, lw=1.5, label=f"{t} (n={len(v)})")
ax.axvline(1, color="k", ls="--", lw=0.8)
ax.axvline(2, color="#d62728", ls=":", lw=0.9)
ax.set_xscale("log")
ax.set_xlim(0.05, 30)
ax.set_xlabel(r"per-frame $P_{\rm fa}/\alpha$ (INT8 QCert)")
ax.set_ylabel("ECDF")
ax.legend(fontsize=8, frameon=False, loc="upper left")
ax.set_title("(b) INT8-QCert: 1.07α in both subsets", fontsize=9)

# (c) candidate load ECDF
ax = axes[1, 0]
for t, c in C_TYPE.items():
    v = np.sort([int(r["n_cand_global_int8"]) for r in pf if stt[r["stem"]] == t])
    y = np.arange(1, len(v) + 1) / len(v)
    ax.step(v, y, where="post", color=c, lw=1.5, label=t)
med = {t: float(np.median([int(r["n_cand_global_int8"]) for r in pf
                           if stt[r["stem"]] == t])) for t in C_TYPE}
ax.set_xlabel("candidates per frame (INT8 global)")
ax.set_ylabel("ECDF")
ax.legend(fontsize=8, frameon=False, loc="upper right",
          title=f"median {med['nearshore']:.0f} vs {med['offshore']:.0f}")
ax.set_title("(c) candidate-load ECDF", fontsize=9)

# (d) coast_score distribution
ax = axes[1, 1]
for t, c in C_TYPE.items():
    v = [cs[s] for s in stt if stt[s] == t]
    ax.hist(v, bins=np.arange(0, 71, 2.5), color=c, alpha=0.55, label=t,
            density=True)
thr = sorted(cs.values(), reverse=True)[39]  # TOP-40 boundary
ax.axvline(thr, color="k", ls="--", lw=0.9)
ax.text(thr + 1, ax.get_ylim()[1] * 0.92, f"TOP-40 cut = {thr:.1f}", fontsize=7.5)
for t, x in [("nearshore", 41.9), ("offshore", 8.9)]:
    ax.text(x + 1, ax.get_ylim()[1] * (0.75 if t == "nearshore" else 0.55),
            f"{t} med\n{x}", fontsize=7.5, color=C_TYPE[t])
ax.set_xlabel("coast_score = std(row) + std(col)")
ax.set_ylabel("density")
ax.legend(fontsize=8, frameon=False)
ax.set_title("(d) scene-type score distribution", fontsize=9)

fig.tight_layout()
save(fig, "fig31_scene_pfa_ecdf")

# ================================================================= Fig.32
syn = json.load(open(os.path.join(R9, "table3_e9.json"), encoding="utf-8"))
real = json.load(open(os.path.join(R9R, "summary.json"), encoding="utf-8"))


def pick(rows, item, setting, metric):
    for r in rows:
        if r["item"] == item and r["setting"] == setting and r["metric"] == metric:
            return r["value"]


fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.4))

# (a) item6: h0 ablation
r_h0 = pick(syn, "6 有/无 h0", "z = (s−μ)/σ", "R（INT8/FP32，汇总）")
r_no = pick(syn, "6 有/无 h0", "s（无 h0，直接取阈值）", "R（INT8/FP32，汇总）")
pd_h0 = pick(syn, "6 有/无 h0", "z = (s−μ)/σ", "P_d@1e-3（INT8，汇总）")
pd_no = pick(syn, "6 有/无 h0", "s（无 h0，直接取阈值）", "P_d@1e-3（INT8，汇总）")
ax = axes[0, 0]
xb = np.arange(2)
ax.bar(xb - 0.18, [r_h0, r_no], width=0.32, color="#1f77b4", label="R (INT8/FP32)")
ax.bar(xb + 0.18, [pd_h0, pd_no], width=0.32, color="#ff7f0e", label="$P_d$@1e-3")
ax.axhline(1.0, ls="--", lw=0.8, color="k")
ax.set_xticks(xb)
ax.set_xticklabels(["with $h_0$: $z=(s-\\mu)/\\sigma$", "no $h_0$: raw $s$"])
for x, v in zip(xb - 0.18, [r_h0, r_no]):
    ax.text(x, v + 0.02, f"{v:.2f}", ha="center", fontsize=7.5)
for x, v in zip(xb + 0.18, [pd_h0, pd_no]):
    ax.text(x, v + 0.02, f"{v:.2f}", ha="center", fontsize=7.5)
ax.legend(fontsize=8, frameon=False)
ax.set_title("(a) E9-6 clutter normalization $h_0$", fontsize=9)

# (b) item7: saturation rate + INT8 P_fa
sat_s = pick(syn, "7 校准集分层", "分层（三海况）", "s 上饱和率（压测集均值）")
sat_c = pick(syn, "7 校准集分层", "只取平静", "s 上饱和率（压测集均值）")
pf_s = pick(syn, "7 校准集分层", "分层（三海况）", "P_fa INT8")
pf_c = pick(syn, "7 校准集分层", "只取平静", "P_fa INT8")
ax = axes[0, 1]
b1 = ax.bar(xb - 0.18, [sat_s * 100, sat_c * 100], width=0.32,
            color="#9467bd", label="saturation rate (%)")
ax.set_ylabel("saturation rate (%)", color="#9467bd")
ax2 = ax.twinx()
ax2.bar(xb + 0.18, [pf_s * 1e4, pf_c * 1e4], width=0.32,
        color="#8c564b", label=r"$P_{\rm fa}$ ($\times10^{-4}$)")
ax2.grid(False)
ax.set_xticks(xb)
ax.set_xticklabels(["stratified (3 sea states)", "calm-only calibration"])
for x, v in zip(xb - 0.18, [sat_s * 100, sat_c * 100]):
    ax.text(x, v + 0.15, f"{v:.1f}%", ha="center", fontsize=7.5, color="#9467bd")
for x, v in zip(xb + 0.18, [pf_s * 1e4, pf_c * 1e4]):
    ax2.text(x, v + 0.15, f"{v:.1f}", ha="center", fontsize=7.5, color="#8c564b")
ax.set_title("(b) E9-7 calibration-set stratification", fontsize=9)
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=8, frameon=False)

# (c) item7: R
rr_s = pick(syn, "7 校准集分层", "分层（三海况）", "R（压测集，INT8/FP32）")
rr_c = pick(syn, "7 校准集分层", "只取平静", "R（压测集，INT8/FP32）")
ax = axes[1, 0]
ax.bar(xb, [rr_s, rr_c], width=0.5, color=["#2ca02c", "#d62728"])
ax.axhline(1.0, ls="--", lw=0.8, color="k")
ax.axhspan(0.8, 1.2, color="k", alpha=0.06)
ax.text(1.28, 1.0, "R=1\n±20%", fontsize=7, va="center")
for x, v in zip(xb, [rr_s, rr_c]):
    ax.text(x, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
ax.set_xticks(xb)
ax.set_xticklabels(["stratified (3 sea states)", "calm-only calibration"])
ax.set_ylabel("R (INT8/FP32)")
ax.set_title("(c) E9-7 drift ratio R (stress set)", fontsize=9)

# (d) item5: global vs per-scene calibration error (real)
it5 = real["conclusions"]["item5"]
ax = axes[1, 1]
vals = [it5["err_global"], it5["err_per_scene"]]
ax.bar(xb, vals, width=0.5, color=["#2ca02c", "#d62728"])
for x, v in zip(xb, vals):
    ax.text(x, v * 1.15, f"{v:.4f}", ha="center", fontsize=8)
ax.set_yscale("log")
ax.set_ylim(0.005, 1.0)
ax.axhline(0.05, ls=":", lw=0.9, color="k")
ax.text(-0.42, 0.06, "0.05 dex", fontsize=7)
ax.set_xticks(xb)
ax.set_xticklabels(["global calibration", "per-scene calibration"])
ax.set_ylabel(r"$\alpha$ err $|\log_{10}|$ (dex)")
ax.set_title(f"(d) E9-5 global vs per-scene\n($\\Delta P_d$ = {it5['d_pd']:.4f}, global keeps $P_d$)", fontsize=9)

fig.tight_layout()
save(fig, "fig32_ablation_summary")

# ================================================================= Fig.33
sota = [("AC-YOLO", 96.7), ("BESW-YOLO", 97.3), ("LSD-DETR", 97.3),
        ("LAMFDet", 99.5)]
ours_f, ours_i = 98.27, 97.98
fig = plt.figure(figsize=(13.6, 4.0))
gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.1, 1.0])

# (a) SSDD mAP50 comparison
ax = fig.add_subplot(gs[0, 0])
names = [n for n, _ in sota[:2]] + ["Ours FP16"] + ["Ours INT8+QCert"] + [sota[2][0], sota[3][0]]
vals = [sota[0][1], sota[1][1], ours_f, ours_i, sota[2][1], sota[3][1]]
cols = ["#7f7f7f", "#7f7f7f", "#1f77b4", "#d62728", "#7f7f7f", "#7f7f7f"]
yb = np.arange(len(names))
ax.barh(yb, vals, color=cols, height=0.6)
ax.axvspan(ours_f - 1, ours_f + 1, color="#2ca02c", alpha=0.10)
ax.text(ours_f + 1.05, 0.45, "Vitis AI gate\n(FP16 ±1 mAP)", fontsize=7,
        color="#2ca02c", va="center")
for y, v in zip(yb, vals):
    ax.text(v + 0.06, y, f"{v:.2f}" if "Ours" in names[y] else f"{v:.1f}",
            va="center", fontsize=8)
ax.set_yticks(yb)
ax.set_yticklabels(names, fontsize=8)
ax.set_xlim(95, 100.6)
ax.set_xlabel("SSDD mAP50 (%)")
ax.set_title("(a) accuracy vs published SOTA on SSDD", fontsize=9)

# (b) Table 0 coverage matrix
ax = fig.add_subplot(gs[0, 1])
rows = ["SAR det SOTA (YOLO/DETR)", "PTQ for detection", "QAT family",
        "Classical CFAR", "CFARnet", "Conformal (SeqCRC)", "Ours: QCert"]
colsM = ["mAP reported", r"$P_{\rm fa}$@α tail", "clutter model",
         "integer-arith", "board test"]
# 2=half, 1=yes, 0=no
M = [[1, 0, 0, 0, 0],
     [1, 0, 2, 1, 2],
     [1, 0, 0, 1, 2],
     [0, 1, 1, 0, 0],
     [0, 1, 2, 0, 0],
     [0, 0, 0, 0, 0],
     [1, 1, 1, 1, 1]]
CMAP = matplotlib.colors.ListedColormap(
    ["#f2b8b8", "#c8e6c9", "#fff3b0"])
ax.imshow(np.array(M), cmap=CMAP, vmin=0, vmax=2, aspect="auto")
TXT = {0: "✗", 1: "✓", 2: "△"}
for i in range(len(rows)):
    for j in range(len(colsM)):
        ax.text(j, i, TXT[M[i][j]], ha="center", va="center", fontsize=8)
ax.set_xticks(range(len(colsM)))
ax.set_xticklabels(colsM, rotation=32, ha="right", fontsize=7.5)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels(rows, fontsize=7.5)
ax.grid(False)
ax.set_title("(b) audit-dimension coverage (Table 0)", fontsize=9)

# (c) P_d comparison (E8)
ax = fig.add_subplot(gs[0, 2])
t8 = json.load(open(os.path.join(CODE, "results", "e8", "table6_e8.json"),
                    encoding="utf-8"))
by = {r["method"]: r for r in t8}
methods = ["CA-CFAR（像素域）", "YOLO11n + 全局阈值", "YOLO11n + QCert(INT8)"]
labels = ["CA-CFAR\n(pixel)", "YOLO11n\n+ global thr", "YOLO11n\n+ QCert (INT8)"]
pds = [by[m]["pd_at_1e-3"] for m in methods]
lat = [by[m]["latency_ms"] for m in methods]
xb = np.arange(3)
bars = ax.bar(xb, pds, width=0.55, color=["#7f7f7f", "#1f77b4", "#d62728"])
for x, v in zip(xb, pds):
    ax.text(x, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)
ax.set_xticks(xb)
ax.set_xticklabels(labels, fontsize=7.5)
ax.set_ylim(0, 1.12)
ax.set_ylabel(r"$P_d$ @ $\alpha=10^{-3}$")
ax.set_title("(c) detection power @ $\\alpha$=1e-3\n(board: 45.5→34.68 ms, 1.31×)", fontsize=9)

fig.tight_layout()
save(fig, "fig33_sota_compare")

# ================================================================= Fig.34/35 helpers
HAN = [Line2D([], [], color="#00ff66", label="GT"),
       Line2D([], [], color="#ff7f0e", label="INT8 det")]


def load_img(stem):
    return cv2.cvtColor(cv2.imread(os.path.join(SSDD, "images", "test", stem + ".jpg")),
                        cv2.COLOR_BGR2RGB)


def load_gt(stem):
    img = cv2.imread(os.path.join(SSDD, "images", "test", stem + ".jpg"))
    ih, iw = img.shape[:2]
    boxes = []
    with open(os.path.join(SSDD, "labels", "test", stem + ".txt")) as f:
        for ln in f:
            parts = ln.split()
            if len(parts) < 5:
                continue
            _, xc, yc, w, h = map(float, parts[:5])
            boxes.append([(xc - w / 2) * iw, (yc - h / 2) * ih,
                          (xc + w / 2) * iw, (yc + h / 2) * ih])
    return boxes


def dets(stem, cfg="int8_calmA"):
    d = np.load(os.path.join(DUMPS, cfg, stem + ".npz"), allow_pickle=True)
    return json.loads(str(d["meta"])).get("boxes", []), d


def draw_boxes(ax, boxes, color, lw=0.5):
    for x1, y1, x2, y2 in boxes:
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec=color, lw=lw))


def to_lb(boxes, l, t, nu, iw, ih):
    """原图坐标 -> letterbox 640 画布坐标。"""
    sx, sy = nu[0] / iw, nu[1] / ih
    return [[l + x1 * sx, t + y1 * sy, l + x2 * sx, t + y2 * sy]
            for x1, y1, x2, y2 in boxes]


def letterbox(img):
    ih, iw = img.shape[:2]
    r = min(IMGSZ / iw, IMGSZ / ih)
    nu = (round(iw * r), round(ih * r))
    rs = cv2.resize(img, nu, interpolation=cv2.INTER_LINEAR)
    dw, dh = IMGSZ - nu[0], IMGSZ - nu[1]
    l, t = dw // 2, dh // 2
    out = cv2.copyMakeBorder(rs, t, dh - t, l, dw - l,
                             cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return out, (l, t, nu)


# Fig.34 nearshore vs offshore side-by-side gallery (per-frame P_fa annotated)
FRAMES = {
    "nearshore": ["001131", "001159", "001039", "001111"],
    "offshore": ["001149", "000109", "000249", "000739"],
}
pfm = {r["stem"]: r for r in pf}
fig, axes = plt.subplots(2, 4, figsize=(13.2, 6.8))
for i, ty in enumerate(["nearshore", "offshore"]):
    for j, st in enumerate(FRAMES[ty]):
        ax = axes[i, j]
        img = load_img(st)
        ih, iw = img.shape[:2]
        im_lb, (l, t, nu) = letterbox(img)
        ax.imshow(im_lb)
        ax.grid(False)
        gtb = load_gt(st)
        db = dets(st, "int8_calmA")[0]
        draw_boxes(ax, to_lb(gtb, l, t, nu, iw, ih), "#00ff66")
        draw_boxes(ax, to_lb(db, l, t, nu, iw, ih), "#ff7f0e")
        r = pfm[st]
        pfr = float(r["pfa_global_int8"]) / ALPHA
        ax.set_title(f"{st}: gt={len(gtb)}, det={len(db)}\n"
                     rf"$P_{{\rm fa}}$ = {pfr:.2f}$\alpha$, cand = "
                     f"{r['n_cand_global_int8']}",
                     fontsize=8, color=C_TYPE[ty])
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    axes[i, 0].set_ylabel(ty, fontsize=9, color=C_TYPE[ty])
axes[0, 0].legend(handles=HAN, loc="lower right", fontsize=8, frameon=True)
fig.suptitle("Nearshore vs offshore (INT8, calmA): identical detector, "
             r"but per-frame $P_{\rm fa}$ tail differs"
             "\n(000739 = offshore tail outlier; QCert keeps 1.07$\\alpha$ on all frames)",
             fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.93])
save(fig, "fig34_scene_gallery_compare")

# Fig.35 heatmap overlay gallery 2x2
HM = [("001111", "nearshore"), ("001159", "nearshore"),
      ("001149", "offshore"), ("000739", "offshore")]
fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.2))
last_im = None
for ax, (st, ty) in zip(axes.flat, HM):
    img = load_img(st)
    im_lb, _ = letterbox(img)
    d = np.load(os.path.join(DUMPS, "int8_calmA", st + ".npz"), allow_pickle=True)
    p3 = d["prob"][: 80 * 80].reshape(80, 80)
    h640 = cv2.resize(p3, (IMGSZ, IMGSZ), interpolation=cv2.INTER_CUBIC)
    ax.imshow(im_lb)
    last_im = ax.imshow(h640, cmap="inferno", alpha=0.45)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"{st} ({ty})", fontsize=8.5, color=C_TYPE[ty])
cb = fig.colorbar(last_im, ax=axes, fraction=0.025, pad=0.015)
cb.set_label("P3 objectness (INT8, calmA)", fontsize=8)
fig.suptitle("P3 objectness heatmaps: land/coastline response (nearshore) vs ship wake only (offshore)",
             fontsize=9.5)
fig.savefig(os.path.join(FIGS, "fig35_scene_heatmap_gallery.png"))
fig.savefig(os.path.join(FIGS, "fig35_scene_heatmap_gallery.pdf"))
plt.close(fig)
print("saved: fig35_scene_heatmap_gallery")
print("ALL-DONE")
