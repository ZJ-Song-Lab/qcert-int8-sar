# -*- coding: utf-8 -*-
"""丰富版可视化：多典型帧/多配置/多数据集拼图 -> paper/figs/samples/

b3 检测框 9 联图（3 海况 x 3 典型帧，GT vs INT8）
b4 双校准检测框 6 联图（3 帧 x calmA/roughB）
h3 objectness 热力图 2x3 拼图（3 海况 x 2 校准）
h4 热力图叠加底图（去 letterbox 对齐，alpha 混合）
s4 四配置逐帧 P_fa 组合面板
s5 压测集逐场景 R 分布（E2 压测数据集 c1/c2/c3 合并）
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
SSDD = os.path.join(ROOT, "neural networks", "asc-sar", "datasets", "ssdd")
DUMPS = os.path.join(CODE, "dumps", "e5s1")
OUT = os.path.join(ROOT, "neural networks", "paper", "figs", "samples")
ALPHA = 1e-3
IMGSZ = 640

plt.rcParams.update({
    "font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
    "figure.dpi": 200, "savefig.bbox": "tight",
})
C_BUCKET = {"rough": "#d62728", "medium": "#1f77b4", "calm": "#2ca02c"}
os.makedirs(OUT, exist_ok=True)


def load_csv(path):
    with open(path, encoding="utf-8") as f:
        rows = [ln.strip().split(",") for ln in f if ln.strip()]
    hdr = rows[0]
    return [dict(zip(hdr, r)) for r in rows[1:]]


def load_npz(config, stem):
    return np.load(os.path.join(DUMPS, config, stem + ".npz"), allow_pickle=True)


def load_img(stem):
    p = os.path.join(SSDD, "images", "test", stem + ".jpg")
    return cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)


def load_gt(stem):
    p = os.path.join(SSDD, "labels", "test", stem + ".txt")
    img = cv2.imread(os.path.join(SSDD, "images", "test", stem + ".jpg"))
    ih, iw = img.shape[:2]
    boxes = []
    if not os.path.exists(p):
        return boxes
    with open(p) as f:
        for ln in f:
            parts = ln.split()
            if len(parts) < 5:
                continue
            _, xc, yc, w, h = map(float, parts[:5])
            boxes.append([(xc - w / 2) * iw, (yc - h / 2) * ih,
                          (xc + w / 2) * iw, (yc + h / 2) * ih])
    return boxes


def letterbox(img):
    """标准 ultralytics 风格 letterbox（r=min fit, 居中 pad），用于热力图对齐重建。"""
    ih, iw = img.shape[:2]
    r = min(IMGSZ / iw, IMGSZ / ih)
    nu = (round(iw * r), round(ih * r))
    rs = cv2.resize(img, nu, interpolation=cv2.INTER_LINEAR)
    dw, dh = IMGSZ - nu[0], IMGSZ - nu[1]
    l, t = dw // 2, dh // 2
    rb, bb = dw - l, dh - t
    out = cv2.copyMakeBorder(rs, t, bb, l, rb, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return out, (l, t, nu)


def unpad_to_orig(hmap, pad, size_iwh):
    """模型空间热力图 -> 原图空间。hmap: IMGSZxIMGSZ float。"""
    l, t, nu = pad
    iw, ih = size_iwh
    h = hmap[t:t + nu[1], l:l + nu[0]]
    return cv2.resize(h, (iw, ih), interpolation=cv2.INTER_LINEAR)


def draw_boxes(ax, boxes, color):
    for x1, y1, x2, y2 in boxes:
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec=color, lw=1.4))


def pick_typical(labels, k=3):
    """每个海况桶选 k 个典型帧：优先多船（GT 多），保证有检出。"""
    by_bucket = {}
    for r in labels:
        by_bucket.setdefault(r["sea_state"], []).append(r["stem"])
    out = {}
    for b, stems in by_bucket.items():
        scored = []
        for st in stems:
            nb_gt = len(load_gt(st))
            if nb_gt == 0:
                continue
            d = load_npz("int8_calmA", st)
            nb_det = len(json.loads(str(d["meta"])).get("boxes", []))
            if nb_det == 0:
                continue
            scored.append((nb_gt + nb_det, nb_gt, st))
        scored.sort(reverse=True)
        out[b] = [st for _, _, st in scored[:k]]
    return out


LABELS = load_csv(os.path.join(CODE, "results", "e7", "e7_labels.csv"))
PFA = load_csv(os.path.join(CODE, "results", "e7", "e7_per_frame_pfa.csv"))
TYP = pick_typical(LABELS, k=3)
print("typical frames:", TYP)


# ---------------------------------------------------------------- b3
def fig_b3_gallery():
    fig, axes = plt.subplots(3, 3, figsize=(13.2, 11.4))
    for i, b in enumerate(["rough", "medium", "calm"]):
        for j, st in enumerate(TYP.get(b, [])[:3]):
            ax = axes[i, j]
            ax.imshow(load_img(st))
            ax.grid(False)
            draw_boxes(ax, load_gt(st), "#00ff66")
            d = load_npz("int8_calmA", st)
            draw_boxes(ax, json.loads(str(d["meta"])).get("boxes", []), "#ff7f0e")
            ax.set_xlabel(f"{st}  (n_gt={len(load_gt(st))})", fontsize=9)
            if j == 0:
                ax.set_ylabel(b, color=C_BUCKET[b], fontsize=12)
    handles = [Line2D([], [], color="#00ff66", label="GT"),
               Line2D([], [], color="#ff7f0e", label="INT8 det")]
    axes[0, 2].legend(handles=handles, loc="upper right", fontsize=9, frameon=False)
    fig.savefig(os.path.join(OUT, "b3_gt_vs_int8_gallery.png"))
    plt.close(fig)


# ---------------------------------------------------------------- b4
def fig_b4_calib_gallery():
    frames = [TYP["rough"][0], TYP["medium"][0], TYP["calm"][0]]
    fig, axes = plt.subplots(3, 2, figsize=(9.2, 10.2))
    for i, st in enumerate(frames):
        img = load_img(st)
        for j, cfg in enumerate(["int8_calmA", "int8_roughB"]):
            ax = axes[i, j]
            ax.imshow(img)
            ax.grid(False)
            draw_boxes(ax, load_gt(st), "#00ff66")
            d = load_npz(cfg, st)
            boxes = json.loads(str(d["meta"])).get("boxes", [])
            draw_boxes(ax, boxes, "#ff7f0e")
            ax.set_xlabel(f"{st} | {cfg} (det={len(boxes)})", fontsize=9)
    handles = [Line2D([], [], color="#00ff66", label="GT"),
               Line2D([], [], color="#ff7f0e", label="INT8 det")]
    axes[0, 1].legend(handles=handles, loc="upper right", fontsize=9, frameon=False)
    fig.savefig(os.path.join(OUT, "b4_calib_gallery.png"))
    plt.close(fig)


# ---------------------------------------------------------------- h3
def fig_h3_heatmap_gallery():
    frames = [TYP["rough"][0], TYP["medium"][0], TYP["calm"][0]]
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.2))
    for j, st in enumerate(frames):
        for i, cfg in enumerate(["int8_calmA", "int8_roughB"]):
            ax = axes[i, j]
            d = load_npz(cfg, st)
            p3 = d["prob"][: 80 * 80].reshape(80, 80)
            im = ax.imshow(p3, cmap="inferno")
            nb = len(json.loads(str(d["meta"])).get("boxes", []))
            ax.set_title(f"{st} | {cfg} (det={nb})", fontsize=9)
            ax.grid(False)
            fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "h3_heatmap_gallery.png"))
    plt.close(fig)


# ---------------------------------------------------------------- h4
def fig_h4_overlay():
    frames = [TYP["rough"][0], TYP["medium"][0]]
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 7.6))
    for i, st in enumerate(frames):
        img = load_img(st)
        ih, iw = img.shape[:2]
        _, pad = letterbox(img)
        for j, cfg in enumerate(["int8_calmA", "int8_roughB"]):
            ax = axes[i, j]
            d = load_npz(cfg, st)
            p3 = d["prob"][: 80 * 80].reshape(80, 80)
            h640 = cv2.resize(p3, (IMGSZ, IMGSZ), interpolation=cv2.INTER_CUBIC)
            h_orig = unpad_to_orig(h640, pad, (iw, ih))
            ax.imshow(img)
            ax.imshow(h_orig, cmap="inferno", alpha=0.45)
            draw_boxes(ax, load_gt(st), "#00ff66")
            boxes = json.loads(str(d["meta"])).get("boxes", [])
            draw_boxes(ax, boxes, "#00ffff")
            ax.grid(False)
            ax.set_xlabel(f"{st} | {cfg}", fontsize=8)
    fig.savefig(os.path.join(OUT, "h4_heatmap_overlay.png"))
    plt.close(fig)


# ---------------------------------------------------------------- s4
def fig_s4_pfa_4config():
    lab = {r["stem"]: r["sea_state"] for r in LABELS}
    order = [r["stem"] for r in PFA]
    cols = [("pfa_global_fp16", "(a) FP16 global"),
            ("pfa_qcert_fp16", "(b) FP16 QCert"),
            ("pfa_global_int8", "(c) INT8 global"),
            ("pfa_qcert_int8", "(d) INT8 QCert")]
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.2), sharex=True, sharey=True)
    for ax, (col, ttl) in zip(axes.flat, cols):
        x = np.arange(len(order))
        y = np.array([float(r[col]) for r in PFA]) / ALPHA
        for b, c in C_BUCKET.items():
            m = np.array([lab[s] == b for s in order])
            ax.scatter(x[m], y[m], s=8, c=c, lw=0)
        ax.axhline(1.0, color="k", lw=0.8, ls="--")
        ax.axhline(2.0, color="#d62728", lw=0.8, ls=":")
        ax.set_yscale("log")
        ax.set_title(ttl, fontsize=9)
        ax.set_xlabel("frame index", fontsize=8)
    axes[0, 0].set_ylabel(r"$P_{\rm fa}/\alpha$")
    axes[1, 0].set_ylabel(r"$P_{\rm fa}/\alpha$")
    handles = [Line2D([], [], color=C_BUCKET[b], marker="o", ls="", label=b) for b in C_BUCKET]
    axes[0, 1].legend(handles=handles, fontsize=7, frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "s4_perframe_pfa_4config.png"))
    plt.close(fig)


# ---------------------------------------------------------------- s5
def fig_s5_stress_R():
    """E2 压测集（c1/c2/c3 合成压测，300 场景）逐场景 R 分布 + 漏检率。"""
    rows = json.load(open(os.path.join(CODE, "results", "e2", "e2_per_scene.json"),
                           encoding="utf-8"))
    rs = np.array([r["R"] for r in rows if np.isfinite(r["R"])])
    n_inf = sum(1 for r in rows if not np.isfinite(r["R"]))
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2))
    ax = axes[0]
    ax.hist(rs, bins=40, color="#1f77b4", alpha=0.85, edgecolor="w")
    ax.axvline(1.0, color="k", ls="--", lw=0.8)
    ax.set_xlabel(r"per-scene $R = P_{\rm fa}^{\rm int8}/P_{\rm fa}^{\rm fp16}$")
    ax.set_ylabel("scenes")
    ax.set_title(f"(a) stress set ({len(rows)} scenes, {n_inf} with $R=\\infty$)", fontsize=9)
    ax = axes[1]
    miss = np.array([r["miss_test"] for r in rows])
    stems = [r["stem"] for r in rows]
    sets = sorted({s.split("_")[0] for s in stems})
    data = [[r["miss_test"] for r in rows if r["stem"].startswith(s)] for s in sets]
    bp = ax.boxplot(data, tick_labels=[f"{s} (n={len(d)})" for s, d in zip(sets, data)],
                    widths=0.5, patch_artist=True)
    for p, c in zip(bp["boxes"], ["#1f77b4", "#ff7f0e", "#2ca02c"]):
        p.set_facecolor(c); p.set_alpha(0.6)
    ax.set_ylabel("miss rate (INT8)")
    ax.set_title("(b) miss rate by stress set", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "s5_stress_R_dist.png"))
    plt.close(fig)


if __name__ == "__main__":
    fig_b3_gallery()
    fig_b4_calib_gallery()
    fig_h3_heatmap_gallery()
    fig_h4_overlay()
    fig_s4_pfa_4config()
    fig_s5_stress_R()
    print("rich figures ->", OUT)
