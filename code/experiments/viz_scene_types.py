# -*- coding: utf-8 -*-
"""远洋/近岸场景类型可视化 -> paper/figs/samples/

n1 近岸识别框图（陆地/海岸线场景，GT vs INT8）
n2 远洋识别框图（开阔海域，GT vs INT8）
n3 热力图叠加底图：近岸 vs 远洋（陆地杂波响应对比）
n4 逐帧 P_fa 与候选数：按场景类型分面
"""
import csv
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
C_TYPE = {"nearshore": "#d62728", "offshore": "#1f77b4"}

plt.rcParams.update({
    "font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
    "figure.dpi": 200, "savefig.bbox": "tight",
})


def load_csv(path):
    with open(path, encoding="utf-8") as f:
        rows = [ln.strip().split(",") for ln in f if ln.strip()]
    hdr = rows[0]
    return [dict(zip(hdr, r)) for r in rows[1:]]


STYPE = {r["stem"]: r["scene_type"] for r in
         load_csv(os.path.join(CODE, "results", "e7", "e7_scene_type.csv"))}


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


def draw_boxes(ax, boxes, color):
    for x1, y1, x2, y2 in boxes:
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec=color, lw=1.4))


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


def typical(stype, k=3):
    """按 GT 船数+检出数选典型帧。"""
    scored = []
    for st, t in STYPE.items():
        if t != stype:
            continue
        gtb = load_gt(st)
        if not gtb:
            continue
        db = dets(st)[0]
        if not db:
            continue
        scored.append((len(gtb) + len(db), st))
    scored.sort(reverse=True)
    return [st for _, st in scored[:k]]


HAN = [Line2D([], [], color="#00ff66", label="GT"),
       Line2D([], [], color="#ff7f0e", label="INT8 det")]


# ---------------------------------------------------------------- n1 / n2
def gallery(stype, fname, title):
    frames = typical(stype, 3)
    print(stype, "typical:", frames)
    fig, axes = plt.subplots(2, 3, figsize=(12.4, 8.0))
    for j, st in enumerate(frames):
        for i, cfg in enumerate(["int8_calmA", "int8_roughB"]):
            ax = axes[i, j]
            ax.imshow(load_img(st))
            ax.grid(False)
            gtb = load_gt(st)
            db = dets(st, cfg)[0]
            draw_boxes(ax, gtb, "#00ff66")
            draw_boxes(ax, db, "#ff7f0e")
            ax.set_xlabel(f"{st} | {cfg} (gt={len(gtb)}, det={len(db)})", fontsize=9)
    axes[0, 0].set_ylabel("calmA calibration", fontsize=10)
    axes[1, 0].set_ylabel("roughB calibration", fontsize=10)
    axes[0, 2].legend(handles=HAN, loc="upper right", fontsize=9, frameon=False)
    fig.suptitle(title, fontsize=11)
    fig.savefig(os.path.join(OUT, fname))
    plt.close(fig)
    return frames


# ---------------------------------------------------------------- n3
def overlay_scene(ns_frame, off_frame):
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.4))
    for ax, st, lab in [(axes[0], ns_frame, "nearshore"),
                        (axes[1], off_frame, "offshore")]:
        img = load_img(st)
        ih, iw = img.shape[:2]
        _, pad = letterbox(img)
        d = np.load(os.path.join(DUMPS, "int8_calmA", st + ".npz"), allow_pickle=True)
        p3 = d["prob"][: 80 * 80].reshape(80, 80)
        h640 = cv2.resize(p3, (IMGSZ, IMGSZ), interpolation=cv2.INTER_CUBIC)
        l, t, nu = pad
        h = cv2.resize(h640[t:t + nu[1], l:l + nu[0]], (iw, ih),
                       interpolation=cv2.INTER_LINEAR)
        ax.imshow(img)
        ax.imshow(h, cmap="inferno", alpha=0.45)
        draw_boxes(ax, load_gt(st), "#00ff66")
        draw_boxes(ax, json.loads(str(d["meta"])).get("boxes", []), "#00ffff")
        ax.grid(False)
        ax.set_xlabel(f"{st} ({lab})", fontsize=9)
    fig.savefig(os.path.join(OUT, "n3_heatmap_scene_overlay.png"))
    plt.close(fig)


# ---------------------------------------------------------------- n4
def pfa_scene():
    lab7 = {r["stem"]: r["sea_state"] for r in
            load_csv(os.path.join(CODE, "results", "e7", "e7_labels.csv"))}
    pfa = load_csv(os.path.join(CODE, "results", "e7", "e7_per_frame_pfa.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.6))
    # (a) global int8 按场景类型
    ax = axes[0]
    x = np.arange(len(pfa))
    for t, c in C_TYPE.items():
        m = np.array([STYPE[r["stem"]] == t for r in pfa])
        y = np.array([float(r["pfa_global_int8"]) for r in pfa])[m] / ALPHA
        ax.scatter(x[m], y, s=8, c=c, lw=0, alpha=0.5, label=t)
    ax.axhline(1.0, color="k", lw=0.8, ls="--")
    ax.axhline(2.0, color="#d62728", lw=0.8, ls=":")
    ax.set_yscale("log")
    ax.set_xlabel("frame index")
    ax.set_ylabel(r"$P_{\rm fa}/\alpha$ (INT8 global)")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("(a) per-frame $P_{\\rm fa}$ by scene type", fontsize=9)
    # (b) 候选数分布箱线
    ax = axes[1]
    data = [[int(r[f"n_cand_global_int8"]) for r in pfa if STYPE[r["stem"]] == t]
            for t in ["nearshore", "offshore"]]
    bp = ax.boxplot(data, tick_labels=[f"nearshore\n(n={len(data[0])})",
                                       f"offshore\n(n={len(data[1])})"],
                    widths=0.5, patch_artist=True)
    for p, c in zip(bp["boxes"], ["#d62728", "#1f77b4"]):
        p.set_facecolor(c); p.set_alpha(0.6)
    ax.set_ylabel("candidates per frame (INT8 global)")
    ax.set_title("(b) candidate load", fontsize=9)
    # (c) 海况 x 场景类型堆叠
    ax = axes[2]
    buckets = ["rough", "medium", "calm"]
    ns = [sum(1 for r in pfa if STYPE[r["stem"]] == "nearshore" and lab7[r["stem"]] == b)
          for b in buckets]
    off = [sum(1 for r in pfa if STYPE[r["stem"]] == "offshore" and lab7[r["stem"]] == b)
           for b in buckets]
    xb = np.arange(3)
    ax.bar(xb, ns, 0.55, label="nearshore", color="#d62728", alpha=0.85)
    ax.bar(xb, off, 0.55, bottom=ns, label="offshore", color="#1f77b4", alpha=0.85)
    for i in range(3):
        ax.text(i, ns[i] / 2, str(ns[i]), ha="center", color="w", fontsize=8)
        ax.text(i, ns[i] + off[i] / 2, str(off[i]), ha="center", color="w", fontsize=8)
    ax.set_xticks(xb)
    ax.set_xticklabels(buckets)
    ax.set_ylabel("frames")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("(c) sea state x scene type", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "n4_pfa_scene_type.png"))
    plt.close(fig)


if __name__ == "__main__":
    ns_frames = gallery("nearshore", "n1_nearshore_gallery.png",
                        "Nearshore: coastline / land clutter scenes (GT vs INT8)")
    off_frames = gallery("offshore", "n2_offshore_gallery.png",
                         "Offshore: open-sea scenes (GT vs INT8)")
    overlay_scene(ns_frames[0], off_frames[0])
    pfa_scene()
    print("scene-type figures ->", OUT)
