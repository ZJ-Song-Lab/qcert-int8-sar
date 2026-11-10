# -*- coding: utf-8 -*-
"""Generate sample figures for the experiment chapter (E1/E5/E7 evidence).

Samples (2-3 per class) -> paper/figs/samples/
  Scatter:  per-frame P_fa vs bucketed sea state; global-vs-QCert candidate load; bucket feature space
  Heatmap:  anchor objectness (80x80) over the image, calmA vs roughB calibration
  Box:      GT vs INT8 detections across sea-state buckets; calmA vs roughB same frame
"""
import json
import os

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = r"D:\dinghuibianyuan"
CODE = os.path.join(ROOT, "neural networks", "code")
SSDD = os.path.join(ROOT, "neural networks", "asc-sar", "datasets", "ssdd")
DUMPS = os.path.join(CODE, "dumps", "e5s1")
OUT = os.path.join(ROOT, "neural networks", "paper", "figs", "samples")
ALPHA = 1e-3

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
    recs = []
    for r in rows[1:]:
        d = dict(zip(hdr, r))
        recs.append(d)
    return recs


def load_npz(config, stem):
    p = os.path.join(DUMPS, config, stem + ".npz")
    return np.load(p, allow_pickle=True)


def load_img(stem):
    p = os.path.join(SSDD, "images", "test", stem + ".jpg")
    return cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)


def load_gt(stem):
    """YOLO txt -> xyxy in original image pixels."""
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


def head_grids():
    return [80, 40, 20]


# ---------------------------------------------------------------- scatter 1
def fig_s1_perframe_pfa(labels, pfa):
    """Per-frame P_fa/alpha by sea-state bucket: INT8 global vs QCert (F3 evidence)."""
    lab = {r["stem"]: r["sea_state"] for r in labels}
    order = [r["stem"] for r in pfa]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4), sharey=True)
    for ax, col, ttl in zip(
            axes, ["pfa_global_int8", "pfa_qcert_int8"],
            ["(a) INT8 global threshold", "(b) INT8 QCert"]):
        x = np.arange(len(order))
        y = np.array([float(r[col]) for r in pfa]) / ALPHA
        for b, c in C_BUCKET.items():
            m = np.array([lab[s] == b for s in order])
            ax.scatter(x[m], y[m], s=10, c=c, lw=0, label=b)
        ax.axhline(1.0, color="k", lw=0.8, ls="--")
        ax.axhline(2.0, color="#d62728", lw=0.8, ls=":")
        ax.set_yscale("log")
        ax.set_title(ttl)
        ax.set_xlabel("frame index")
    axes[0].set_ylabel(r"$P_{\rm fa}/\alpha$")
    axes[0].text(2, 1.06, r"$\alpha$", fontsize=8)
    axes[0].text(2, 2.1, r"$2\alpha$", fontsize=8, color="#d62728")
    axes[1].legend(loc="upper right", frameon=False, fontsize=8, ncol=3)
    fig.savefig(os.path.join(OUT, "s1_perframe_pfa_int8.png"))
    plt.close(fig)


# ---------------------------------------------------------------- scatter 2
def fig_s2_candidate_load(pfa):
    """Candidate count global vs QCert (log-log), fp16 + int8 (E7 load decoupling)."""
    fig, ax = plt.subplots(figsize=(5.0, 4.4))
    for gcol, qcol, c, lab in [
            ("n_cand_global_fp16", "n_cand_qcert_fp16", "#1f77b4", "FP16"),
            ("n_cand_global_int8", "n_cand_qcert_int8", "#d62728", "INT8")]:
        x = np.array([float(r[gcol]) for r in pfa])
        y = np.array([float(r[qcol]) for r in pfa])
        ax.scatter(x, y, s=7, c=c, lw=0, alpha=0.4, label=lab)
    lim = [0.7, 1200]
    ax.plot(lim, lim, "k--", lw=0.8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("candidates, global threshold")
    ax.set_ylabel("candidates, QCert")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.savefig(os.path.join(OUT, "s2_cand_global_vs_qcert.png"))
    plt.close(fig)


# ---------------------------------------------------------------- scatter 3
def fig_s3_bucket_space(labels):
    """Sea-state bucket feature space: cv vs weibull_c, kmeans labels."""
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    cv = np.array([float(r["cv"]) for r in labels])
    wc = np.array([float(r["weibull_c"]) for r in labels])
    ct = np.array([float(r["contrast"]) for r in labels])
    for b, c in C_BUCKET.items():
        m = np.array([r["sea_state"] == b for r in labels])
        ax.scatter(wc[m], cv[m], s=14 + 40 * (ct[m] - ct.min()) / (ct.max() - ct.min() + 1e-9),
                   c=c, lw=0, alpha=0.75, label=f"{b} (n={m.sum()})")
    ax.set_xlabel("Weibull shape $c$")
    ax.set_ylabel("clutter cv")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(os.path.join(OUT, "s3_bucket_feature_space.png"))
    plt.close(fig)


# ---------------------------------------------------------------- heatmap 1
def fig_h1_score_heatmap(stem, config="int8_calmA"):
    """Anchor objectness (80x80 P3 head) as heatmap overlaid on the image."""
    d = load_npz(config, stem)
    m = json.loads(str(d["meta"]))
    img = load_img(stem)
    prob = d["prob"]
    grids = head_grids()
    off = 0
    h = None
    for g in grids:
        n = g * g
        off += n
    # use P3 (first 6400) only
    p3 = prob[: 80 * 80].reshape(80, 80)
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.5))
    axes[0].imshow(img)
    for b in m.get("boxes", []):
        x1, y1, x2, y2 = b
        axes[0].add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                        fill=False, ec="#00ff66", lw=1.6))
    axes[0].set_title(f"INT8 input & detections ({stem})")
    axes[0].grid(False)
    im = axes[1].imshow(p3, cmap="inferno",
                        extent=[0, m["imgsz"], m["imgsz"], 0])
    # un-letterbox: show input-space overlay is complex; keep model-space grid
    axes[1].set_title("objectness heatmap (80x80, P3)")
    axes[1].grid(False)
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    fig.savefig(os.path.join(OUT, f"h1_heatmap_{stem}_{config}.png"))
    plt.close(fig)


# ---------------------------------------------------------------- heatmap 2
def fig_h2_calib_compare(stem):
    """Same frame, int8_calmA vs int8_roughB objectness heatmaps (calibration shift)."""
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6))
    for ax, cfg in zip(axes, ["int8_calmA", "int8_roughB"]):
        d = load_npz(cfg, stem)
        p3 = d["prob"][: 80 * 80].reshape(80, 80)
        im = ax.imshow(p3, cmap="inferno")
        nb = len(json.loads(str(d["meta"])).get("boxes", []))
        ax.set_title(f"{cfg}  (boxes={nb})")
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.savefig(os.path.join(OUT, f"h2_calib_compare_{stem}.png"))
    plt.close(fig)


# ---------------------------------------------------------------- box figs
def _draw_boxes(ax, boxes, color, label=None):
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                   fill=False, ec=color, lw=1.6))
        if label and i == 0:
            ax.legend(handles=[Line2D([], [], color=color, label=label)],
                      loc="lower right", fontsize=7, frameon=False)


def fig_b1_gt_vs_det(labels, cfg="int8_calmA"):
    """GT vs INT8 detections, one frame per sea-state bucket."""
    buckets = {}
    for r in labels:
        buckets.setdefault(r["sea_state"], []).append(r["stem"])
    stems = [buckets[b][max(0, len(buckets[b]) // 2)] for b in ["rough", "medium", "calm"]]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3))
    for ax, st in zip(axes, stems):
        img = load_img(st)
        ax.imshow(img)
        ax.grid(False)
        ax.set_xlabel(st)
        _draw_boxes(ax, load_gt(st), "#00ff66", "GT")
        d = load_npz(cfg, st)
        boxes = json.loads(str(d["meta"])).get("boxes", [])
        _draw_boxes(ax, boxes, "#ff7f0e", "INT8")
    fig.savefig(os.path.join(OUT, f"b1_gt_vs_int8_{cfg}.png"))
    plt.close(fig)


def fig_b2_calib_boxes(labels, cfgs=("int8_calmA", "int8_roughB")):
    """Same frame (multi-ship), calmA vs roughB detection boxes."""
    # pick a rough frame with most GT boxes
    best, nb = None, -1
    for r in labels:
        if r["sea_state"] != "rough":
            continue
        st = r["stem"]
        n = len(load_gt(st))
        if n > nb:
            best, nb = st, n
    stem = best
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4))
    img = load_img(stem)
    for ax, cfg in zip(axes, cfgs):
        ax.imshow(img)
        ax.grid(False)
        _draw_boxes(ax, load_gt(stem), "#00ff66", "GT")
        d = load_npz(cfg, stem)
        boxes = json.loads(str(d["meta"])).get("boxes", [])
        _draw_boxes(ax, boxes, "#ff7f0e", cfg)
        ax.set_xlabel(f"{stem} ({cfg})")
    fig.savefig(os.path.join(OUT, f"b2_calib_boxes_{stem}.png"))
    plt.close(fig)


if __name__ == "__main__":
    labels = load_csv(os.path.join(CODE, "results", "e7", "e7_labels.csv"))
    pfa = load_csv(os.path.join(CODE, "results", "e7", "e7_per_frame_pfa.csv"))

    fig_s1_perframe_pfa(labels, pfa)
    fig_s2_candidate_load(pfa)
    fig_s3_bucket_space(labels)

    # pick a medium frame with detections for heatmaps
    hm_stem = None
    for r in labels:
        if r["sea_state"] == "medium":
            d = load_npz("int8_calmA", r["stem"])
            if json.loads(str(d["meta"])).get("boxes"):
                hm_stem = r["stem"]
                break
    fig_h1_score_heatmap(hm_stem)
    fig_h2_calib_compare(hm_stem)

    fig_b1_gt_vs_det(labels)
    fig_b2_calib_boxes(labels)
    print("samples ->", OUT)
    print("heatmap frame:", hm_stem)
