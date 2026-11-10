# -*- coding: utf-8 -*-
"""SSDD 模型 INT8 vs FP16 部署口径对比: 基于 acl_infer --save-dir 的预测 txt。

回答: mAP 只掉 0.3pp, 但部署阈值(conf=0.25/0.5)下 INT8 是否还能检出目标。
"""
from pathlib import Path

import numpy as np

R = Path("/root/asc_sar/results")


def load_preds(d):
    """{stem: (N,6) [cls x1 y1 x2 y2 conf]}"""
    out = {}
    for p in sorted(Path(R / d).glob("*.txt")):
        rows = [list(map(float, ln.split())) for ln in p.read_text().splitlines() if ln.strip()]
        out[p.stem] = np.array(rows, dtype=np.float64).reshape(-1, 6)
    return out


def iou(a, b):
    """a:(N,4) b:(M,4) -> (N,M)"""
    ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ix1 = np.maximum(ax1[:, None], bx1[None, :])
    iy1 = np.maximum(ay1[:, None], by1[None, :])
    ix2 = np.minimum(ax2[:, None], bx2[None, :])
    iy2 = np.minimum(ay2[:, None], by2[None, :])
    iw = np.clip(ix2 - ix1, 0, None)
    ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)          # (N,)
    area_b = (bx2 - bx1) * (by2 - by1)          # (M,)
    ua = area_a[:, None] + area_b[None, :] - inter
    return inter / np.clip(ua, 1e-9, None)


def main():
    f16 = load_preds("preds_ssdd_fp16")
    i8 = load_preds("preds_ssdd_int8")
    print(f"图数: fp16={len(f16)} int8={len(i8)}")

    for name, preds in (("fp16", f16), ("int8", i8)):
        n_all = sum(len(v) for v in preds.values())
        n25 = sum(int((v[:, 5] >= 0.25).sum()) for v in preds.values())
        n50 = sum(int((v[:, 5] >= 0.50).sum()) for v in preds.values())
        top1 = np.array([v[:, 5].max() if len(v) else 0.0 for v in preds.values()])
        print(f"\n[{name}] 候选总数={n_all}  >=0.25: {n25}  >=0.50: {n50}")
        print(f"     每图top-1 conf: mean={top1.mean():.4f} p50={np.median(top1):.4f} "
              f"max={top1.max():.4f}  top1>=0.25 图数={(top1 >= 0.25).sum()}")

    # ---- 部署阈值检出对比: fp16 conf>=0.25 的框, int8 在任意 conf 下能否找到 IoU>=0.5 ----
    print("\n[部署阈值 0.25 口径] fp16 框 -> int8 检出(IoU>=0.5, 不限 conf):")
    tot_f, tot_m, matched_25, matched_50 = 0, 0, 0, 0
    img_lost = 0
    for stem, pf in f16.items():
        pf25 = pf[pf[:, 5] >= 0.25]
        pi = i8.get(stem, np.zeros((0, 6)))
        if len(pf25) == 0:
            continue
        tot_f += len(pf25)
        if len(pf) and len(pi):
            m = iou(pf25[:, 1:5], pi[:, 1:5]).max(axis=1)
        else:
            m = np.zeros(len(pf25))
        matched_25 += int((m >= 0.25).sum())
        matched_50 += int((m >= 0.5).sum())
        if (m >= 0.5).sum() < len(pf25):
            img_lost += 1
    print(f"  fp16 框数={tot_f}  int8 IoU>=0.5 检出={matched_50} "
          f"({matched_50 / max(tot_f, 1):.1%})  IoU>=0.25={matched_25} "
          f"({matched_25 / max(tot_f, 1):.1%})")
    print(f"  至少丢一个目标的图: {img_lost}")

    # ---- int8 自身在 0.25 阈值下的直接检出 ----
    n25_i8 = sum(int((v[:, 5] >= 0.25).sum()) for v in i8.values())
    print(f"\n[int8 @conf>=0.25 直接候选] {n25_i8} 个")


if __name__ == "__main__":
    main()
