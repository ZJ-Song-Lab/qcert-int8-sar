"""mAP 评估 (VOC 风格, 单类别 ship)。

既可被 deploy/acl_infer.py 调用, 也可独立使用:
    python tools/eval_map.py --pred-dir <预测txt目录> --gt-dir <标签txt目录> \
        --img-dir <图像目录(用于归一化反算)> [--conf 0.001] [--iou 0.5] [--map5095]

预测txt格式: class x1 y1 x2 y2 conf (绝对像素坐标)
GT txt格式:   class xc yc w h (归一化)
"""
import argparse
from pathlib import Path

import numpy as np


def load_yolo_gt(label_path, img_w, img_h):
    """GT: YOLO 归一化 -> xyxy 绝对坐标。"""
    boxes = []
    p = Path(label_path)
    if not p.exists():
        return np.zeros((0, 4))
    for line in p.read_text().strip().splitlines():
        c, xc, yc, w, h = line.split()[:5]
        xc, yc, w, h = float(xc) * img_w, float(yc) * img_h, float(w) * img_w, float(h) * img_h
        boxes.append([xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2])
    return np.array(boxes, dtype=np.float32).reshape(-1, 4)


def load_pred(pred_path):
    """预测: class x1 y1 x2 y2 conf。"""
    p = Path(pred_path)
    if not p.exists():
        return np.zeros((0, 6))
    rows = [list(map(float, l.split())) for l in p.read_text().strip().splitlines()]
    return np.array(rows, dtype=np.float32).reshape(-1, 6)


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    xx1 = np.maximum(a[:, None, 0], b[None, :, 0])
    yy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    xx2 = np.minimum(a[:, None, 2], b[None, :, 2])
    yy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
    area_a = ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, None]
    area_b = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))[None, :]
    return inter / np.maximum(area_a + area_b - inter, 1e-9)


def voc_ap(rec, prec):
    """VOC2010 全点插值 AP。"""
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def evaluate(preds, gts, iou_thres=0.5):
    """preds/gts: {img_stem: (N,4|6)}。返回 (ap, precision, recall)。"""
    # 汇总所有预测并按置信度排序
    entries = []  # (conf, img, pred_idx)
    for stem, pred in preds.items():
        for j in range(len(pred)):
            entries.append((pred[j, 5], stem, j))
    entries.sort(key=lambda e: -e[0])

    n_gt = sum(len(v) for v in gts.values())
    tp = np.zeros(len(entries))
    fp = np.zeros(len(entries))
    matched = {stem: np.zeros(len(v), dtype=bool) for stem, v in gts.items()}

    for k, (_, stem, j) in enumerate(entries):
        gt = gts.get(stem, np.zeros((0, 4)))
        if len(gt) == 0:
            fp[k] = 1
            continue
        ious = iou_matrix(preds[stem][j, 1:5].reshape(1, 4), gt)[0]
        jmax = ious.argmax()
        if ious[jmax] >= iou_thres and not matched[stem][jmax]:
            tp[k] = 1
            matched[stem][jmax] = True
        else:
            fp[k] = 1

    if n_gt == 0:
        return 0.0, 0.0, 0.0
    tp_cum, fp_cum = np.cumsum(tp), np.cumsum(fp)
    rec = tp_cum / n_gt
    prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    ap = voc_ap(rec, prec)
    return ap, float(prec[-1]) if len(prec) else 0.0, float(rec[-1]) if len(rec) else 0.0


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--pred-dir", required=True)
    ap_.add_argument("--gt-dir", required=True)
    ap_.add_argument("--img-dir", required=True)
    ap_.add_argument("--conf", type=float, default=0.001)
    ap_.add_argument("--iou", type=float, default=0.5)
    ap_.add_argument("--map5095", action="store_true")
    args = ap_.parse_args()

    from PIL import Image

    preds, gts = {}, {}
    for img in sorted(Path(args.img_dir).iterdir()):
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
            continue
        with Image.open(img) as im:
            iw, ih = im.size
        gts[img.stem] = load_yolo_gt(Path(args.gt_dir) / (img.stem + ".txt"), iw, ih)
        pred = load_pred(Path(args.pred_dir) / (img.stem + ".txt"))
        preds[img.stem] = pred[pred[:, 5] >= args.conf]

    ap50, p, r = evaluate(preds, gts, args.iou)
    print(f"mAP@0.5 = {ap50:.4f}  (P={p:.4f} R={r:.4f})")
    if args.map5095:
        aps = []
        for t in np.arange(0.5, 1.0, 0.05):
            aps.append(evaluate(preds, gts, float(t))[0])
        print(f"mAP@0.5:0.95 = {np.mean(aps):.4f}")


if __name__ == "__main__":
    main()
