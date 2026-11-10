"""YOLO 检测输出解码 + NMS + 预处理 (纯 numpy + PIL, 无需 cv2, 可在 Atlas 开发板运行)。

适配 ultralytics v8/v11 检测头导出的两种形式:
- 单输出: [1, 4+nc, N] (默认, 三个尺度的预测已拼接)
- 多输出: 3 个步长头 [1, 4+nc, N_i], 按元素数从大到小拼接
"""
from pathlib import Path

import numpy as np
from PIL import Image

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def load_image_rgb(path) -> np.ndarray:
    """读图 -> uint8 HWC RGB。"""
    with Image.open(Path(path)) as im:
        im.load()
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def letterbox(img: np.ndarray, new_shape=640, color=114):
    """等比缩放 + 居中填充到 new_shape(默认 640x640, 与训练/静态输入 OM 一致)。

    必须严格输出 ``(new_shape, new_shape, 3)``: OM 是静态 shape, 少补像素会让
    ``infer()`` 只拷贝一部分数据, 剩余缓冲区是未初始化内存, 预测直接失效。

    返回 (padded, ratio, (pad_left, pad_top))。
    """
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    h0, w0 = img.shape[:2]
    r = min(new_shape[0] / h0, new_shape[1] / w0)
    new_w, new_h = int(round(w0 * r)), int(round(h0 * r))
    if (new_w, new_h) != (w0, h0):
        img = np.asarray(Image.fromarray(img).resize((new_w, new_h), Image.BILINEAR))
    dw, dh = (new_shape[1] - new_w) / 2, (new_shape[0] - new_h) / 2
    left, top = int(round(dw - 0.1)), int(round(dh - 0.1))
    right, bottom = int(round(dw + 0.1)), int(round(dh + 0.1))
    padded = np.full((new_h + top + bottom, new_w + left + right, 3), color, dtype=np.uint8)
    padded[top:top + new_h, left:left + new_w] = img
    return padded, r, (left, top)


def scale_coords(boxes: np.ndarray, img0_shape, pad_info):
    """letterbox 坐标系 xyxy -> 原图坐标。pad_info = (ratio, dw, dh)。"""
    ratio, dw, dh = pad_info
    boxes = boxes.copy().astype(np.float32)
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dw) / ratio
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dh) / ratio
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, img0_shape[1])
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, img0_shape[0])
    return boxes


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thres=0.45):
    """标准贪心 NMS。boxes: (N,4) xyxy。"""
    if len(boxes) == 0:
        return np.zeros(0, dtype=int)
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / np.maximum(areas[i] + areas[order[1:]] - inter, 1e-9)
        order = order[1:][iou <= iou_thres]
    return np.array(keep, dtype=int)


def infer_head_sizes(n_total: int, imgsz: int = 640):
    """单输出 (8400) 时反推三个尺度的 (h, w)；推不出则返回 None。

    顺序与 ultralytics 拼接一致：stride 8 -> 16 -> 32（即 80x80 -> 40x40 -> 20x20）。
    """
    strides = (8, 16, 32)
    sizes = [(imgsz // s, imgsz // s) for s in strides]
    if sum(h * w for h, w in sizes) == n_total:
        return sizes
    return None


def _square_hw(n: int):
    """n 个网格能否还原成 h x w（h==w）的方阵，能则返回 (h, w) 否则 (0, 0)。"""
    s = int(round(n ** 0.5))
    return (s, s) if s * s == n else (0, 0)


def decode_dense(outputs, nc=1, imgsz=640):
    """绕过阈值过滤与 NMS，返回稠密分数场。

    这是 v3（QCert）的入口：P_fa 契约只看**每个网格的分数**，
    与被 NMS 留下的框无关。

    Returns
    -------
    dict:
        ``prob``       (N,) float32   逐网格 max_c sigmoid(cls)，即 s(u)
        ``logit``      (N,) float32   sigmoid 之前的概率对数比（诊断用）
        ``box``        (N,4) float32  xywh（letterbox 坐标系）
        ``head_sizes`` list[(h,w)]    各尺度网格尺寸，顺序与拼接一致
        ``n_per_head`` list[int]
    """
    heads = []
    for o in outputs:
        a = np.asarray(o)
        if a.ndim == 4:  # (1, 4+nc, h, w)
            h, w = int(a.shape[2]), int(a.shape[3])
            heads.append((a.reshape(a.shape[1], h * w), h, w))
        elif a.ndim == 3:  # (1, 4+nc, N)
            h, w = _square_hw(a.reshape(a.shape[1], -1).shape[1])
            heads.append((a.reshape(a.shape[1], -1), h, w))
        else:
            raise ValueError(f"无法识别的输出维度: {a.shape}")
    if len(heads) > 1:
        heads.sort(key=lambda t: t[0].shape[1], reverse=True)
    elif heads[0][1] == 0:
        hs = infer_head_sizes(heads[0][0].shape[1], imgsz)
        if hs is not None:
            raw = heads[0][0]
            off = 0
            split = []
            for h, w in hs:
                split.append((raw[:, off:off + h * w], h, w))
                off += h * w
            heads = split

    pred = np.concatenate([t[0] for t in heads], axis=1)
    prob = pred[4:4 + nc].max(axis=0).astype(np.float32)
    box = pred[:4].T.astype(np.float32)
    p = np.clip(prob.astype(np.float64), 1e-7, 1.0 - 1e-7)
    logit = np.log(p / (1.0 - p)).astype(np.float32)
    head_sizes = [(int(h), int(w)) for _, h, w in heads]
    return {
        "prob": prob,
        "logit": logit,
        "box": box,
        "head_sizes": head_sizes,
        "n_per_head": [int(h.shape[1]) for h, _, _ in heads],
    }


def decode_outputs(outputs, nc=1, conf_thres=0.25, iou_thres=0.45):
    """ACL 输出 -> (boxes_xyxy, scores, class_ids), 已做置信度过滤与 NMS。

    outputs: list[np.ndarray], 每个元素为该输出的完整 shape(如 (1,5,8400))。
    """
    if len(outputs) > 1:
        heads = sorted((o.reshape(1, o.shape[1], -1) for o in outputs),
                       key=lambda t: t.shape[1] * t.shape[2], reverse=True)
        pred = np.concatenate(heads, axis=2)[0]
    else:
        o = outputs[0]
        pred = o.reshape(o.shape[0], o.shape[1], -1)[0]

    box = pred[:4].T.astype(np.float32)
    # ultralytics v8/v11 的 Detect 头在 forward 内已做 sigmoid, 导出 ONNX 后输出即概率,
    # 这里不能再 sigmoid(否则 conf 会被压到 0.5 附近, mAP 完全失真)。
    scores_all = pred[4:].T.astype(np.float32)  # (N, nc)
    cls_ids = scores_all.argmax(axis=1)
    confs = scores_all.max(axis=1)
    mask = confs >= conf_thres
    if not mask.any():
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, int)

    box, confs, cls_ids = box[mask], confs[mask], cls_ids[mask]
    box[:, 0] -= box[:, 2] / 2  # xywh -> xyxy
    box[:, 1] -= box[:, 3] / 2
    box[:, 2] += box[:, 0]
    box[:, 3] += box[:, 1]

    keep = nms(box, confs, iou_thres)
    return box[keep], confs[keep], cls_ids[keep]


def preprocess(img, imgsz=640):
    """HWC RGB uint8 或图片路径 -> ((1,3,H,W) float32 NCHW, ratio, (dw, dh))。"""
    if isinstance(img, (str, Path)):
        img = load_image_rgb(img)
    img, ratio, pad = letterbox(img, imgsz)
    x = np.ascontiguousarray(img.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
    return x, ratio, pad
