# -*- coding: utf-8 -*-
"""板上一次性脚本: CPU 仿真量化精度(ORT QOperator), 定位并修 A 问题。

背景: 板上 INT8 OM 分数场崩坏(前景 corr 0.18, conf>=0.25 候选 35->2)。
CPU 仿真(同 scale 计算方式, 同批 232 张图)基线: corr 0.632 / 候选 35->18 /
argmax 37.5%, 由此分叉出两个问题:
  A) MinMax scale 本身对检测头就有实质损伤(候选砍半) —— 本脚本修它;
  B) 板上 OM 相比 CPU 仿真又额外崩一截(18->2) —— A 修完后重跑链路复测。

实验矩阵(yolo 量化标准做法: 排除检测头敏感层):
  excl-final7  : 排除 3 尺度 cv2/cv3 最终输出 conv + dfl(7 节点);
  excl-model23 : 排除整个检测头 /model.23(25 节点)。
"""
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import (CalibrationDataReader, QuantFormat,
                                      QuantType, quantize_static)

sys.path.insert(0, "/root/asc_sar/deploy")
from postprocess import preprocess  # noqa: E402

MODEL = "/root/asc_sar/models/yolo11n.onnx"
NODES = json.load(open("/root/asc_sar/deploy/quant_nodes_88.json"))
IMG_DIR = Path("/root/asc_sar/data/calib")
EVAL_DIR = Path("/root/asc_sar/data/images")
CONF_TH, FG_TH = 0.25, 0.1

FINAL7 = [
    "/model.23/cv2.0/cv2.0.2/Conv",
    "/model.23/cv3.0/cv3.0.2/Conv",
    "/model.23/cv2.1/cv2.1.2/Conv",
    "/model.23/cv3.1/cv3.1.2/Conv",
    "/model.23/cv2.2/cv2.2.2/Conv",
    "/model.23/cv3.2/cv3.2.2/Conv",
    "/model.23/dfl/conv/Conv",
]
CONFIGS = [
    ("excl-final7", FINAL7),
    ("excl-model23", ["/model.23/"]),
]


class Reader(CalibrationDataReader):
    def __init__(self, xs):
        self.xs, self.i = xs, 0

    def get_next(self):
        if self.i >= len(self.xs):
            return None
        x = self.xs[self.i]
        self.i += 1
        return {"images": x}


so = ort.SessionOptions()
so.log_severity_level = 3
s0 = ort.InferenceSession(MODEL, so, providers=["CPUExecutionProvider"])
in_name = s0.get_inputs()[0].name
assert in_name == "images", f"模型输入名是 {in_name}, 脚本按 'images' 写死"

calib_imgs = sorted(p for p in IMG_DIR.iterdir()
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"))
eval_imgs = sorted(p for p in EVAL_DIR.iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"))
print(f"校准图 {len(calib_imgs)} 张, 评估图 {len(eval_imgs)} 张")
xs = [preprocess(p, 640)[0] for p in calib_imgs]
eval_xs = [(p, preprocess(p, 640)[0]) for p in eval_imgs]

# 原始模型基准(只跑一遍)
base = {}
for p, x in eval_xs:
    base[p.stem] = s0.run(None, {"images": x})[0][0, 4:].max(axis=0).astype(
        np.float64)
print("原始模型基准完成")


def evaluate(s1):
    all_mad = n_all = fg_mad = n_fg = 0
    fg_a, fg_b = [], []
    ta, tb, same, ca, cb = [], [], 0, 0, 0
    for p, x in eval_xs:
        a = base[p.stem]
        b = s1.run(None, {"images": x})[0][0, 4:].max(axis=0).astype(np.float64)
        d = np.abs(a - b)
        all_mad += d.sum()
        n_all += d.size
        fg = a > FG_TH
        if fg.any():
            fg_mad += d[fg].sum()
            n_fg += int(fg.sum())
            fg_a.append(a[fg])
            fg_b.append(b[fg])
        ta.append(float(a.max()))
        tb.append(float(b.max()))
        same += int(int(a.argmax()) == int(b.argmax()))
        ca += int((a >= CONF_TH).sum())
        cb += int((b >= CONF_TH).sum())
    fa = np.concatenate(fg_a) if fg_a else np.zeros(1)
    fb = np.concatenate(fg_b) if fg_b else np.zeros(1)
    corr = (np.corrcoef(fa, fb)[0, 1] if fa.size > 1 and fa.std() > 0
            else float("nan"))
    ta_, tb_ = np.asarray(ta), np.asarray(tb)
    td = np.abs(ta_ - tb_)
    n = len(ta)
    return {"mad": all_mad / max(n_all, 1), "fg_mad": fg_mad / max(n_fg, 1),
            "n_fg": n_fg, "corr": corr, "t_mad": td.mean(), "t_max": td.max(),
            "same": same / n, "ca": ca, "cb": cb}


for name, excl in CONFIGS:
    nodes = [n_ for n_ in NODES if not any(e in n_ for e in excl)]
    qpath = f"/root/asc_sar/models/yolo11n_sim_{name}.onnx"
    quantize_static(MODEL, qpath, Reader(xs),
                    quant_format=QuantFormat.QOperator,
                    activation_type=QuantType.QInt8,
                    weight_type=QuantType.QInt8,
                    per_channel=True,
                    op_types_to_quantize=("Conv", "Gemm", "MatMul"),
                    nodes_to_quantize=nodes)
    s1 = ort.InferenceSession(qpath, so, providers=["CPUExecutionProvider"])
    r = evaluate(s1)
    print(f"=== {name}: {len(nodes)}/{len(NODES)} 节点量化 ===")
    print(f"[1] 全场 mean|d| = {r['mad']:.3e}")
    print(f"[2] 前景>{FG_TH} (n={r['n_fg']}): mean|d| = {r['fg_mad']:.4f}"
          f"  corr = {r['corr']:.6f}")
    print(f"[3] top-1: mean|d| = {r['t_mad']:.4f}  max|d| = {r['t_max']:.4f}"
          f"  argmax 同网格率 = {r['same']:.2%}")
    print(f"[4] conf>={CONF_TH} 候选总数: fp32={r['ca']}  quant={r['cb']}")
print("(基线 88 节点全量化: corr=0.632 候选35->18; "
      "板上 INT8 OM: corr=0.180 候选35->2)")
