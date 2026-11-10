# -*- coding: utf-8 -*-
"""板上一次性探测: QuantConfig 默认值 + yolo11n.onnx 的 depthwise conv 数量。"""
import inspect

from modelslim.onnx.post_training_quant import QuantConfig
print("QuantConfig:", inspect.signature(QuantConfig))

import onnx
from onnx import numpy_helper

m = onnx.load("/root/asc_sar/models/yolo11n.onnx")
inits = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
convs = [n for n in m.graph.node if n.op_type == "Conv"]
dw = [n.name for n in convs
      if n.input[1] in inits and inits[n.input[1]].ndim == 4
      and inits[n.input[1]].shape[1] == 1]
print(f"total Conv: {len(convs)}, depthwise-like: {len(dw)}")
if dw:
    print("depthwise:", dw[:10])
