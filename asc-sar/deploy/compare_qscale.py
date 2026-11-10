# -*- coding: utf-8 -*-
"""板上一次性脚本: B 问题定位——对比 modelslim 输出 ONNX 与 CPU 仿真
QOperator ONNX 的逐节点 scale + 图结构。

链路: modelslim 内部 = ORT quantize_static(与 CPU 仿真同参数/同校准集,
quantize_tool.py L44-52 确认: QOperator/per_channel/QInt8/nodes_to_quantize,
无 calibrate_method/extra_options) → convert_graph_lf_quant() 转
AscendQuant/AscendDequant → reduce_redundant_quant_node()。

判定:
  - 逐 conv 激活 x_scale 与 dequant 乘积(x_scale*w_scale[i]) 一致
    → AscendQuant 转换忠实, B 问题在 ATC/OM 侧;
  - 不一致 → convert_graph_lf_quant 阶段丢/改了 scale, 转换 bug。
同时检查: Conv 权重是否 int8、AscendDequant→AscendQuant 链是否被
reduce_redundant_quant_node 错误折叠(图结构走样)。
"""
from collections import Counter

import numpy as np
import onnx
from onnx import numpy_helper

MS = "/root/asc_sar/models/yolo11n_int8.onnx"      # modelslim 产物(88节点)
SIM = "/root/asc_sar/models/yolo11n_qdq_sim.onnx"  # CPU QOperator 仿真(88节点)
DT = {1: "f32", 2: "u8", 3: "i8", 5: "i16", 6: "i32",
      7: "i64", 9: "bool", 10: "f16", 11: "f64"}

# ---------------- modelslim 输出 ----------------
m1 = onnx.load(MS)
inits1 = {i.name: i for i in m1.graph.initializer}

# AscendQuant: 输出 tensor -> x_scale (attr scale 存的是倒数 1/x_scale)
aq_scale = {}
for n in m1.graph.node:
    if n.op_type == "AscendQuant":
        a = {x.name: x for x in n.attribute}
        aq_scale[n.output[0]] = 1.0 / a["scale"].f

# Conv: input[0] 来自 AscendQuant -> 激活 x_scale; 输出 tensor -> 名字
xs_ms, conv_out = {}, {}
for n in m1.graph.node:
    if n.op_type == "Conv":
        if n.input[0] in aq_scale:
            xs_ms[n.name] = aq_scale[n.input[0]]
        conv_out[n.output[0]] = n.name

# AscendDequant: input[0]=conv 输出, input[1] uint64 init 低32位
# = float32(w_scale[i] * x_scale) —— 每通道乘积
prod_ms = {}
for n in m1.graph.node:
    if n.op_type == "AscendDequant" and n.input[0] in conv_out:
        arr = np.asarray(numpy_helper.to_array(inits1[n.input[1]])).ravel()
        f32 = (np.asarray(arr, dtype=np.uint64).astype(np.uint32)
               .view(np.float32).astype(np.float64))
        prod_ms[conv_out[n.input[0]]] = f32

# Conv 权重 dtype(量化后应为 int8)
wdt = {}
for n in m1.graph.node:
    if n.op_type == "Conv" and n.input[1] in inits1:
        wdt[n.name] = DT.get(inits1[n.input[1]].data_type, inits1[n.input[1]].data_type)

n_aq = sum(1 for n in m1.graph.node if n.op_type == "AscendQuant")
n_ad = sum(1 for n in m1.graph.node if n.op_type == "AscendDequant")
n_conv = sum(1 for n in m1.graph.node if n.op_type == "Conv")
print(f"[ms] AscendQuant={n_aq}  AscendDequant={n_ad}  Conv={n_conv}")
print(f"[ms] 挂上 AscendQuant 的 Conv={len(xs_ms)}  带 dequant 乘积的={len(prod_ms)}")
print(f"[ms] Conv 权重 dtype: {dict(Counter(wdt.values()))}")

# 图结构: AscendDequant 输出被谁消费(应链到下一个 AscendQuant 或 float 算子)
prod_out = {}
for n in m1.graph.node:
    if n.op_type == "AscendDequant" and n.input[0] in conv_out:
        prod_out[n.output[0]] = conv_out[n.input[0]]
consumers = {}
for n in m1.graph.node:
    for i in n.input:
        if i in prod_out:
            consumers.setdefault(i, []).append(n.op_type)
cons_stat = Counter(tuple(v) for v in consumers.values())
print(f"[ms] AscendDequant 输出的下游算子组合: {dict(cons_stat)}")

# ---------------- CPU 仿真 ----------------
m2 = onnx.load(SIM)
inits2 = {i.name: numpy_helper.to_array(i) for i in m2.graph.initializer}
xs_sim, ws_sim = {}, {}
for n in m2.graph.node:
    if n.op_type == "QLinearConv":
        xs_sim[n.name] = float(np.asarray(inits2[n.input[1]]).ravel()[0])
        ws_sim[n.name] = np.asarray(inits2[n.input[4]]).ravel().astype(np.float64)

common = sorted(set(xs_ms) & set(xs_sim))
only_ms = sorted(set(xs_ms) - set(xs_sim))
only_sim = sorted(set(xs_sim) - set(xs_ms))
print(f"[sim] QLinearConv={len(xs_sim)}; 与 ms 名字匹配 {len(common)}"
      f"  (仅ms {len(only_ms)}: {only_ms[:3]}  仅sim {len(only_sim)}: {only_sim[:3]})")

# ---------------- 对比 1: 激活 x_scale ----------------
rows = []
for name in common:
    a, b = xs_ms[name], xs_sim[name]
    rows.append((abs(np.log(a / b)), name, a, b))
rows.sort(reverse=True)
n_bad = sum(1 for r in rows if abs(np.exp(r[0]) - 1) > 0.01)
print(f"\n[1] 激活 x_scale: {n_bad}/{len(rows)} 偏差>1%")
for d, name, a, b in rows[:5]:
    print(f"    {name}\n      ms={a:.6g}  sim={b:.6g}  ratio={a / b:.6f}")

# ---------------- 对比 2: dequant 乘积 x_scale*w_scale[i] ----------------
rows2 = []
for name in common:
    pa = prod_ms.get(name)
    pb = xs_sim[name] * ws_sim.get(name, np.zeros(0))
    if pa is None or pa.shape != pb.shape:
        continue
    d = np.abs(np.log(pa / np.where(pb == 0, np.nan, pb)))
    rows2.append((float(np.nanmax(d)) if np.isnan(d).all() else
                  float(np.max(d)), name,
                  len(np.flatnonzero(np.exp(d) - 1 > 0.01))))
rows2.sort(reverse=True)
n_bad2 = sum(1 for r in rows2 if r[0] > 0.01)
print(f"\n[2] dequant 乘积(每通道): {n_bad2}/{len(rows2)} 个 conv 偏差>1%")
for d, name, k in rows2[:5]:
    print(f"    {name}: max|log(p_ms/p_sim)|={d:.4f}  (>1% 通道数 {k})")

# ---------------- 对比 3: 反推 w_scale 检验 ----------------
# ms: prod[i]/x_scale_ms[i] vs sim: w_scale[i]  (x_scale 一致才有意义)
worst_w = []
for name in common:
    pa, pb = prod_ms.get(name), ws_sim.get(name)
    if pa is None or pb is None or pa.shape != pb.shape:
        continue
    wa = pa / xs_ms[name]
    dw = np.abs(np.log(wa / pb))
    worst_w.append((float(dw.max()), name))
worst_w.sort(reverse=True)
print(f"\n[3] 反推 w_scale: 最差 max|log(w_ms/w_sim)| top5")
for d, name in worst_w[:5]:
    print(f"    {name}: {d:.4f}")

print("\n判定: [1][2] 一致(偏差<1%) => 转换忠实, B 在 ATC/OM; 否则转换 bug")
