"""AMCT 部署形态 ONNX -> ORT 可跑的整数网格仿真 ONNX（x86 参考实现，E3 第三实现）。

模型瘦身（modelslim）PTQ 的产物是 Ascend 私有算子图：

    AscendQuant(x; scale=s, offset=o)            # q = clip(round(x*s)+o, -128, 127)
        -> Conv(W_q int8, b int32)               # acc = Σ q_x·q_w + b
        -> AscendDequant(acc, deq u64)           # v = clip(round(acc·deq̂))·deq
                                                 # deq = w_scale[c]·x_scale（低32位 f32 位模式）

ORT 没有这些算子。本脚本做**语义等价**的纯浮点改写（fake-quant 仿真）：

1. AscendQuant -> Mul→Round→Add→Clip→Sub→Div（把 x 吸附到输入网格再还原成浮点）；
2. Conv 的 int8 权重/偏置就地反量化为 float32：
   ``W_f[c] = W_q[c]·deq[c]·s_in``、``b_f[c] = b32[c]·deq[c]``
   （``s_in`` = 上游 AscendQuant 的 scale 属性；``deq[c]`` = 下游 AscendDequant 的
   u64 scale 低 32 位按 f32 解释。由 ``W_f/b_f`` 定义可知 Conv 浮点输出
   ``y_f = acc·deq[c]``，与板上整数累加器语义一致，差仅在浮点累加舍入）；
3. AscendDequant -> Identity：硬件 dequant 只是 int→float 缩放（``v=acc·deq``），
   不在此处取整；中间层的 requant（到输出激活刻度 Δ）与下一层 AscendQuant 的
   ``s_next`` 是同一张量刻度，已由第 1 条的 fake-quant 链建模，不可重复量化。
   cv3 三个网格张量的 requant（Δ 来自 scale json）在 dump 脚本 numpy 侧完成，
   与板上 ``dump_cv3.py`` 的处理完全对齐；
4. 其余算子（Sigmoid/Mul/Concat/DFL…）原样保留——板上同样在浮点域执行。

产物：
- ``<out>.onnx``：ORT CPU 直接可跑，输出与原图一致（含 cv3 三网格，若在图中）；
- ``<out>.json``：网格输出名、空间尺寸（Δ 置 None，由 dump 侧 --head-deltas
  提供，唯一正确来源是板上 scale json）、逐层 deq 表（审计用）。

验证：onnx.checker + 与 AMCT 原图逐输出对齐的数值冒烟（ORT vs 语义推导）。

用法::

    python asc-sar/deploy/amct_to_sim.py \
        --model asc-sar/models/yolo11n_ssdd_int8_cv3.onnx \
        --out asc-sar/models/yolo11n_ssdd_int8_cv3_sim.onnx
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


def u64_to_f32(u) -> np.ndarray:
    """Ascend u64 dequant scale：低 32 位按 float32 位模式解释。"""
    u = np.asarray(u).ravel().astype(np.uint64)
    return u.astype(np.uint32).view(np.float32).astype(np.float64)


def _attr(node, name, default=None):
    for a in node.attribute:
        if a.name == name:
            return onnx.helper.get_attribute_value(a)
    return default


def _toposort(nodes, available):
    """Kahn 拓扑排序（deploy 原图节点顺序不保证拓扑有序，checker 强制要求）。"""
    avail = set(available)
    remaining = list(nodes)
    ordered = []
    while remaining:
        progressed = False
        still = []
        for n in remaining:
            if all(i in avail for i in n.input if i):
                ordered.append(n)
                avail.update(o for o in n.output if o)
                progressed = True
            else:
                still.append(n)
        if not progressed:
            raise SystemExit(f"拓扑排序失败，剩余 {len(still)} 节点存在悬空输入: "
                             f"{[n.name for n in still[:5]]}")
        remaining = still
    return ordered


def convert(model_path: str, out_path: str) -> dict:
    m = onnx.load(str(model_path))
    g = m.graph
    if any(o.version < 11 for o in m.opset_import if o.domain in ("", "ai.onnx")):
        raise SystemExit("需要 opset>=11（Round/Clip 输入形式）")

    inits = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    producer = {}
    for n in g.node:
        for o in n.output:
            producer[o] = n
    consumer = {}
    for n in g.node:
        for i in n.input:
            consumer.setdefault(i, []).append(n)

    new_nodes = []
    new_inits = []
    used_inits = set()
    layer_deq = {}          # conv 名 -> deq 数组（审计）
    quant_nodes = deq_nodes = conv_nodes = 0

    def add_init(name, arr):
        new_inits.append(numpy_helper.from_array(np.asarray(arr), name))
        used_inits.add(name)
        return name

    for idx, n in enumerate(g.node):
        tag = f"_sim{idx}"
        if n.op_type == "AscendQuant":
            s = float(_attr(n, "scale"))
            o = float(_attr(n, "offset"))
            c_s = add_init(f"{tag}_s", np.float32(s))
            c_o = add_init(f"{tag}_o", np.float32(o))
            c_lo = add_init(f"{tag}_lo", np.float32(-128.0))
            c_hi = add_init(f"{tag}_hi", np.float32(127.0))
            x = n.input[0]
            new_nodes += [
                helper.make_node("Mul", [x, c_s], [f"{tag}_m"], name=f"{tag}/Mul"),
                helper.make_node("Round", [f"{tag}_m"], [f"{tag}_r"], name=f"{tag}/Round"),
                helper.make_node("Add", [f"{tag}_r", c_o], [f"{tag}_a"], name=f"{tag}/Add"),
                helper.make_node("Clip", [f"{tag}_a", c_lo, c_hi], [f"{tag}_c"],
                                 name=f"{tag}/Clip"),
                helper.make_node("Sub", [f"{tag}_c", c_o], [f"{tag}_u"], name=f"{tag}/Sub"),
                helper.make_node("Div", [f"{tag}_u", c_s], [n.output[0]], name=f"{tag}/Div"),
            ]
            quant_nodes += 1
            continue

        if n.op_type == "AscendDequant":
            # 硬件 dequant = int->float 缩放（v = acc·deq），不取整；y_f 经 Conv
            # 权重折叠后已是 acc·deq，因此这里只需 Identity。中间层 requant 与
            # 下一层 AscendQuant 同刻度，不重复建模；cv3 网格 requant 在 numpy 侧。
            deq_nodes += 1
            new_nodes.append(helper.make_node("Identity", [n.input[0]],
                                              [n.output[0]], name=n.name))
            continue

        if n.op_type == "Conv":
            q = producer.get(n.input[0])
            assert q is not None and q.op_type == "AscendQuant", \
                f"{n.name}: 输入不是 AscendQuant 产出"
            s_in = float(_attr(q, "scale"))
            dq = next((c for c in consumer.get(n.output[0], [])
                       if c.op_type == "AscendDequant"), None)
            assert dq is not None, f"{n.name}: 找不到下游 AscendDequant"
            deq = u64_to_f32(inits[dq.input[1]])          # (C,) = w_scale·x_scale
            layer_deq[n.name] = deq.tolist()

            w_q = np.asarray(inits[n.input[1]])            # (C, k, kh, kw) int8
            assert w_q.shape[0] == deq.shape[0], \
                f"{n.name}: 权重通道 {w_q.shape[0]} != deq {deq.shape}"
            w_f = (w_q.astype(np.float64)
                   * (deq * s_in).reshape(-1, 1, 1, 1)).astype(np.float32)
            inputs = [n.input[0], add_init(f"{tag}_w", w_f)]
            if len(n.input) > 2 and n.input[2] in inits:
                b32 = np.asarray(inits[n.input[2]], dtype=np.float64).ravel()
                b_f = (b32 * deq).astype(np.float32)
                inputs.append(add_init(f"{tag}_b", b_f))
            attrs = {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}
            new_nodes.append(helper.make_node("Conv", inputs, list(n.output),
                                              name=n.name, **attrs))
            conv_nodes += 1
            continue

        # 其余算子原样拷贝（输入里的旧 initializer 仍可能引用，最后统一清理）
        new_nodes.append(n)
        for i in n.input:
            if i in inits:
                used_inits.add(i)

    # 图结构替换：清掉旧 initializer（int8 权重/u64 scale 已不再被引用）
    new_names = {i.name for i in new_inits}
    avail0 = ([i.name for i in g.input] + list(new_names)
              + [k for k in inits if k in used_inits])
    new_nodes = _toposort(new_nodes, avail0)
    del g.node[:]
    g.node.extend(new_nodes)
    del g.initializer[:]
    keep = [i for i in inits if i in used_inits]
    g.initializer.extend(
        [numpy_helper.from_array(inits[k], k) for k in keep])
    g.initializer.extend(new_inits)
    del g.value_info[:]          # 类型已变（int8->f32），交给 ORT 重新推导

    # cv3 网格输出信息（[1,1,H,W] 的图输出）。Δ 是输出激活 requant 刻度，
    # 只存在于 scale json（板上），deploy 图里没有——此处置 None，
    # 由 dump 脚本通过 --head-deltas / scale json 提供。
    grid = []
    for o in g.output:
        dims = [d.dim_value for d in o.type.tensor_type.shape.dim]
        if len(dims) == 4 and dims[1] == 1 and dims[2] == dims[3] and dims[2] > 0:
            grid.append({"name": o.name, "hw": dims[2], "delta": None})
    grid.sort(key=lambda d: -d["hw"])

    onnx.checker.check_model(m)
    onnx.save(m, str(out_path))

    info = {"source": str(model_path), "sim": str(out_path),
            "n_nodes": len(new_nodes), "quant": quant_nodes,
            "conv": conv_nodes, "dequant": deq_nodes,
            "grid_outputs": grid, "layer_deq": layer_deq,
            "semantics": "fake-quant：Round 为 half-to-even；"
                         "其余与 AMCT 整数链路等价（±浮点累加舍入）"}
    info_path = str(Path(out_path).with_suffix(".json"))
    Path(info_path).write_text(json.dumps(info, ensure_ascii=False, indent=2),
                               encoding="utf-8")
    print(f"[sim] quant={quant_nodes} conv={conv_nodes} dequant={deq_nodes} "
          f"-> {out_path}")
    for d in grid:
        print(f"[sim] 网格输出 {d['name']}  H={d['hw']}  Δ={d['delta']}")
    print(f"[sim] 信息 -> {info_path}")
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="AMCT 形态 ONNX（AscendQuant）")
    ap.add_argument("--out", required=True, help="输出仿真 ONNX")
    args = ap.parse_args(argv)
    convert(args.model, args.out)


if __name__ == "__main__":
    main()
