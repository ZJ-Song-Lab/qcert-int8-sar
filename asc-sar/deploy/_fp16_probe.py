# -*- coding: utf-8 -*-
"""本地 fp16 注入实验: 在 CPU QDQ 仿真图上模拟 OM 的 fp16 中间算子。

OM dtype 审计(results/om_info/int8.json)结论:
- MaxPoolV3(SPPF x3) 输入 DT_FLOAT16  <- model.9/cv2 逆转点直接上游
- BatchMatMulV2(attention x2) DT_FLOAT16 <- model.10 恶化区
- 7 个量化豁免 Conv 跑 fp16(model.1, model.2/4/16 的 m.0/cv2 等)

方法: 对目标算子做 Cast(fp32->fp16) + Cast(fp16->fp32) 环绕, 模拟 fp16
表示损失(MaxPool 的 max 是选择操作, Cast 环绕忠实; MatMul 只模拟输入
表示损失, 不含 fp16 累加)。

产物(在 CPU 上推, 配合 compare_layers.py --sim <变体> 比对):
  models/yolo11n_sim_mp16.onnx        仅 MaxPool fp16
  models/yolo11n_sim_mp16_mm16.onnx   MaxPool + MatMul fp16
  models/yolo11n_sim_all16.onnx       MaxPool + MatMul + 豁免 Conv fp16
"""
import sys
from pathlib import Path

import onnx
from onnx import helper

HERE = Path(__file__).resolve().parent
MODELS = (HERE / ".." / "models").resolve()
SRC = MODELS / "yolo11n_qdq_sim_debug.onnx"


def cast_wrap(model, nodes):
    """对 nodes 里的每个节点做 Cast fp16/fp32 环绕(改名输出, 尾部 Cast 还名)。"""
    graph = model.graph
    node_by_out = {}
    for n in graph.node:
        for o in n.output:
            node_by_out[o] = n

    targets = {id(n) for n in nodes}
    in_casts_of, out_casts_of = {}, {}
    for node in nodes:
        # 输入侧: 只对 fp32 数据输入环绕(跳过权重/常量/量化解耦点)
        in_casts, new_inputs = [], []
        for inp in node.input:
            prod = node_by_out.get(inp)
            is_const = prod is None  # 无生产者=initializer
            if is_const or prod.op_type in ("DequantizeLinear",):
                new_inputs.append(inp)
                continue
            h = f"{inp}__fp16in"
            in_casts.append(make_cast(h, inp, h + "_c", onnx.TensorProto.FLOAT16))
            new_inputs.append(h + "_c")
        # 输出侧: 全部改名, 尾部 Cast 还原名(多消费者安全)
        out_casts = []
        old_outs = list(node.output)
        renamed = [o + "__raw" for o in old_outs]
        del node.output[:]
        node.output.extend(renamed)
        del node.input[:]
        node.input.extend(new_inputs)
        for old, new in zip(old_outs, renamed):
            out_casts.append(
                make_cast(old + "__fp16out", new, old, onnx.TensorProto.FLOAT))
        in_casts_of[id(node)] = in_casts
        out_casts_of[id(node)] = out_casts

    # 重建节点序列(protobuf RepeatedField 不支持切片赋值)
    new_seq = []
    for n in graph.node:
        if id(n) in targets:
            new_seq.extend(in_casts_of[id(n)])
            new_seq.append(n)
            new_seq.extend(out_casts_of[id(n)])
        else:
            new_seq.append(n)
    del graph.node[:]
    graph.node.extend(new_seq)
    return model


def make_cast(name, src, dst, to):
    return helper.make_node("Cast", [src], [dst], name=name, to=to)


def main():
    base = onnx.load(str(SRC))
    convs = [n for n in base.graph.node if n.op_type == "QLinearConv"]
    q_outs = {n.output[0] for n in base.graph.node
              if n.op_type == "QuantizeLinear"}
    plain = [c for c in convs if c.input[0] not in q_outs]  # x 非 Q 输出 = 未量化
    maxpools = [n for n in base.graph.node if n.op_type == "MaxPool"]
    matmuls = [n for n in base.graph.node if n.op_type == "MatMul"]
    print(f"[graph] conv={len(convs)} 量化豁免={len(plain)} "
          f"MaxPool={len(maxpools)} MatMul={len(matmuls)}")
    for c in plain:
        print(f"  豁免 conv: {c.name}")

    variants = [
        ("yolo11n_sim_mp16.onnx", maxpools),
        ("yolo11n_sim_mp16_mm16.onnx", maxpools + matmuls),
        ("yolo11n_sim_all16.onnx", maxpools + matmuls + plain),
    ]
    for name, nodes in variants:
        m = onnx.load(str(SRC))
        cast_wrap(m, [n for n in m.graph.node
                      if n.name in {x.name for x in nodes}])
        # Cast 环绕不改 graph.output, 但要清 shape 推断缓存
        onnx.checker.check_model(m) if False else None
        out = MODELS / name
        onnx.save(m, str(out))
        print(f"[saved] {out} (wrapped {len(nodes)} nodes)")


if __name__ == "__main__":
    main()
