# -*- coding: utf-8 -*-
"""枚举 Neck 三层 SiLU 输出的全部 AscendQuant 消费者 + 验证生产链。

AMCT 重命名后 SiLU = Mul(dequant_out, Sigmoid(dequant_out))，
act 张量 = `/model.X/cv2/act/Mul_output_0`（非 fp ONNX 的
`/model.X/act/Mul_output_0`）。
"""
import onnx

MODEL = "/root/asc_sar/models/yolo11n_ssdd_int8.onnx"

# (tag, 已 dump 的 conv dequant 输出)
SOURCES = (
    ("p3", "/model.16/cv2/conv/Conv_quant_dequant_output"),
    ("p4", "/model.19/cv2/conv/Conv_quant_dequant_output"),
    ("p5", "/model.22/cv2/conv/Conv_quant_dequant_output"),
)


def main():
    m = onnx.load(MODEL)
    consumers, producers = {}, {}
    for n in m.graph.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)
        for o in n.output:
            producers[o] = n

    for tag, src in SOURCES:
        print(f"=== {tag}  src={src} ===")
        # 找 SiLU 的 Mul 节点（同时消费 src 和 Sigmoid(src)）
        mul = None
        for c in consumers.get(src, []):
            if c.op_type == "Mul":
                mul = c
                break
        if mul is None:
            print("  [!!] 找不到 SiLU Mul")
            continue
        act = mul.output[0]
        print(f"  SiLU: [{mul.op_type}] {mul.name}  inputs={list(mul.input)}")
        print(f"  act tensor = {act}")
        # act 的全部消费者
        for c in consumers.get(act, []):
            if c.op_type == "AscendQuant":
                attrs = {a.name: onnx.helper.get_attribute_value(a)
                         for a in c.attribute}
                # quant 输出的消费者（确认是哪个 conv）
                downs = [f"{cc.op_type}:{cc.name}"
                         for cc in consumers.get(c.output[0], [])]
                print(f"    [QUANT] {c.name}  scale={attrs.get('scale')} "
                      f"offset={attrs.get('offset')}  -> {downs}")
            else:
                print(f"    [{c.op_type}] {c.name}")


if __name__ == "__main__":
    main()
