# -*- coding: utf-8 -*-
"""板上探针: yolo11n_int8.onnx 的 AscendQuant 原始 attrs 审计。

背景: sim(QDQ 形态) 的激活量化是非对称(x_zp 多为 -127, uint8-as-int8),
而 scale json dump 出的 AQ zero_point 全是 0(对称)。x_scale 又逐层一致。
怀疑: modelslim convert_graph_lf_quant 在 QDQ->AscendQuant 转换时丢弃了
zp(属性根本没写 offset), 或 scale_table.py 读漏了(属性名不对)。
本脚本直接打印 AQ 节点的全部原始属性, 一锤定音。
板上运行: python3 _aq_attr_probe.py
"""
from collections import Counter

import onnx

MODEL = "/root/asc_sar/models/yolo11n_int8.onnx"


def main():
    m = onnx.load(MODEL)
    aqs = [n for n in m.graph.node if n.op_type == "AscendQuant"]
    print(f"AQ total: {len(aqs)}")

    print("\n=== 前 8 个 AQ 的全部原始 attrs ===")
    for n in aqs[:8]:
        print(f"{n.name}")
        for a in n.attribute:
            print(f"    {a.name} = {onnx.helper.get_attribute_value(a)!r}")

    offs = {}
    scales = []
    for n in aqs:
        d = {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}
        if "offset" in d:
            offs[n.name] = d["offset"]
        scales.append(d.get("scale"))

    print(f"\n带 offset 属性的 AQ: {len(offs)}/{len(aqs)}")
    if offs:
        print("offset 值分布(前 10):", Counter(offs.values()).most_common(10))

    # scale 属性是 1/x_scale, 顺便验几个
    print(f"scale 属性示例: {scales[:3]}")

    # 顺带看 AscendDequant 的 attrs(是否有相关 offset)
    ads = [n for n in m.graph.node if n.op_type == "AscendDequant"]
    print(f"\nAD total: {len(ads)}")
    if ads:
        print(f"AD[0] attrs: "
              f"{[(a.name, onnx.helper.get_attribute_value(a)) for a in ads[0].attribute]}")
        print(f"AD[0] inputs: {list(ads[0].input)}")


if __name__ == "__main__":
    main()
