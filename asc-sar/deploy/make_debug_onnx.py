# -*- coding: utf-8 -*-
"""板上一次性脚本: 构造中间层审计用 debug ONNX。

把 modelslim INT8 ONNX(AscendQuant 形态)的每个 AscendDequant 输出、
CPU QDQ 仿真(QOperator 形态)的每个 QLinearConv 输出, 追加为 graph.output,
使 ATC 转出的 OM 与 ORT 推理都能拿到每层反量化后的 fp 值, 逐层比对。

输出(以 --tag 区分, 默认 COCO 版保持旧文件名兼容):
  yolo11n[_<tag>]_int8_debug.onnx      (Ascend 形态, 88 个 DQ 输出 + 原输出)
  yolo11n[_<tag>]_qdq_sim_debug.onnx   (QDQ 形态, --sim 给定时)
  debug_layers[_<tag>].json            (输出 index -> conv 名 映射, 比对用)

E2 真实化(SSDD 版, 只需 Ascend 形态):
  python make_debug_onnx.py --ms yolo11n_ssdd_int8.onnx --tag ssdd
"""
import argparse
import json

import onnx
from onnx import TensorProto, helper

MODELS = "/root/asc_sar/models"


def conv_producer(model):
    """tensor 名 -> 生产它的 Conv 节点名。"""
    return {n.output[0]: n.name for n in model.graph.node if n.op_type == "Conv"}


def expose(model, tensors_desc, tag):
    """把 [(tensor 名, conv 名, 生产节点)] 追加为 graph.output, 返回映射。"""
    existing = {o.name for o in model.graph.output}
    mapping, idx = [], len(model.graph.output)
    for tensor, conv, op in tensors_desc:
        if tensor in existing:
            continue
        model.graph.output.append(
            helper.make_tensor_value_info(tensor, TensorProto.FLOAT, None))
        mapping.append({"index": idx, "tensor": tensor, "conv": conv, "op": op})
        existing.add(tensor)
        idx += 1
    print(f"[{tag}] 追加 {len(mapping)} 个输出, 总输出 {idx}")
    return mapping


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ms", default="yolo11n_int8.onnx",
                    help="modelslim INT8 ONNX (AscendQuant 形态)")
    ap.add_argument("--sim", default=None,
                    help="CPU QOperator 仿真 ONNX (不给则跳过 QDQ 部分)")
    ap.add_argument("--tag", default=None, help="文件名后缀 (如 ssdd)")
    args = ap.parse_args()

    sfx = f"_{args.tag}" if args.tag else ""
    stem = args.ms.replace(".onnx", "_debug.onnx")
    out_layers = f"debug_layers{sfx}.json"

    # ---- Ascend 形态: AscendDequant 输出(fp 值) ----
    m1 = onnx.load(f"{MODELS}/{args.ms}")
    producers = conv_producer(m1)
    dq_list = []
    for n in m1.graph.node:
        if n.op_type == "AscendDequant":
            conv = producers.get(n.input[0], n.input[0])
            dq_list.append((n.output[0], conv, "AscendDequant"))
    # 按图内出现顺序稳定排序(graph.output 顺序 = OM 输出 index)
    map1 = expose(m1, dq_list, "ascend")
    onnx.save(m1, f"{MODELS}/{stem}")
    print(f"[ok] {stem}")

    # ---- QDQ 形态: QLinearConv 对应 DequantizeLinear 的输出(fp 值) ----
    # 注意: QOperator 形态里 QLinearConv.output[0] 是 int8 码, 不能直接比;
    # 须取其后 DequantizeLinear 的输出才是与 AscendDequant 同语义的 fp 值。
    map2 = []
    if args.sim:
        m2 = onnx.load(f"{MODELS}/{args.sim}")
        dq_of = {}
        for n in m2.graph.node:
            if n.op_type == "DequantizeLinear":
                dq_of[n.input[0]] = n.output[0]
        qc_list = [(dq_of.get(n.output[0], n.output[0]), n.name,
                    "QLinearConv+DequantizeLinear")
                   for n in m2.graph.node if n.op_type == "QLinearConv"]
        assert len(dq_of) == 88, f"DequantizeLinear 应为 88 个, 实际 {len(dq_of)}"
        map2 = expose(m2, qc_list, "qdq")
        sim_out = args.sim.replace(".onnx", "_debug.onnx")
        onnx.save(m2, f"{MODELS}/{sim_out}")
        print(f"[ok] {sim_out}")

    with open(f"{MODELS}/{out_layers}", "w", encoding="utf-8") as f:
        json.dump({"ascend": map1, "qdq": map2}, f, ensure_ascii=False, indent=2)
    print(f"[ok] {out_layers}: ascend={len(map1)} qdq={len(map2)}")


if __name__ == "__main__":
    main()
