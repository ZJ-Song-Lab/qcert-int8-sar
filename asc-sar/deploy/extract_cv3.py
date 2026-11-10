"""从 QDQ ONNX 中抽出三个检测头 cls 终卷积（cv3.*.2）的量化输出作为图输出。

用途：饱和主线（E0 No-Go 后的预注册转向）——
OM 的最终输出经图内 float sigmoid 已不在整数网格上（grid_probe 实测），
最后一个**整数域张量**是 cv3 终卷积的输出。把它暴露成 OM 输出后，
板上可直接读到 ``q * Δ`` 的网格值，从而：

1. 经验性验证 A3（网格可读出、Δ 与 scale 表一致）；
2. E2 饱和审计：真实张量的 ±127 触顶率、并列数（ties）；
3. E5 阈值搬运：λ_logit = logit(λ_prob)，λ_code = round(λ_logit / Δ_h)。

用法::

    python asc-sar/deploy/extract_cv3.py \
        --src asc-sar/models/yolo11n_ssdd_int8.onnx \
        --dst asc-sar/models/yolo11n_ssdd_int8_cv3.onnx
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import onnx
from onnx import utils


def find_cv3_outputs(model) -> list[str]:
    """找 cv3.{0,1,2}.2 终卷积后面的 dequant（QDQ 里最后一个 DQ）输出名。

    AMCT 的 QDQ 导出把每个量化张量包成 ``..._quant_quant`` /
    ``..._quant_dequant`` 节点对；网格值是 **dequant 节点的输出**。
    """
    pat = re.compile(r"/model\.23/cv3\.(\d)/cv3\.\1\.2/Conv_quant_dequant$")
    outs = []
    for node in model.graph.node:
        if pat.fullmatch(node.name):
            if node.op_type not in ("DequantizeLinear", "Conv"):
                # AMCT 模拟节点可能是自定义 op_type，宽松处理：只按名字收
                pass
            outs.append((int(pat.fullmatch(node.name).group(1)),
                        node.output[0], node.name, node.op_type))
    if not outs:
        # 退一步：直接收 Conv 节点本身的输出（其后应紧跟 DQ）
        pat2 = re.compile(r"/model\.23/cv3\.(\d)/cv3\.\1\.2/Conv$")
        for node in model.graph.node:
            m = pat2.fullmatch(node.name)
            if m:
                outs.append((int(m.group(1)), node.output[0],
                             node.name, node.op_type))
    if len(outs) != 3:
        names = [n.name for n in model.graph.node if "cv3" in n.name]
        raise RuntimeError(f"cv3 终卷积找齐 3 个失败（{len(outs)} 个）。"
                           f"图里 cv3 相关节点：\n" + "\n".join(names))
    outs.sort()
    return [o[1] for o in outs]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    args = ap.parse_args()

    model = onnx.load(args.src)
    keep = [o.name for o in model.graph.output]
    extra = find_cv3_outputs(model)
    print(f"[extract] 原输出 {keep}")
    print(f"[extract] 追加 cv3 输出 {extra}")

    # 直接把张量挂成图输出（AMCT 自定义节点 shape inference 走不通，
    # Extractor 依赖 value_info 会失败；手挂 output 最稳）。
    # 必须给明确 shape——None 会让 ATC 陷入超长 shape 推导（实测卡死 40min+）。
    # cv3.{0,1,2}.2 是分类头终卷积：通道=nc=1，stride 8/16/32 @640 → 80/40/20。
    head_hw = {0: 80, 1: 40, 2: 20}
    pat = re.compile(r"/model\.23/cv3\.(\d)/")
    for name in extra:
        if name not in keep:
            m = pat.search(name)
            hw = head_hw[int(m.group(1))]
            model.graph.output.append(onnx.helper.make_tensor_value_info(
                name, onnx.TensorProto.FLOAT, [1, 1, hw, hw]))
    Path(args.dst).parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.dst)
    print(f"[extract] -> {args.dst}  输出 {keep + extra}")


if __name__ == "__main__":
    main()
