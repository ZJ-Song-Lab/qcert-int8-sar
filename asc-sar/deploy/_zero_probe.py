# -*- coding: utf-8 -*-
"""B 问题归因第三探: 三方逐位置归因(fp32 真值仲裁 sim 的零 vs om 的非零)。

核心问题: sim 的精确零位置上 fp32 是多少?
  - fp32~0 且 om 非0 => om 无中生有(OM 破坏)
  - fp32 非0 且 sim=0 => sim 吃零(QOperator int8 输出量化), om 更准
  分桶 rel(om,fp32) vs rel(sim,fp32) 看谁在哪些幅值段更准。
"""
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from postprocess import preprocess  # noqa: E402

LAYERS = [
    "/model.1/conv/Conv_quant",
    "/model.9/cv1/conv/Conv_quant",
    "/model.9/cv2/conv/Conv_quant",
    "/model.10/cv1/conv/Conv_quant",
    "/model.10/m/m.0/attn/qkv/conv/Conv_quant",
    "/model.10/m/m.0/attn/pe/conv/Conv_quant",
    "/model.10/m/m.0/attn/proj/conv/Conv_quant",
    "/model.10/m/m.0/ffn/ffn.0/conv/Conv_quant",
    "/model.10/m/m.0/ffn/ffn.1/conv/Conv_quant",
    "/model.10/cv2/conv/Conv_quant",
]


def rel(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def corr(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    if a.size == 0:
        return float("nan")
    a -= a.mean() if a.size else 0
    b -= b.mean() if b.size else 0
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def main():
    with open("results/debug_layers.json", encoding="utf-8") as f:
        maps = json.load(f)
    asc = {e["conv"]: e["index"] for e in maps["ascend"]}
    qdq = {e["conv"]: e["index"] for e in maps["qdq"]}

    # fp32 真值(与 compare_layers 相同: 暴露 88 个 conv 输出)
    m = onnx.load("yolo11n.onnx")
    convs = [n for n in m.graph.node if n.op_type == "Conv"]
    existing = {o.name for o in m.graph.output}
    names, idx = [], len(m.graph.output)
    for n in convs:
        if n.output[0] in existing:
            continue
        m.graph.output.append(
            helper.make_tensor_value_info(n.output[0], TensorProto.FLOAT, None))
        names.append((n.name, idx))
        idx += 1
    fp32_sess = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])

    x, _, _ = preprocess("datasets/ssdd/images/test/000001.jpg", 640)
    fp32_outs = fp32_sess.run(None, {fp32_sess.get_inputs()[0].name: x})
    fp32 = {name: fp32_outs[i] for name, i in names}

    sim_sess = ort.InferenceSession("models/yolo11n_qdq_sim_debug.onnx",
                                    providers=["CPUExecutionProvider"])
    sim_outs = sim_sess.run(None, {sim_sess.get_inputs()[0].name: x})

    print(f"{'层':<40} {'om~fp32':>8} {'sim~fp32':>8} {'sim零%':>6} {'om零%':>5} "
          f"{'|fp32|@sim零':>11} {'|om|@sim零':>10} {'rel(om,f32)@零':>14} {'rel(sim,f32)@非零':>16}")
    for conv in LAYERS:
        om = np.load(f"results/layers_int8/000001__out{asc[conv]}.npy").astype(np.float64)
        sim = sim_outs[qdq[conv]].astype(np.float64)
        ref = fp32[conv.replace("_quant", "")].astype(np.float64)
        if not (om.shape == sim.shape == ref.shape):
            print(f"[skip] {conv}")
            continue

        zs = sim.ravel() == 0.0
        zo = om.ravel() == 0.0
        n = sim.size
        ref_f, om_f, sim_f = ref.ravel(), om.ravel(), sim.ravel()

        # sim 零位置上的三方统计
        if zs.any():
            r_at_z = rel(om_f[zs], ref_f[zs])
            s_at_nz = rel(sim_f[~zs], ref_f[~zs])
            mean_ref = float(np.mean(np.abs(ref_f[zs])))
            mean_om = float(np.mean(np.abs(om_f[zs])))
        else:
            r_at_z = s_at_nz = float("nan")
            mean_ref = mean_om = float("nan")

        print(f"{conv:<40} {rel(om, ref):8.4f} {rel(sim, ref):8.4f} "
              f"{zs.mean():6.1%} {zo.mean():5.1%} {mean_ref:11.4g} {mean_om:10.4g} "
              f"{r_at_z:14.4f} {s_at_nz:16.4f}")

    # ffn.1 深挖: sim 零位置 fp32 vs om 的相关性 + 样本
    conv = "/model.10/m/m.0/ffn/ffn.1/conv/Conv_quant"
    om = np.load(f"results/layers_int8/000001__out{asc[conv]}.npy").astype(np.float64).ravel()
    sim = sim_outs[qdq[conv]].astype(np.float64).ravel()
    ref = fp32[conv.replace("_quant", "")].astype(np.float64).ravel()
    zs = sim == 0.0
    print(f"\n[ffn.1] sim 零元素 {zs.sum()}/{sim.size} ({zs.mean():.1%}); "
          f"om 零 {int((om == 0).sum())}; fp32 零 {int((ref == 0).sum())}")
    print(f"[ffn.1] @sim零: mean|fp32|={np.abs(ref[zs]).mean():.4g} "
          f"mean|om|={np.abs(om[zs]).mean():.4g} corr(om,fp32)={corr(om[zs], ref[zs]):.4f}")
    print(f"[ffn.1] @sim零 rel(om,fp32)={rel(om[zs], ref[zs]):.4f}; "
          f"@sim非零 rel(om,fp32)={rel(om[~zs], ref[~zs]):.4f} rel(sim,fp32)={rel(sim[~zs], ref[~zs]):.4f}")
    idx = np.where(zs)[0][:16]
    print("[ffn.1] 样本 (fp32, sim, om) @sim零:")
    for j in idx[:16]:
        print(f"    ({ref[j]:+.4g}, {sim[j]:+.4g}, {om[j]:+.4g})")
    idx2 = np.where(~zs)[0][:8]
    print("[ffn.1] 样本 (fp32, sim, om) @sim非零:")
    for j in idx2:
        print(f"    ({ref[j]:+.4g}, {sim[j]:+.4g}, {om[j]:+.4g})")


if __name__ == "__main__":
    main()
