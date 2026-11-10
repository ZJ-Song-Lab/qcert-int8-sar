# -*- coding: utf-8 -*-
"""B 问题归因第二探: 误差的通道/空间分布 + 小值元素配对。

若误差集中在特定通道 => per-channel scale 错位(weight 量化 per-channel);
若集中在特定空间区 => NC1HWC0/fractal 布局错位;
若均匀分布且小值元素被吃 => 量化域截断/offset 语义问题。
"""
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from postprocess import preprocess  # noqa: E402

LAYERS = [
    "/model.10/cv1/conv/Conv_quant",
    "/model.10/m/m.0/attn/qkv/conv/Conv_quant",
    "/model.10/m/m.0/attn/pe/conv/Conv_quant",
    "/model.10/m/m.0/attn/proj/conv/Conv_quant",
    "/model.10/m/m.0/ffn/ffn.0/conv/Conv_quant",
    "/model.10/m/m.0/ffn/ffn.1/conv/Conv_quant",
    "/model.10/cv2/conv/Conv_quant",
]


def main():
    with open("results/debug_layers.json", encoding="utf-8") as f:
        maps = json.load(f)
    asc = {e["conv"]: e["index"] for e in maps["ascend"]}
    qdq = {e["conv"]: e["index"] for e in maps["qdq"]}

    x, _, _ = preprocess("datasets/ssdd/images/test/000001.jpg", 640)
    sess = ort.InferenceSession("models/yolo11n_qdq_sim_debug.onnx",
                                providers=["CPUExecutionProvider"])
    sim_outs = sess.run(None, {sess.get_inputs()[0].name: x})

    for conv in LAYERS:
        om = np.load(f"results/layers_int8/000001__out{asc[conv]}.npy").astype(np.float64)
        sim = sim_outs[qdq[conv]].astype(np.float64)
        if om.shape != sim.shape:
            print(f"[skip] {conv}")
            continue
        print(f"\n===== {conv}  shape={om.shape}")

        # ---- 通道分布 ----
        if om.ndim == 4:
            B, C, H, W = om.shape
            ec = np.linalg.norm((om - sim).reshape(C, -1), axis=1)
            sc = np.linalg.norm(sim.reshape(C, -1), axis=1) + 1e-12
            relc = ec / sc
            order = np.argsort(-relc)
            print(f"  per-channel rel: median={np.median(relc):.3f} "
                  f"p90={np.quantile(relc, 0.9):.3f} max={relc[order[0]]:.3f}")
            print(f"  top6 坏通道: " + " ".join(
                f"ch{c}={relc[c]:.2f}" for c in order[:6]))
            # 通道编号 mod 16 分组(检验 C0=16 fractal 对齐)
            g = relc.reshape(-1, 16).mean(axis=0)
            print(f"  通道号 mod16 分组均值: "
                  + " ".join(f"{i}:{v:.2f}" for i, v in enumerate(g)))

            # ---- 空间分布 ----
            es = np.linalg.norm((om - sim).transpose(0, 2, 3, 1).reshape(H * W, -1), axis=1)
            ss = np.linalg.norm(sim.transpose(0, 2, 3, 1).reshape(H * W, -1), axis=1) + 1e-12
            rels = (es / ss).reshape(H, W)
            print(f"  空间 rel map ({H}x{W}, 行=H):")
            for row in rels:
                print("    " + " ".join(f"{v:5.2f}" for v in row))

            # ---- 小值元素配对(|sim| 最小的 512 个) ----
            flat_sim, flat_om = sim.ravel(), om.ravel()
            idx = np.argsort(np.abs(flat_sim))[:512]
            pair = np.stack([flat_sim[idx], flat_om[idx]])
            n_zero_om = int(np.sum(flat_om[idx] == 0.0))
            n_sign_flip = int(np.sum(flat_sim[idx] * flat_om[idx] < 0))
            print(f"  最小 512 元素: om 精确零={n_zero_om}, 符号翻转={n_sign_flip}")
            print("  配对样本 (sim, om):")
            for i in range(0, 24, 4):
                print("    " + "  ".join(
                    f"({flat_sim[j]:+.3g},{flat_om[j]:+.3g})" for j in idx[i:i + 4]))
        break  # 先只看第一层, 逐层细看


if __name__ == "__main__":
    main()
