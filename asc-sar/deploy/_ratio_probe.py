# -*- coding: utf-8 -*-
"""B 问题数值归因: 嫌疑层 om vs sim 逐元素统计。

检验三个假设:
  H1 requant scale 错误 => om/sim 比值近似常数 alpha != 1
  H2 f16 饱和/下溢     => |om| 出现 65504 钳位或小值桶误差爆炸
  H3 f16 舍入传播      => rel_l2(om, f16(sim)) << rel_l2(om, sim)
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
    "/model.1/conv/Conv_quant",                       # 健康对照
    "/model.9/cv1/conv/Conv_quant",                   # 健康
    "/model.9/cv2/conv/Conv_quant",                   # 萌芽
    "/model.10/cv1/conv/Conv_quant",                  # 爆炸起点
    "/model.10/m/m.0/attn/qkv/conv/Conv_quant",
    "/model.10/m/m.0/attn/pe/conv/Conv_quant",
    "/model.10/m/m.0/attn/proj/conv/Conv_quant",
    "/model.10/m/m.0/ffn/ffn.0/conv/Conv_quant",
    "/model.10/m/m.0/ffn/ffn.1/conv/Conv_quant",      # 最大偏差
    "/model.10/cv2/conv/Conv_quant",
]


def pearson(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    a -= a.mean()
    b -= b.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def rel_l2(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def main():
    with open("results/debug_layers.json", encoding="utf-8") as f:
        maps = json.load(f)
    asc = {e["conv"]: e["index"] for e in maps["ascend"]}
    qdq = {e["conv"]: e["index"] for e in maps["qdq"]}

    x, _, _ = preprocess("datasets/ssdd/images/test/000001.jpg", 640)

    # sim: QOperator 形态 debug ONNX, 88 个 DQ 输出(fp 值)
    sess = ort.InferenceSession("models/yolo11n_qdq_sim_debug.onnx",
                                providers=["CPUExecutionProvider"])
    sim_outs = sess.run(None, {sess.get_inputs()[0].name: x})

    print(f"{'层':<40} {'om~sim':>8} {'比值p50':>8} {'p10':>7} {'p90':>7} "
          f"{'f16sim~om':>9} {'om_max':>10} {'sim_max':>9} {'饱和数':>6} {'corr':>7}")
    for conv in LAYERS:
        om = np.load(f"results/layers_int8/000001__out{asc[conv]}.npy").astype(np.float32)
        sim = sim_outs[qdq[conv]].astype(np.float32)
        if om.shape != sim.shape:
            print(f"[skip] {conv}: shape {om.shape} vs {sim.shape}")
            continue

        n_inf = int(np.sum(~np.isfinite(om)))
        om_f = om.copy()
        om_f[~np.isfinite(om_f)] = 0.0  # 统计用

        # H1: 比值分布(|sim| 取显著值, 避开 0 附近噪声)
        sig = np.abs(sim) > np.quantile(np.abs(sim), 0.5)
        r = (om_f[sig] / sim[sig]).astype(np.float64)
        r = r[np.isfinite(r)]
        r_med = float(np.median(r)) if r.size else float("nan")
        r_p10 = float(np.quantile(r, 0.10)) if r.size else float("nan")
        r_p90 = float(np.quantile(r, 0.90)) if r.size else float("nan")

        # H3: om vs f16(sim)
        f16sim = sim.astype(np.float16).astype(np.float32)
        d_f16 = rel_l2(om_f, f16sim)

        # H2: 饱和元素数(|om| > 60000 即 f16 接近上限)
        n_sat = int(np.sum(np.abs(om_f) > 60000))

        print(f"{conv:<40} {rel_l2(om_f, sim):8.4f} {r_med:8.3f} {r_p10:7.3f} {r_p90:7.3f} "
              f"{d_f16:9.4f} {np.abs(om_f).max():10.2f} {np.abs(sim).max():9.2f} "
              f"{n_sat:6d} {pearson(om_f, sim):7.4f}{'  INF!' if n_inf else ''}")

        # 附加: 按 |sim| 幅值分 4 桶看误差分布(f16 下溢特征 = 小桶误差大)
        qs = np.quantile(np.abs(sim), [0.25, 0.5, 0.75])
        masks = [
            np.abs(sim) <= qs[0],
            (np.abs(sim) > qs[0]) & (np.abs(sim) <= qs[1]),
            (np.abs(sim) > qs[1]) & (np.abs(sim) <= qs[2]),
            np.abs(sim) > qs[2],
        ]
        buckets = [rel_l2(om_f[m], sim[m]) if m.any() else float("nan") for m in masks]
        print(f"{'':>40} 分桶 om~sim(|sim| 小→大): "
              + " ".join(f"{b:7.4f}" for b in buckets))


if __name__ == "__main__":
    main()
