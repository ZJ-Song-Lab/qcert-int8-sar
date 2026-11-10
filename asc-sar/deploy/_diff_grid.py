# -*- coding: utf-8 -*-
"""验证 fp16 Cast 注入是否真实生效 + OM diff 网格分析。

1) 生效性验证: 原 sim vs mp16/mm16 在同一输入下的输出差——若 bit-exact 则注入失败。
2) 网格分析: B 问题层的 (om - sim)/step 分布——
   - 集中在整数格点 → 两边都在 int8 网格上, 只是格点选择不同(requant/round 差异);
   - 连续分布 → OM 输出不在 int8 网格上(fp16 直出/累加损失)。
"""
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from postprocess import preprocess  # noqa: E402

IMG = "datasets/ssdd/images/test/000001.jpg"


def outs(model_path, x):
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    return sess.run(None, {sess.get_inputs()[0].name: x})


def main():
    x, _, _ = preprocess(IMG, 640)

    # ---- 1) 注入生效性验证 ----
    print("=" * 60)
    print("[1] fp16 Cast 注入生效性验证")
    base = outs("models/yolo11n_qdq_sim_debug.onnx", x)
    for tag, path in [("mp16", "models/yolo11n_sim_mp16.onnx"),
                      ("mp16_mm16", "models/yolo11n_sim_mp16_mm16.onnx"),
                      ("all16", "models/yolo11n_sim_all16.onnx")]:
        v = outs(path, x)
        # 找 3 个代表性输出 index: 开头/中部/检测头
        idxs = [0, len(v) // 2, len(v) - 1]
        diffs = [float(np.abs(v[i].astype(np.float64) - base[i].astype(np.float64)).max())
                 for i in idxs]
        exact = all(d == 0.0 for d in diffs)
        print(f"  {tag:<10} max|Δ| @ [首/中/尾] = {diffs}  "
              f"{'BIT-EXACT(注入失效!)' if exact else '有差异(注入生效)'}")

    # ---- 2) OM diff 网格分析 ----
    print("=" * 60)
    print("[2] OM vs sim 的 (diff/step) 网格分析")
    with open("results/debug_layers.json", encoding="utf-8") as f:
        maps = json.load(f)
    asc = {e["conv"]: e for e in maps["ascend"]}
    qdq = {e["conv"]: e for e in maps["qdq"]}

    # 从 sim ONNX 提取每层 DQ 的 scale(输出网格 step)
    m = onnx.load("models/yolo11n_qdq_sim_debug.onnx")
    init = {t.name: t for t in m.graph.initializer}
    dq_scale = {}  # qdq conv 名 -> scale
    node_by_out = {o: n for n in m.graph.node for o in n.output}
    for c, e in qdq.items():
        # qdq 侧该 conv 对应的 DQ 节点: 找以该层 DQ 输出结尾的链
        pass
    # 更直接: debug ONNX 的输出 index e['index'] 的生产链尾是 DequantizeLinear
    out_names = [o.name for o in m.graph.output]
    for c, e in qdq.items():
        oname = out_names[e["index"]]
        prod = node_by_out.get(oname)
        if prod is None:
            continue
        if prod.op_type == "DequantizeLinear":
            s_name = prod.input[1]
            t = init.get(s_name)
            if t is not None:
                dq_scale[c] = float(onnx.numpy_helper.to_array(t).ravel()[0])

    targets = ["/model.9/cv2/conv/Conv_quant", "/model.10/cv1/conv/Conv_quant",
               "/model.10/m/m.0/ffn/ffn.1/conv/Conv_quant",
               "/model.10/cv2/conv/Conv_quant", "/model.23/cv3.2/cv3.2.1/cv3.2.1.1/conv/Conv_quant",
               "/model.0/conv/Conv_quant"]  # 最后一行放个健康层对照
    sim_outs = outs("models/yolo11n_qdq_sim_debug.onnx", x)
    print(f"{'层':<46} {'step':>10} {'|d|/step 分位数 p50/p90/p99/max':>32}  整格占比")
    for conv in targets:
        e_a, e_q = asc.get(conv), qdq.get(conv)
        if not e_a or not e_q:
            print(f"[warn] 映射缺失 {conv}")
            continue
        om = np.load(Path("results/layers_int8") / f"000001__out{e_a['index']}.npy")
        sim = sim_outs[e_q["index"]]
        if om.shape != sim.shape:
            print(f"[warn] {conv} shape 不一致")
            continue
        step = dq_scale.get(conv)
        d = (om.astype(np.float64) - sim.astype(np.float64)).ravel()
        if step is None or step <= 0:
            print(f"[warn] {conv} 无 scale, 跳过网格分析")
            continue
        ds = np.abs(d) / step
        frac_int = float(np.mean(np.abs(ds - np.round(ds)) < 0.05))
        q = np.percentile(ds, [50, 90, 99, 100])
        print(f"{conv:<46} {step:>10.3e} "
              f"p50={q[0]:6.2f} p90={q[1]:6.2f} p99={q[2]:7.2f} max={q[3]:8.2f}  "
              f"{frac_int:5.1%}")
        # 直方图(粗): diff/step 的整格点分布
        hist, edges = np.histogram(np.clip(ds, 0, 10), bins=[0, 0.05, 0.5, 1.05, 1.5,
                                                              2.05, 3.05, 5.05, 10.1])
        labels = ["~0", "0-0.5", "~1", "1-1.5", "~2", "2-3", "3-5", "5-10"]
        print("    " + "  ".join(f"{l}:{h / len(ds):5.2%}" for l, h in zip(labels, hist)))


if __name__ == "__main__":
    main()
