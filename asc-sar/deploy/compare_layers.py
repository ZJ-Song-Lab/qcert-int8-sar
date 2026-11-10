# -*- coding: utf-8 -*-
"""B 问题三角逐层比对: 板上 INT8 OM vs CPU QDQ 仿真 vs fp32 源模型。

三方数据(同一张输入图, 同一 letterbox 预处理):
- fp32 真值: yolo11n.onnx 的 88 个 Conv 输出(in-memory 追加为 graph.output, ORT 推)
- sim:       yolo11n_qdq_sim_debug.onnx 的 88 个 DQ 输出(ORT 推)
- OM:        results/layers_int8/000001__out{i}.npy(板上 acl_infer.py --dump-layers)

映射: fp32 conv 名 + '_quant' == ms/sim 的 conv 名(debug_layers.json 按 conv 名 join)。

判定:
- sim~fp32 = CPU 仿真量化损伤(A 问题同源, MinMax 已知偏大);
- OM~fp32  = 板上量化损伤(B 问题主体);
- OM~sim   = 两条量化路径的差异;
- 若 OM~fp32 与 sim~fp32 在前 k 层同量级、在检测头附近分化 => 看 head 层;
- 若 OM~fp32 从某层起显著大于 sim~fp32 => 该层是 OM 侧额外损伤源(退化层嫌疑)。
"""
import argparse
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
    d = a.astype(np.float64) - b
    return float(np.linalg.norm(d) / (np.linalg.norm(b.astype(np.float64)) + 1e-12))


def run_fp32(fp32_path, x):
    """加载 fp32 源模型, in-memory 把 88 个 Conv 输出追加为 graph.output 并推理。"""
    m = onnx.load(fp32_path)
    convs = [n for n in m.graph.node if n.op_type == "Conv"]
    existing = {o.name for o in m.graph.output}
    names = []  # [(conv 名, 追加的输出 index)]
    idx = len(m.graph.output)
    for n in convs:
        if n.output[0] in existing:
            continue
        m.graph.output.append(
            helper.make_tensor_value_info(n.output[0], TensorProto.FLOAT, None))
        names.append((n.name, idx))
        idx += 1
    sess = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
    outs = sess.run(None, {sess.get_inputs()[0].name: x})
    fp32 = {f"{name}_quant": outs[i] for name, i in names}
    print(f"[fp32] 暴露 {len(names)} 个 conv 输出, ORT 推理完成")
    return fp32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="datasets/ssdd/images/test/000001.jpg")
    ap.add_argument("--fp32", default="yolo11n.onnx")
    ap.add_argument("--sim", default="models/yolo11n_qdq_sim_debug.onnx")
    ap.add_argument("--om-dir", default="results/layers_int8")
    ap.add_argument("--stem", default="000001")
    ap.add_argument("--mapping", default="results/debug_layers.json")
    args = ap.parse_args()

    with open(args.mapping, encoding="utf-8") as f:
        maps = json.load(f)
    asc = {e["conv"]: e for e in maps["ascend"]}
    qdq = {e["conv"]: e for e in maps["qdq"]}
    order = [e["conv"] for e in maps["ascend"]]
    print(f"[map] ascend={len(asc)} qdq={len(qdq)}")

    # ---- 同一预处理, 三方推理 ----
    x, _, _ = preprocess(args.image, 640)
    fp32 = run_fp32(args.fp32, x)
    sess = ort.InferenceSession(str(args.sim), providers=["CPUExecutionProvider"])
    sim_outs = sess.run(None, {sess.get_inputs()[0].name: x})

    # ---- 三角比对 ----
    rows = []
    for conv in order:
        e_a, e_q = asc[conv], qdq.get(conv)
        ref = fp32.get(conv)
        if e_q is None or ref is None:
            print(f"[warn] 映射缺失 {conv}: qdq={e_q is not None} fp32={ref is not None}")
            continue
        om = np.load(Path(args.om_dir) / f"{args.stem}__out{e_a['index']}.npy")
        sim = sim_outs[e_q["index"]]
        if om.shape != sim.shape or om.shape != ref.shape:
            print(f"[warn] {conv}: shape 不一致 om={om.shape} sim={sim.shape} fp32={ref.shape}")
            continue
        rows.append({"conv": conv,
                     "om_sim": rel_l2(om, sim), "om_fp32": rel_l2(om, ref),
                     "sim_fp32": rel_l2(sim, ref), "corr_om": pearson(om, ref)})

    print(f"\n{'层(拓扑序)':<44} {'om~fp32':>9} {'sim~fp32':>9} {'om~sim':>9} {'corr(om,fp32)':>13}")
    for r in rows:
        flag = ""
        if r["om_fp32"] > 0.15:
            flag = " <<<"
        print(f"{r['conv']:<44} {r['om_fp32']:9.4%} {r['sim_fp32']:9.4%} "
              f"{r['om_sim']:9.4%} {r['corr_om']:13.4f}{flag}")

    def med(k):
        return float(np.median([r[k] for r in rows]))

    print(f"\n[stat] median: om~fp32={med('om_fp32'):.4%} sim~fp32={med('sim_fp32'):.4%} "
          f"om~sim={med('om_sim'):.4%}")
    print(f"[stat] OM 侧额外损伤(om~fp32 - sim~fp32) top10:")
    for r in sorted(rows, key=lambda r: -(r["om_fp32"] - r["sim_fp32"]))[:10]:
        print(f"  {r['conv']:<44} om~fp32={r['om_fp32']:8.4%} sim~fp32={r['sim_fp32']:8.4%} "
              f"Δ={r['om_fp32'] - r['sim_fp32']:+8.4%}")


if __name__ == "__main__":
    main()
