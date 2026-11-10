# -*- coding: utf-8 -*-
"""B 问题 scale 交叉验证: int8(Ascend) ONNX vs qdq_sim ONNX 逐 conv 量化参数。

ATC 的输入是板上 yolo11n_int8.onnx(AscendQuant 形态), OM 的量化参数全部来自它;
sim 是 yolo11n_qdq_sim.onnx(QOperator 形态), 同一 PTQ 的兄弟产物。
若两图上游量化参数(x_scale / w_scale / bias)不一致, om~sim 偏差直接可解释。

对齐关系(按 conv 名, 两图 Conv/QLinearConv 节点名一致, 均为 .../Conv_quant):
  int8 图: AscendQuant(scale=1/x_scale 属性, offset, quant_bit)
                -> Conv_quant(w int8, bias int32)
                -> AscendDequant(input[1]=u64, 低32位 f32 位模式 = w_scale[c]*x_scale)
  sim  图: QuantizeLinear / QLinearConv(x, x_scale, x_zp, w, w_scale[], w_zp,
                                       y_scale, y_zp, bias int32)

检验:
  C1  x_scale 一致性         sim.x_scale vs int8.x_scale
  C2  w_scale 一致性         sim.w_scale[c] vs int8.deq[c]/int8.x_scale
  C3  本地 ssdd_int8.onnx 是否就是板上 dump json 的那个文件
  C4  int32 bias 是否逐元素一致
"""
import json

import numpy as np
import onnx
from onnx import numpy_helper

INT8 = "models/yolo11n_ssdd_int8.onnx"        # 候选: 板上 yolo11n_int8.onnx 本地副本
SIM = "models/yolo11n_qdq_sim_debug.onnx"
SCALE_JSON = "results/yolo11n_int8_scale.json"

SUSPECT = [
    "/model.1/conv/Conv_quant",                       # 健康对照
    "/model.8/conv/Conv_quant",                       # 健康对照
    "/model.9/cv1/conv/Conv_quant",                    # 健康 om~fp32=5.7%
    "/model.9/cv2/conv/Conv_quant",                    # 萌芽 om~fp32=22.0%
    "/model.10/cv1/conv/Conv_quant",                   # 爆炸 om~fp32=44.8%
    "/model.10/m/m.0/attn/qkv/conv/Conv_quant",
    "/model.10/m/m.0/attn/proj/conv/Conv_quant",
    "/model.10/m/m.0/ffn/ffn.0/conv/Conv_quant",
    "/model.10/m/m.0/ffn/ffn.1/conv/Conv_quant",       # 峰值 om~fp32=77.3%
    "/model.10/cv2/conv/Conv_quant",
]


def u64_to_f32(u):
    """u64 低 32 位按 float32 位模式解释。"""
    u = np.asarray(u).ravel().astype(np.uint64)
    return u.astype(np.uint32).view(np.float32).astype(np.float64)


def parse_int8(path):
    m = onnx.load(path)
    inits = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
    producer = {}
    for n in m.graph.node:
        for o in n.output:
            producer[o] = n

    convs = {}
    for n in m.graph.node:
        if n.op_type != "Conv":
            continue
        info = {"x_scale": None, "zp": None, "bits": None, "chain": "",
                "deq": None, "deq_raw": None, "w": None, "bias": None}
        # 输入侧: 最多上溯 4 层找 AscendQuant
        chain, p = [], producer.get(n.input[0])
        for _ in range(4):
            if p is None:
                break
            chain.append(p.op_type)
            if p.op_type == "AscendQuant":
                attrs = {a.name: a for a in p.attribute}
                inv = attrs.get("scale")
                if inv is not None:
                    info["x_scale"] = 1.0 / inv.f
                off = attrs.get("offset")
                if off is not None:
                    info["zp"] = round(off.f)
                qb = attrs.get("quant_bit")
                if qb is not None:
                    info["bits"] = qb.i
                break
            p = producer.get(p.input[0]) if p.input else None
        info["chain"] = "->".join(chain)
        # 输出侧: AscendDequant 消费 Conv 输出
        for c in m.graph.node:
            if c.op_type == "AscendDequant" and c.input[0] == n.output[0]:
                u = inits.get(c.input[1])
                if u is not None:
                    info["deq_raw"] = np.asarray(u).ravel().astype(np.uint64)
                    info["deq"] = u64_to_f32(u)
                break
        info["w"] = inits.get(n.input[1])
        if len(n.input) > 2:
            info["bias"] = inits.get(n.input[2])
        convs[n.name] = info
    return convs


def parse_sim(path):
    m = onnx.load(path)
    inits = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
    convs = {}
    for n in m.graph.node:
        if n.op_type != "QLinearConv":
            continue
        # inputs: x, x_scale, x_zp, w, w_scale, w_zp, y_scale, y_zp, bias?
        def scal(i):
            a = inits.get(n.input[i])
            return None if a is None else float(np.asarray(a).ravel()[0])
        convs[n.name] = {
            "x_scale": scal(1), "x_zp": scal(2),
            "w_scale": (np.asarray(inits[n.input[4]], dtype=np.float64).ravel()
                        if n.input[4] in inits else None),
            "y_scale": scal(6),
            "bias": inits.get(n.input[8]) if len(n.input) > 8 else None,
        }
    return convs


def compare(conv, a, s, board):
    """返回 (行, 明细行)。"""
    rx = (s["x_scale"] / a["x_scale"]) if (a["x_scale"] and s["x_scale"]) else float("nan")
    ratio = None
    if a["deq"] is not None and s["w_scale"] is not None and a["x_scale"]:
        w_int8 = a["deq"] / a["x_scale"]
        nch = min(len(w_int8), len(s["w_scale"]))
        ratio = s["w_scale"][:nch] / w_int8[:nch]
        wr_p50 = float(np.median(ratio))
        w_maxrel = float(np.max(np.abs(s["w_scale"][:nch] - w_int8[:nch])
                                / (np.abs(w_int8[:nch]) + 1e-30)))
    else:
        nch, wr_p50, w_maxrel = 0, float("nan"), float("nan")

    # C3: 本地 deq vs 板上 json (同文件验证)
    c3 = []
    if a["deq"] is not None:
        for i in range(len(a["deq"])):
            key = f"{conv}_dequant[{i}]"
            if key in board:
                c3.append(abs(board[key] - a["deq"][i]) / max(abs(board[key]), 1e-30))
    c3 = max(c3) if c3 else float("nan")

    # C4: int32 bias 逐元素比较
    bmsg = "-"
    if a["bias"] is not None and s["bias"] is not None:
        ba, bs = np.asarray(a["bias"]).ravel(), np.asarray(s["bias"]).ravel()
        if ba.dtype == bs.dtype and ba.shape == bs.shape:
            bmsg = f"max|Δ|={int(np.max(np.abs(ba.astype(np.int64) - bs.astype(np.int64))))}"
        else:
            bmsg = f"dtype {ba.dtype}/{bs.dtype} shape {ba.shape}/{bs.shape}"

    line = (f"{conv:<42} {rx:9.6f} {wr_p50:10.6f} {w_maxrel:11.3e} "
            f"{c3:10.3e} {nch:4d} {bmsg}")
    detail = (f"{'':>42} 输入链={a['chain']} zp int8/sim={a['zp']}/{s['x_zp']} "
              f"bits={a['bits']} sim.y_scale={s['y_scale']:.6g}")
    if ratio is not None:
        detail += (f" | w比值 min/max={ratio.min():.6f}/{ratio.max():.6f}")
    if a["deq_raw"] is not None and "cv2/conv" in conv and "model.9" in conv:
        hi = (a["deq_raw"][:4] >> 32).astype(np.uint32)
        detail += f" | u64高32位={[hex(int(v)) for v in hi]}"
    return line, detail


def main():
    a8 = parse_int8(INT8)
    sim = parse_sim(SIM)
    print(f"[int8] Conv {len(a8)} 个; [sim] QLinearConv {len(sim)} 个")
    ka, ks = set(a8), set(sim)
    print(f"同名 {len(ka & ks)}; 仅int8 {len(ka - ks)}; 仅sim {len(ks - ka)}")
    for k in sorted(ka - ks)[:5]:
        print("  仅int8:", k)
    for k in sorted(ks - ka)[:5]:
        print("  仅sim:", k)

    with open(SCALE_JSON, encoding="utf-8") as f:
        board = {e["name"]: e["delta"] for e in json.load(f)["entries"]}

    print(f"\n{'conv':<42} {'x_s比值':>9} {'w比值p50':>10} {'w最大rel差':>11} "
          f"{'C3本地vs板上':>10} {'通道':>4} bias(int32)")
    for conv in SUSPECT:
        a, s = a8.get(conv), sim.get(conv)
        if a is None or s is None:
            print(f"[skip] {conv}: int8={'有' if a else '无'} sim={'有' if s else '无'}")
            continue
        line, detail = compare(conv, a, s, board)
        print(line)
        print(detail)

    # ---- 全量扫描: 找所有 conv 中 x/w 不一致最大的前 10 ----
    rows = []
    for k in ka & ks:
        a, s = a8[k], sim[k]
        rx = (s["x_scale"] / a["x_scale"]) if (a["x_scale"] and s["x_scale"]) else float("nan")
        if a["deq"] is not None and s["w_scale"] is not None and a["x_scale"]:
            w_int8 = a["deq"] / a["x_scale"]
            nch = min(len(w_int8), len(s["w_scale"]))
            w_maxrel = float(np.max(np.abs(s["w_scale"][:nch] - w_int8[:nch])
                                    / (np.abs(w_int8[:nch]) + 1e-30)))
        else:
            w_maxrel = float("nan")
        rows.append((k, rx, w_maxrel))
    bad_x = [r for r in rows if np.isfinite(r[1]) and abs(r[1] - 1) > 1e-4]
    bad_w = sorted([r for r in rows if np.isfinite(r[2]) and r[2] > 1e-4],
                   key=lambda r: -r[2])[:10]
    print(f"\n[全量 {len(rows)} conv] |x_s比值-1|>1e-4 的有 {len(bad_x)} 个")
    for k, rx, _ in bad_x[:10]:
        print(f"  {k:<44} x_s比值={rx:.8f}")
    print(f"[全量] w 最大 rel 差 >1e-4 的有 "
          f"{sum(1 for r in rows if np.isfinite(r[2]) and r[2] > 1e-4)} 个, 前10:")
    for k, _, w in bad_w:
        print(f"  {k:<44} w_maxrel={w:.3e}")


if __name__ == "__main__":
    main()
