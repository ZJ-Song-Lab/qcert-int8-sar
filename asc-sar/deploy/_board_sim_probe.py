# -*- coding: utf-8 -*-
"""B 问题决定性检验(修正版): 板上 scale json vs 本地 qdq_sim_debug。

_scale_probe.py 的教训: 本地 yolo11n_ssdd_int8.onnx 是另一份校准的陈旧产物
(其 x_scale 与 sim、与板上 json 都对不上), 以它为基准的 C1/C2 检验作废。
板上 yolo11n_int8.onnx(ATC 的真实输入)的 scale 已 dump 进 json:
  AQ: {conv}_quant            delta = x_scale (attr scale 的倒数)
  AD: {conv}_dequant[{c}]     delta = w_scale[c] * x_scale (逐通道)
本脚本用它 vs sim 的 QLinearConv(input[1]=x_scale, input[4]=w_scale[]) 对比。

判定:
  [1] x_scale 一致 + [2] 乘积一致 → 板上 PTQ 转换忠实, B 在 ATC/OM 数值侧
  不一致(集中在 model.9/10)       → 校准/转换阶段发散, om~sim 偏差可解释
  [1] 一致但 zp 语义不同(sim 非对称 zp≠0 vs 板上对称 zp=0) → convert 阶段
      丢 zp, 量化语义不同, 需评估削波损失
"""
import json
from collections import Counter

import numpy as np
import onnx
from onnx import numpy_helper

SIM = "models/yolo11n_qdq_sim_debug.onnx"
SCALE_JSON = "results/yolo11n_int8_scale.json"

SUSPECT = [
    "/model.1/conv/Conv_quant",
    "/model.2/cv1/conv/Conv_quant",
    "/model.8/conv/Conv_quant",
    "/model.9/cv1/conv/Conv_quant",
    "/model.9/cv2/conv/Conv_quant",
    "/model.10/cv1/conv/Conv_quant",
    "/model.10/m/m.0/attn/qkv/conv/Conv_quant",
    "/model.10/m/m.0/attn/proj/conv/Conv_quant",
    "/model.10/m/m.0/ffn/ffn.0/conv/Conv_quant",
    "/model.10/m/m.0/ffn/ffn.1/conv/Conv_quant",
    "/model.10/cv2/conv/Conv_quant",
]


def main():
    with open(SCALE_JSON, encoding="utf-8") as f:
        d = json.load(f)
    board = {e["name"]: e for e in d["entries"]}
    n_aq = sum(1 for e in d["entries"] if e["source"] == "AscendQuant")
    n_ad = sum(1 for e in d["entries"] if e["source"] == "AscendDequant")
    print(f"[board json] 共 {len(board)} 项; AQ {n_aq} / AD {n_ad}")

    m = onnx.load(SIM)
    inits = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
    print("[sim] op 统计:", dict(Counter(n.op_type for n in m.graph.node)))
    sim = {}
    for n in m.graph.node:
        if n.op_type != "QLinearConv":
            continue
        sim[n.name] = {
            "x_scale": float(np.asarray(inits[n.input[1]]).ravel()[0]),
            "x_zp": float(np.asarray(inits[n.input[2]]).ravel()[0]),
            "w_scale": np.asarray(inits[n.input[4]]).ravel().astype(np.float64),
            "y_scale": float(np.asarray(inits[n.input[6]]).ravel()[0]),
        }

    # ---- [1] x_scale: board AQ delta vs sim ----
    rows = []
    for name, s in sim.items():
        e = board.get(f"{name}_quant")
        if e is None:
            continue
        rows.append((name, e["delta"], s["x_scale"], s["x_zp"], e["zero_point"]))
    n_bad = sum(1 for r in rows if abs(r[1] / r[2] - 1) > 0.01)
    print(f"\n[1] x_scale: 匹配 {len(rows)}/{len(sim)}; 偏差>1%: {n_bad}")
    for name, bx, sx, szp, bzp in sorted(rows, key=lambda r: -abs(np.log(r[1] / r[2])))[:8]:
        print(f"    {name}: board={bx:.6g} sim={sx:.6g} ratio={bx / sx:.6f} "
              f"zp b/s={bzp}/{szp:.0f}")
    zps = [r[3] for r in rows]
    print(f"    [info] sim x_zp 分布: min={min(zps):.0f} max={max(zps):.0f} "
          f"(|zp|>8 的占 {sum(1 for z in zps if abs(z) > 8)}/{len(zps)})")

    # ---- [2] dequant 乘积: board AD delta[c] vs sim x_scale*w_scale[c] ----
    rows2 = []
    for name, s in sim.items():
        bp, sp = [], []
        for c in range(len(s["w_scale"])):
            e = board.get(f"{name}_dequant[{c}]")
            if e is None:
                continue
            bp.append(e["delta"])
            sp.append(s["x_scale"] * s["w_scale"][c])
        if not bp:
            continue
        bp, sp = np.array(bp), np.array(sp)
        with np.errstate(divide="ignore", invalid="ignore"):
            lg = np.abs(np.log(bp / np.where(sp == 0, np.nan, sp)))
        mx = float(np.nanmax(lg)) if np.isnan(lg).all() is False else 0.0
        rows2.append((mx, name, int(np.sum(np.exp(lg[~np.isnan(lg)]) - 1 > 0.01)), len(bp)))
    rows2.sort(reverse=True)
    n_bad2 = sum(1 for r in rows2 if r[0] > 0.01)
    print(f"\n[2] dequant 乘积(每通道): 有乘积的 conv {len(rows2)}; 偏差>1%: {n_bad2}")
    for mx, name, k, nch in rows2[:8]:
        print(f"    {name}: max|log(p_b/p_s)|={mx:.4f} (>1% 通道 {k}/{nch})")

    # ---- [3] SUSPECT 明细: x_scale / zp / 反推 w_scale ----
    print("\n=== SUSPECT 明细 ===")
    for name in SUSPECT:
        s = sim.get(name)
        if s is None:
            print(f"  {name}: sim 无")
            continue
        e = board.get(f"{name}_quant")
        if e is None:
            print(f"  {name}: board 无专属 AQ 项(共享 AQ)")
            continue
        bx = e["delta"]
        keys = [f"{name}_dequant[{c}]" for c in range(len(s["w_scale"]))]
        bp = np.array([board[k]["delta"] for k in keys if k in board])
        line2 = ""
        if len(bp):
            nw = bp / bx
            sw = s["w_scale"][:len(bp)]
            wrel = float(np.max(np.abs(nw - sw) / (np.abs(sw) + 1e-30)))
            line2 = f"  w 反推 rel差 max={wrel:.3e} 通道 {len(bp)}/{len(s['w_scale'])}"
        print(f"  {name}")
        print(f"    x_scale b/s={bx:.6g}/{s['x_scale']:.6g} ratio={bx / s['x_scale']:.6f}"
              f"  zp b/s={e['zero_point']}/{s['x_zp']:.0f}  bits={e['bits']}{line2}")

    print("\n判定: [1][2] <1% → 板上 PTQ 忠实, B 在 ATC/OM; 否则校准/转换发散")


if __name__ == "__main__":
    main()
