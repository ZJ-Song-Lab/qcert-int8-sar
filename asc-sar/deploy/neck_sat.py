"""E2 Neck 层饱和摘要（板上）· 2026-11-04 修正版。

主变量口径（§5 修订）：Neck 三支路 SiLU 输出经 AscendQuant 的 int8 激活码
饱和率。AscendQuant 语义（由量化 ONNX 节点属性确认）::

    q = round(x * scale) + offset，超界 clamp 到 [-128, 127]

每个 Neck 层的 act 张量只有一个 AscendQuant（per-tensor 标定），其 int8
输出扇出到全部下游 conv（下采样路 + Detect head 的 cv2/cv3 分支）。

板上 dump 的是 conv 的 AscendDequant 输出 x（SiLU 之前）。SiLU =
x*sigmoid(x) 是确定性逐元素 fp 运算，且 x 正是板上 SiLU 的真实输入，
故::

    act = x * sigmoid(x)
    q_pre = round(act * scale) + offset
    sat = mean(q_pre > 127) + mean(q_pre < -128)    # clamp 比例

即板上真实激活码的饱和率，无需重编 debug OM。

（旧版用 scale.json 的 ``Conv_quant_dequant[N]`` delta 反推 q —— 那是
int32 累加器步长（delta_act*delta_w[c]），不是激活量化 scale，已废弃。）

用法（板上）::

    python3 neck_sat.py --layers results/layers_stress \
        --out results/neck_sat_stress.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# index:tag:scale:offset —— index 为 debug OM 输出序号（debug_layers_ssdd.json），
# scale/offset 为该层 act 张量 AscendQuant 节点属性（inspect_quant.py 实测）。
DEFAULT_NECK = ("48:p3:20.61783218383789:-122,"
                "61:p4:14.403969764709473:-124,"
                "79:p5:6.5524163246154785:-126")


def layer_stats(q_pre, act):
    """q_pre: 量化后未 clamp 的码；act: SiLU 输出 fp 值。"""
    up = float(np.mean(q_pre > 127))
    dn = float(np.mean(q_pre < -128))
    rail = float(np.mean((q_pre >= 127) | (q_pre <= -128)))
    return {"sat": up + dn, "sat_up": up, "sat_down": dn, "rail": rail,
            "qmin": int(q_pre.min()), "qmax": int(q_pre.max()),
            "act_max": float(act.max()), "act_p999": float(np.percentile(act, 99.9))}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layers", required=True, help="--dump-layers 输出目录")
    ap.add_argument("--out", required=True, help="输出 summary json")
    ap.add_argument("--neck", default=DEFAULT_NECK,
                    help="index:tag:scale:offset 逗号分隔")
    args = ap.parse_args()

    neck = {}
    for kv in args.neck.split(","):
        i, tag, sc, off = kv.split(":")
        neck[int(i)] = (tag, float(sc), float(off))

    root = Path(args.layers)
    manifest = json.load(open(root / "manifest.json"))
    by = {}
    for m in manifest:
        i = m["index"]
        if i not in neck:
            continue
        tag, scale, offset = neck[i]
        x = np.load(root / m["file"]).astype(np.float64).ravel()
        act = x * (1.0 / (1.0 + np.exp(-x)))  # SiLU: x * sigmoid(x)
        q_pre = np.rint(act * scale) + offset
        by.setdefault(m["stem"], {})[tag] = layer_stats(q_pre, act)

    cal = {tag: {"x_max": (127 - off) / sc, "x_min": (-128 - off) / sc}
           for tag, sc, off in neck.values()}
    json.dump({"neck": {str(k): v[0] for k, v in neck.items()},
               "quant": {v[0]: {"scale": v[1], "offset": v[2]} for v in neck.values()},
               "calibrated_range": cal,
               "stems": by},
              open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    print(f"[ok] {len(by)} stems -> {args.out}")
    for tag in [v[0] for v in neck.values()]:
        sats = [v[tag]["sat"] for v in by.values()]
        if sats:
            print(f"  [{tag}] n={len(sats)} sat mean={float(np.mean(sats)):.5f} "
                  f"max={float(np.max(sats)):.5f}")


if __name__ == "__main__":
    main()
