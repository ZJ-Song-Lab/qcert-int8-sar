# -*- coding: utf-8 -*-
"""modelslim scale 表 → E5 spec json 转换器（E5 设置 1 用）。

输入 PTQ 产出的 <model>_scale.json，定位三个输出头 cls 分支末端 conv
（/model.*/cv3.{l}/cv3.{l}.2/Conv_quant_dequant），按 cell 数加权几何平均
（w=6400/1600/400，即 640 输入下 80²/40²/20²）算 delta_s，产出
run_e5.py --old/--new 可读的 spec json（只需 delta_s / delta_sigma 两键，
其余为审计信息）。

公式与 e5_int8.json note 一致；m0 基准自检：
    python scale_to_spec.py --scale results/m0/yolo11n_ssdd_int8_scale.json \
        --out specs/_m0_selfcheck.json
    # 产出 delta_s 应 == 0.0007762214456100264（m0 基准）
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
# 输出头 cls 分支末端 conv：cv3.{l}/cv3.{l}.2（nc=1 时单条目无 [i] 后缀）
HEAD_RE = re.compile(
    r"^/model\.\d+/cv3\.([012])/cv3\.\1\.2/Conv_quant_dequant(?:\[\d+\])?$")
# 640 输入下三输出头 cell 数 80²/40²/20²
W = {0: 6400, 1: 1600, 2: 400}
# fp16 判决层 σ 网格（判决层不在 PTQ 范围内，与 e5_int8.json 一致）
DELTA_SIGMA = 1.0 / 128
M0_REF = 0.0007762214456100264


def head_deltas(scale_path):
    """返回 {l: 代表 delta}（每头若多通道则通道几何平均）。"""
    with open(scale_path, encoding="utf-8") as f:
        ents = json.load(f)["entries"]
    per = defaultdict(list)
    for e in ents:
        m = HEAD_RE.match(e.get("name", ""))
        if m:
            per[int(m.group(1))].append(float(e["delta"]))
    missing = [l for l in (0, 1, 2) if not per[l]]
    if missing:
        raise SystemExit(f"[scale_to_spec] 缺输出头 cv3.{{{missing}}}: {scale_path}")
    return {l: float(np.exp(np.mean(np.log(v)))) for l, v in per.items()}, \
        {l: len(v) for l, v in per.items()}


def delta_s_of(heads):
    w = np.array([W[l] for l in (0, 1, 2)], dtype=np.float64)
    d = np.array([heads[l] for l in (0, 1, 2)], dtype=np.float64)
    return float(np.exp(np.sum(w * np.log(d)) / w.sum()))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scale", required=True, help="modelslim <out>_scale.json")
    ap.add_argument("--out", required=True, help="产出 spec json 路径")
    ap.add_argument("--tag", default="", help="备注标签（写入 note）")
    args = ap.parse_args()

    heads, nch = head_deltas(args.scale)
    ds = delta_s_of(heads)
    spec = {
        "delta_s": ds,
        "delta_sigma": DELTA_SIGMA,
        "shift_bits": 16,
        "domain": "raw logit = logit(prob)（OM 输出 prob 反变换）",
        "heads": {f"cv3.{l}.2": {"delta": heads[l], "cells": W[l],
                                  "n_channels": nch[l]} for l in (0, 1, 2)},
        "note": (f"由 {Path(args.scale).name} 生成（scale_to_spec.py）：三输出头 "
                 f"Conv_quant_dequant 的 delta 按 cell 数加权几何平均 "
                 f"(w=6400/1600/400)。dSigma 同 fp16 判决层 1/128。{args.tag}"),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8")
    rel = abs(ds - M0_REF) / M0_REF if out.name.startswith("_m0") else None
    print(f"[scale_to_spec] {Path(args.scale).name}")
    for l in (0, 1, 2):
        print(f"  cv3.{l}.2 delta={heads[l]:.6g} (w={W[l]}, ch={nch[l]})")
    print(f"  delta_s={ds:.16g} -> {out}")
    if rel is not None:
        print(f"  [m0 自检] vs 基准 {M0_REF}: rel={rel:.3e} "
              f"({'OK' if rel < 1e-9 else 'MISMATCH'})")


if __name__ == "__main__":
    main()
