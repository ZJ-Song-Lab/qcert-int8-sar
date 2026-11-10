# -*- coding: utf-8 -*-
"""扫描 SSDD 全集（test+train）Neck 三层 SiLU 激活分布，定位自然高激活图。

背景（E2 压测集重设计，2026-11-04）：
  probe_stress_tf.py 证明 11 种图像域变换均无法推高 Neck 激活（超限 0/12），
  激活峰值由强目标/大尺度结构驱动。AMCT 标定上限 P3/P4/P5 = 12.08/17.43/38.61
  来自校准集统计（calib256 从 train 分层抽样），故全集应存在激活接近上限的
  自然图。本脚本验证该假设并给出压测集候选（按激活分档）。

输出：
  asc-sar/results/act_scan_ssdd.csv   split,stem,p3,p4,p5 fp act_max
  终端：各 split 各层 med/P90/P99/max、超限计数 vs CAL_MAX、top25（按 P5）
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image

HERE = Path(__file__).resolve().parent
DEPLOY = HERE.parent.parent / "asc-sar" / "deploy"
sys.path.insert(0, str(DEPLOY))
from postprocess import preprocess  # noqa: E402  与板上同一预处理口径

FP_ONNX = DEPLOY.parent / "runs" / "yolo11n_ssdd" / "weights" / "best.onnx"
IMAGES = DEPLOY.parent / "datasets" / "ssdd" / "images"
OUT_CSV = DEPLOY.parent / "results" / "act_scan_ssdd.csv"
CAL_MAX = {"p3": 12.08, "p4": 17.43, "p5": 38.61}   # AMCT 标定上限(SiLU 输出)
NECK_NODES = {"p3": 16, "p4": 19, "p5": 22}


def neck_silu_tensors(model):
    """Neck 三层 SiLU(Mul) 的输出张量名。"""
    pat = {t: re.compile(rf"^/model\.{n}/cv2/act/Mul$") for t, n in NECK_NODES.items()}
    found = {}
    for nd in model.graph.node:
        if nd.op_type != "Mul":
            continue
        for t, p in pat.items():
            if nd.name and p.match(nd.name):
                found[t] = nd.output[0]
    missing = [t for t in NECK_NODES if t not in found]
    if missing:
        raise SystemExit(f"[err] 未找到 {missing} 的 SiLU 节点")
    return found


def main():
    m = onnx.load(str(FP_ONNX))
    targets = neck_silu_tensors(m)
    existing = {o.name for o in m.graph.output}
    for name in targets.values():
        if name not in existing:
            m.graph.output.append(onnx.helper.make_tensor_value_info(
                name, onnx.TensorProto.FLOAT, None))
    tmp = HERE / "_fp_neck_scan.onnx"
    onnx.save(m, str(tmp))

    sess = ort.InferenceSession(str(tmp), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]
    t_idx = {t: out_names.index(v) for t, v in targets.items()}

    rows = []
    for split in ("test", "train"):
        paths = sorted((IMAGES / split).glob("*.jpg"))
        print(f"[scan] {split}: {len(paths)} 张")
        for i, p in enumerate(paths):
            img = np.asarray(Image.open(p).convert("RGB"))
            x, _, _ = preprocess(img, 640)
            outs = sess.run(None, {in_name: x})
            rec = {t: float(np.asarray(outs[t_idx[t]], dtype=np.float64).max())
                   for t in NECK_NODES}
            rows.append({"split": split, "stem": p.stem, **rec})
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(paths)}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=["split", "stem", "p3", "p4", "p5"])
        w.writeheader()
        w.writerows(rows)
    print(f"[ok] {len(rows)} 行 -> {OUT_CSV}")

    for split in ("test", "train", "all"):
        sub = rows if split == "all" else [r for r in rows if r["split"] == split]
        print(f"\n===== {split}（n={len(sub)}）=====")
        for t in ("p3", "p4", "p5"):
            v = np.asarray([r[t] for r in sub])
            over = int((v > CAL_MAX[t]).sum())
            print(f"  {t}: med={np.median(v):6.2f} P90={np.quantile(v, .9):6.2f} "
                  f"P99={np.quantile(v, .99):6.2f} max={v.max():7.2f} "
                  f"超{CAL_MAX[t]}: {over}/{len(sub)}")

    top = sorted(rows, key=lambda r: r["p5"], reverse=True)[:25]
    print("\n===== top25 by P5 act_max =====")
    for r in top:
        flags = "".join(t for t in ("p3", "p4", "p5") if r[t] > CAL_MAX[t])
        print(f"  {r['split']:<5} {r['stem']}: p3={r['p3']:7.2f} p4={r['p4']:7.2f} "
              f"p5={r['p5']:7.2f} {'超限层:' + flags if flags else ''}")


if __name__ == "__main__":
    main()
