# -*- coding: utf-8 -*-
"""验证 copy-paste 强目标增强能否把 Neck 激活推过 AMCT 标定上限。

背景（probe_act_scan.py 结果）：
  自然图 P5 激活 med=11.7 / P90=26.6 / max=40.3，超 38.61 仅 4/1067——
  纯自然分档高档饱和率过低（~4%），撑不起 E2 的"饱和→并列数"叙事。
  激活峰值由强目标结构驱动（probe_stress_tf.py 排除全局对比度路线）。

假设：把超限图的强目标 patch（围绕 P5 激活峰值）贴到低激活接收图，
激活跟随 patch 保留 → 接收图激活被推到源图量级 → 可构造"强目标密度"
分档压测集（k=0/2/4 patch 三档，单调逼出饱和）。

源图（test 域 p5 top）：000211(39.38) / 000691(38.67) / 000299(38.60)
接收图：test 域 p5 最低 12 张（med 附近背景）
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
from postprocess import letterbox, preprocess  # noqa: E402

FP_ONNX = DEPLOY.parent / "runs" / "yolo11n_ssdd" / "weights" / "best.onnx"
IMAGES = DEPLOY.parent / "datasets" / "ssdd" / "images"
SCAN_CSV = DEPLOY.parent / "results" / "act_scan_ssdd.csv"
CAL_MAX = {"p3": 12.08, "p4": 17.43, "p5": 38.61}
NECK_NODES = {"p3": 16, "p4": 19, "p5": 22}
STRIDE = {"p3": 8, "p4": 16, "p5": 32}
PATCH = 64          # patch 边长（letterbox 后 640 域）
N_RECV = 12
SOURCES = ["000211", "000691", "000299"]
QUAD = [(160, 160), (480, 160), (160, 480), (480, 480)]   # 640 域四象限中心


def neck_silu_tensors(model):
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


def load_lb(stem, split="test"):
    """原图 -> letterbox 后 640x640 uint8 RGB（操作域）。"""
    img = np.asarray(Image.open(IMAGES / split / f"{stem}.jpg").convert("RGB"))
    lb, _, _ = letterbox(img, 640)
    return lb


def patch_at(lb, cx, cy, half=PATCH // 2):
    h, w = lb.shape[:2]
    y0, y1 = max(0, cy - half), min(h, cy + half)
    x0, x1 = max(0, cx - half), min(w, cx + half)
    return lb[y0:y1, x0:x1].copy()


def paste(lb, patch, cx, cy, half=PATCH // 2):
    out = lb.copy()
    h, w = out.shape[:2]
    y0, y1 = max(0, cy - half), min(h, cy + half)
    x0, x1 = max(0, cx - half), min(w, cx + half)
    ph, pw = patch.shape[:2]
    out[y0:y1, x0:x1] = patch[: y1 - y0, : x1 - x0]   # 边界裁剪
    return out


def act_max(sess, in_name, t_idx, img_lb):
    """letterbox 域 uint8 -> 各层 SiLU act_max 与 p5 激活图。"""
    x = np.ascontiguousarray(img_lb.transpose(2, 0, 1)[None],
                             dtype=np.float32) / 255.0
    outs = sess.run(None, {in_name: x})
    rec = {t: float(np.asarray(outs[t_idx[t]], dtype=np.float64).max())
           for t in NECK_NODES}
    return rec, np.asarray(outs[t_idx["p5"]], dtype=np.float64)[0]


def main():
    m = onnx.load(str(FP_ONNX))
    targets = neck_silu_tensors(m)
    existing = {o.name for o in m.graph.output}
    for name in targets.values():
        if name not in existing:
            m.graph.output.append(onnx.helper.make_tensor_value_info(
                name, onnx.TensorProto.FLOAT, None))
    tmp = HERE / "_fp_neck_probe.onnx"
    onnx.save(m, str(tmp))
    sess = ort.InferenceSession(str(tmp), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]
    t_idx = {t: out_names.index(v) for t, v in targets.items()}

    # ---- 源 patch：每张源图取 P5 峰值位置的 patch ----
    patches = []
    for stem in SOURCES:
        lb = load_lb(stem)
        _, amap = act_max(sess, in_name, t_idx, lb)
        c, y, x = np.unravel_index(int(amap.argmax()), amap.shape)
        cy, cx = y * STRIDE["p5"] + 16, x * STRIDE["p5"] + 16
        pat = patch_at(lb, cx, cy)
        print(f"[src] {stem}: 峰值 ch{c} @act(y={y},x={x}) -> 输入({cy},{cx}) "
              f"p5={amap.max():.2f} patch={pat.shape}")
        patches.append(pat)

    # ---- 接收图：test 域 p5 最低 N 张 ----
    rows = list(csv.DictReader(open(SCAN_CSV, encoding="utf-8")))
    recv = [r for r in rows if r["split"] == "test"]
    recv.sort(key=lambda r: float(r["p5"]))
    recv = recv[:N_RECV]
    recv_lb = {r["stem"]: load_lb(r["stem"]) for r in recv}
    print(f"[recv] {N_RECV} 张低激活接收图: "
          f"p5 [{float(recv[0]['p5']):.1f}..{float(recv[-1]['p5']):.1f}]")

    # ---- k 档：k 个 patch 贴四象限（源轮换） ----
    for k in (0, 1, 2, 4):
        rec = {t: [] for t in NECK_NODES}
        for r in recv:
            lb = recv_lb[r["stem"]]
            for i in range(k):
                cx, cy = QUAD[i]
                lb = paste(lb, patches[i % len(patches)], cx, cy)
            vals, _ = act_max(sess, in_name, t_idx, lb)
            for t in NECK_NODES:
                rec[t].append(vals[t])
        line = "  ".join(
            f"{t}: med={np.median(rec[t]):6.2f} max={max(rec[t]):6.2f} "
            f"超限{sum(v > CAL_MAX[t] for v in rec[t])}/{N_RECV}"
            for t in ("p3", "p4", "p5"))
        print(f"k={k}  {line}")

    # ---- 同 patch 叠贴能否超线性（4 贴同源） ----
    for si, stem in enumerate(SOURCES):
        rec = []
        for r in recv:
            lb = recv_lb[r["stem"]]
            for i in range(4):
                cx, cy = QUAD[i]
                lb = paste(lb, patches[si], cx, cy)
            vals, _ = act_max(sess, in_name, t_idx, lb)
            rec.append(vals["p5"])
        print(f"同源{stem}×4: med={np.median(rec):6.2f} max={max(rec):6.2f} "
              f"超限{sum(v > CAL_MAX['p5'] for v in rec)}/{N_RECV}")


if __name__ == "__main__":
    main()
