# -*- coding: utf-8 -*-
"""E2 压测集候选变换本地验证（fp ONNX CPU 仿真，不上板）。

背景（2026-11-04 板上实测）：
  v1 压测集 clip(img*c, 0, 255) 纯增益+裁剪无效——×3 时像素大量饱和到
  0/255，图像结构破坏，Neck 激活反降（p5 act_max med: c1 10.27 → c3 7.77）。
  且 AMCT 输入 AscendQuant 标定范围 [0,1]（校准用同一 preprocess），
  任何把输入推过 1 的增益都会在输入量化处被 clip，引入混淆项。

本脚本：本地 fp ONNX 对候选变换（全部 uint8 [0,255] 合法）测 Neck 三层
SiLU 输出激活 max，对比 AMCT 标定上限（P3/P4/P5 = 12.08/17.43/38.61），
筛选能逼出上饱和且随强度单调的分档方案，用于重生成 E2 压测集。

判定：某档 ≥ 半数图 P5 act_max > 38.61 → 可产生饱和；
      分档 act_max 单调上升 → 可作 E2 新压测集。
"""
from __future__ import annotations

import json
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
STRESS = DEPLOY.parent / "data" / "stress"
CAL_MAX = {"p3": 12.08, "p4": 17.43, "p5": 38.61}   # AMCT 标定上限(SiLU 输出)
NECK_NODES = {"p3": 16, "p4": 19, "p5": 22}
N_PROBE = 12


# ---------- 候选变换（全部 uint8 合法，输入不超 [0,255]） ----------
def tf_gain(img, c):
    return np.clip(img.astype(np.float64) * c, 0, 255).astype(np.uint8)


def tf_gamma(img, g):
    x = img.astype(np.float64) / 255.0
    return np.round(255.0 * np.power(x, g)).astype(np.uint8)


def tf_stretch(img, lo_q=0.02, hi_q=0.98):
    x = img.astype(np.float64)
    lo, hi = np.quantile(x, [lo_q, hi_q])
    if hi - lo < 1e-6:
        return img.copy()
    return np.clip((x - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)


def tf_clahe(img, clip):
    import cv2
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    eq = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(gray)
    return cv2.cvtColor(eq, cv2.COLOR_GRAY2RGB)


def tf_clutter(img, L, n_spots, sigma, peak, seed=0):
    """合成海杂波：乘性相干斑（gamma 分布，视数 L 越小重尾越强，均值 1）
    + 高斯强散射亮斑（峰值 peak，max 融合不叠加溢出）。uint8 合法。"""
    rng = np.random.default_rng(seed)
    out = img.astype(np.float64) * rng.gamma(
        shape=L, scale=1.0 / L, size=img.shape[:2])[..., None]
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    for _ in range(n_spots):
        cy, cx = int(rng.integers(0, h)), int(rng.integers(0, w))
        r = max(1, int(sigma * 3))
        y0, y1 = max(0, cy - r), min(h, cy + r + 1)
        x0, x1 = max(0, cx - r), min(w, cx + r + 1)
        g = np.exp(-(((yy[y0:y1, x0:x1] - cy) ** 2 +
                      (xx[y0:y1, x0:x1] - cx) ** 2) / (2.0 * sigma * sigma)))
        out[y0:y1, x0:x1] = np.maximum(out[y0:y1, x0:x1], peak * g[..., None])
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


TF = {
    "base":      lambda im: im,
    "gain1.5":   lambda im: tf_gain(im, 1.5),     # 旧方案对照（预期无效）
    "gain2":     lambda im: tf_gain(im, 2.0),
    "stretch":   lambda im: tf_stretch(im),
    "gamma0.6":  lambda im: tf_gamma(im, 0.6),
    "gamma0.4":  lambda im: tf_gamma(im, 0.4),
    "clahe2":    lambda im: tf_clahe(im, 2.0),
    "clahe4":    lambda im: tf_clahe(im, 4.0),
    "clutter_w": lambda im: tf_clutter(im, 6.0, 20, 3.0, 230),
    "clutter_m": lambda im: tf_clutter(im, 2.5, 80, 3.0, 255),
    "clutter_s": lambda im: tf_clutter(im, 1.0, 250, 4.0, 255),
}


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
        cands = [nd.name for nd in model.graph.node
                 if nd.name and "model.16" in nd.name]
        raise SystemExit(f"[err] 未找到 {missing} 的 SiLU 节点; model.16 邻域: {cands}")
    return found


def main():
    manifest = json.load(open(STRESS / "manifest.json", encoding="utf-8"))
    top = sorted(manifest["p99_fp16"],
                 key=manifest["p99_fp16"].get, reverse=True)[:N_PROBE]
    print(f"[probe] top{N_PROBE} 高杂波 stem: {top[:4]} ...")

    m = onnx.load(str(FP_ONNX))
    targets = neck_silu_tensors(m)
    print(f"[probe] Neck SiLU 输出张量: {targets}")
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
    print(f"[probe] 输入={in_name} 输出数={len(out_names)}")

    stats = {}
    for tf_name, fn in TF.items():
        rec = {t: [] for t in NECK_NODES}
        for stem in top:
            img = np.asarray(Image.open(STRESS / f"c1_{stem}.jpg").convert("RGB"))
            x, _, _ = preprocess(fn(img), 640)
            outs = sess.run(None, {in_name: x})
            for t, i in t_idx.items():
                a = np.asarray(outs[i], dtype=np.float64)
                rec[t].append(float(a.max()))
        stats[tf_name] = rec
        line = "  ".join(
            f"{t}: med={np.median(rec[t]):7.2f} max={max(rec[t]):7.2f} "
            f"超限{sum(v > CAL_MAX[t] for v in rec[t])}/{N_PROBE}"
            for t in ("p3", "p4", "p5"))
        print(f"{tf_name:<10} {line}")

    out_json = HERE / "_probe_stress_tf.json"
    json.dump({"cal_max": CAL_MAX, "n_probe": N_PROBE, "stems": top,
               "act_max": stats}, open(out_json, "w"), indent=1)
    print(f"[ok] {out_json}")


if __name__ == "__main__":
    main()
