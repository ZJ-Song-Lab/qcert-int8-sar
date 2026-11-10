"""合成杂波场：E1/E4/E6/E7/E9 的离线通路，以及 M0 的管线自检。

**定位**：这不是用来"造论文数据"的，只用来
1) 在没有板子的时候把 E0–E9 的代码路径全部跑通（demo dump）；
2) 给 E1/E4 这类需要可控 ground truth 的实验提供逐位置变化的 ``sigma(u)``。

关键点（v3 的 R11 教训）：``sigma`` 必须**逐位置变化**。若用全局常数，
``Delta_sigma`` 只会造成一次确定性缩放，自标定会把它完全抵消，
Fig.4 就是一条平线，什么也测不出来。
"""
from __future__ import annotations

import numpy as np

from qcert.pfa_eval import block_ids, cell_centers
from qcert.scores import ScoreField

DEFAULT_HEADS = ((80, 80), (40, 40), (20, 20))
IMGSZ = 640


def heads_array(heads=DEFAULT_HEADS) -> np.ndarray:
    return np.asarray(heads, dtype=np.int32).reshape(-1, 2)


def scene_geometry(imgsz: int = IMGSZ, heads=DEFAULT_HEADS):
    hs = heads_array(heads)
    return cell_centers(hs, imgsz), hs


def make_scene(seed: int = 0, n_target: int = 0, sigma0: float = 0.30,
               sigma_ln: float = 0.45, mu0: float = 0.06, corr_px: float = 96.0,
               imgsz: int = IMGSZ, heads=DEFAULT_HEADS, target_snr: float = 2.2,
               contrast: float = 1.0, sea_state: str = "moderate",
               dist: str = "gauss"):
    """生成一个场景的 (s, mu, sigma, z, centers, boxes, target_idx)。

    - ``sigma(u)`` 由"块相关 + 白噪声"两层构成，保证逐位置不同；
    - ``contrast`` 模拟压测集的人工增益（E2 用来逼出饱和）；
    - ``dist='t3'`` 换成 t(3) 重尾（E1 §4.1 的第二个分布）。
    """
    centers, hs = scene_geometry(imgsz, heads)
    n = centers.shape[0]
    rng = np.random.default_rng(seed)
    ids = block_ids(centers, corr_px)
    uniq, inv = np.unique(ids, return_inverse=True)

    sigma = sigma0 * np.exp(sigma_ln * rng.standard_normal(uniq.size))[inv]
    sigma *= np.exp(0.25 * sigma_ln * rng.standard_normal(n))
    mu = mu0 + 0.02 * rng.standard_normal(uniq.size)[inv]

    if dist == "t3":
        z = rng.standard_t(3, n) / np.sqrt(3.0)      # 单位方差
    elif dist == "gauss":
        z = rng.standard_normal(n)
    else:
        raise ValueError(f"未知分布: {dist}")

    s = mu + sigma * z
    boxes, target_idx = [], np.zeros(n, dtype=bool)

    if n_target > 0:
        pick = rng.choice(n, size=min(int(n_target), max(1, n // 50)), replace=False)
        for i in pick:
            cx, cy = centers[i]
            half = float(rng.uniform(12.0, 32.0))
            boxes.append([cx - half, cy - half, cx + half, cy + half])
        if boxes:
            from qcert.pfa_eval import points_in_boxes
            target_idx = points_in_boxes(centers, np.asarray(boxes), 0.0)
            amp = target_snr + np.abs(rng.standard_normal(int(target_idx.sum())))
            s[target_idx] = mu[target_idx] + sigma[target_idx] * amp

    s = s * float(contrast)
    return {
        "s": s, "mu": mu, "sigma": sigma, "z": z,
        "centers": centers, "head_sizes": hs,
        "boxes": np.asarray(boxes, dtype=np.float64).reshape(-1, 4),
        "target_idx": target_idx,
        "meta": {"imgsz": imgsz, "iw": float(imgsz), "ih": float(imgsz),
                 "ratio": 1.0, "pad": [0.0, 0.0], "nc": 1,
                 "sea_state": sea_state, "contrast": float(contrast),
                 "dist": dist},
    }


def to_field(scene, impl: str = "fp32", delta: float | None = None,
             zero_point: int = 0, bits: int = 8, stem: str = "synth",
             logit: bool = True) -> ScoreField:
    """把场景变成 ScoreField；给 ``delta`` 时按 AMCT 的仿射语义量化。"""
    from qcert.qint import dequantize, quantize

    s = np.asarray(scene["s"], dtype=np.float64)
    meta = dict(scene["meta"])
    meta["impl"] = impl
    meta["boxes"] = np.asarray(scene["boxes"], dtype=np.float64).tolist()
    meta["synth"] = True

    if delta:
        s_q = quantize(s, delta, zero_point, bits=bits)
        prob = dequantize(s_q, delta, zero_point).astype(np.float32)
    else:
        s_q, prob = None, s.astype(np.float32)

    return ScoreField(
        prob=prob,
        logit=(s.astype(np.float32) if logit else None),
        centers=np.asarray(scene["centers"], dtype=np.float32),
        head_sizes=np.asarray(scene["head_sizes"], dtype=np.int32),
        s_q=s_q, delta=(float(delta) if delta else None), zero_point=int(zero_point),
        meta=meta, stem=stem)


def write_demo_dumps(out_dir, n_scenes: int = 12, impl: str = "fp32",
                     delta: float | None = None, seed0: int = 1000,
                     contrasts=(1.0,), sea_states=("calm", "moderate", "rough"),
                     n_target: int = 3, sigma_ln: float = 0.45):
    """写一批 demo dump（M0 的管线自检用，不是实验数据）。"""
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(int(n_scenes)):
        sea = sea_states[i % len(sea_states)]
        con = contrasts[i % len(contrasts)]
        sc = make_scene(seed=seed0 + i, n_target=n_target,
                        sigma_ln=sigma_ln, sea_state=sea, contrast=con)
        f = to_field(sc, impl=impl, delta=delta, stem=f"demo_{i:03d}")
        from qcert.scores import save_score_npz
        paths.append(save_score_npz(out / f"{f.stem}.npz", f))
    return paths


def clutter_sample(n: int = 200_000, sigma0: float = 0.30,
                   sigma_ln: float = 0.45, dist: str = "gauss",
                   corr: int = 64, seed: int = 0):
    """纯杂波样本 ``(s, mu, sigma)``，逐位置 sigma 变化。

    给 E1/E4 用来验证 Prop.2：不依赖网格几何，只要样本量够。
    相关性用"每 ``corr`` 个样本共享一个块因子"近似，便于分析 n_eff 的影响。
    """
    rng = np.random.default_rng(seed)
    n = int(n)
    nb = max(1, n // max(1, int(corr)))
    block = np.exp(sigma_ln * rng.standard_normal(nb))
    sigma = sigma0 * np.repeat(block, int(np.ceil(n / nb)))[:n]
    sigma *= np.exp(0.25 * sigma_ln * rng.standard_normal(n))
    mu = 0.06 + 0.02 * np.repeat(rng.standard_normal(nb),
                                 int(np.ceil(n / nb)))[:n]
    if dist == "t3":
        z = rng.standard_t(3, n) / np.sqrt(3.0)
    elif dist == "gauss":
        z = rng.standard_normal(n)
    else:
        raise ValueError(f"未知分布: {dist}")
    return mu + sigma * z, mu, sigma


def quantize_sigma(sigma, eps_sigma: float, sigma_ref: float | None = None,
                   bits: int = 16, mode: str = "linear"):
    """只量化 sigma 一路（E1 的分母机制）。

    ``eps_sigma = Delta_sigma / (2 sigma_ref)`` 是**标称**步长，逐位置的
    实际 ``eps_sigma(u) = Delta_sigma/(2 sigma(u))`` 随 sigma 变化。
    mode: linear | log（``w = log sigma`` 上量化）| fp16（仅用于对照）
    """
    sigma = np.asarray(sigma, dtype=np.float64)
    if float(eps_sigma) <= 0.0:
        return sigma.copy()                     # eps=0 是对照点：不量化
    ref = float(sigma_ref) if sigma_ref else float(np.median(sigma))
    if mode == "fp16":
        return sigma.astype(np.float16).astype(np.float64)
    if mode == "log":
        w = np.log(sigma)
        dw = 2.0 * float(eps_sigma)
        wq = np.round(w / dw) * dw
        return np.exp(wq)
    if mode != "linear":
        raise ValueError(f"未知 mode: {mode}")
    step = 2.0 * float(eps_sigma) * ref
    return np.round(sigma / step) * step


def actual_eps_sigma(sigma, sigma_q) -> np.ndarray:
    """逐位置实现出来的 eps_sigma(u) = |Delta_sigma| / (2 sigma(u))。"""
    sigma = np.asarray(sigma, dtype=np.float64)
    d = np.abs(np.asarray(sigma_q, dtype=np.float64) - sigma)
    return d / np.maximum(2.0 * sigma, 1e-12)


def sigma_steps(sigma, centers=None, eps_sigma: float = 0.0,
                block_px: float = 64.0, mode: str = "linear") -> dict:
    """在**标定样本**上定一次 σ 的量化器步长表（AMCT 语义，E4 用）。

    返回的字典可以原样作用到另一批样本上（见 :func:`apply_sigma_steps`），
    这样"标定"和"测量"用的就是同一个量化器，而不会把测量样本的信息
    泄漏进步长里。

    mode: linear（σ 上量化）| log（``w = log σ`` 上量化）| fp16（对照）
    ``block_px=None`` 表示逐张量（一条 scale），否则按 ``block_px`` 逐通道。
    """
    sigma = np.asarray(sigma, dtype=np.float64)
    eps = float(eps_sigma)
    if eps <= 0.0 or mode == "fp16" or sigma.size == 0:
        return {"mode": mode, "per_block": False, "steps": {}}
    width = 2.0 * eps
    if mode == "log":
        if centers is None or block_px is None:
            return {"mode": "log", "per_block": False, "steps": {"*": width}}
        ids = block_ids(np.asarray(centers, dtype=np.float64), block_px)
        return {"mode": "log", "per_block": True,
                "steps": {int(b): width for b in np.unique(ids)}}
    if mode != "linear":
        raise ValueError(f"未知 mode: {mode}")
    if centers is None or block_px is None:
        return {"mode": "linear", "per_block": False,
                "steps": {"*": width * float(np.median(sigma))}}
    ids = block_ids(np.asarray(centers, dtype=np.float64), block_px)
    return {"mode": "linear", "per_block": True,
            "steps": {int(b): width * float(np.median(sigma[ids == b]))
                      for b in np.unique(ids)}}


def apply_sigma_steps(sigma, steps: dict, centers=None,
                      block_px: float = 64.0) -> np.ndarray:
    """把 :func:`sigma_steps` 的步长表作用到任意一批 σ 上。"""
    sigma = np.asarray(sigma, dtype=np.float64)
    mode = steps.get("mode", "linear")
    tbl = steps.get("steps") or {}
    if mode == "fp16":
        return sigma.astype(np.float16).astype(np.float64)
    if not tbl:
        return sigma.copy()                     # eps=0 对照：不量化
    if not steps.get("per_block"):
        step = float(tbl["*"])
        if mode == "log":
            return np.exp(np.round(np.log(sigma) / step) * step)
        return np.round(sigma / step) * step

    ids = block_ids(np.asarray(centers, dtype=np.float64), block_px)
    default = float(np.median(list(tbl.values())))
    out = sigma.copy()
    for b in np.unique(ids):
        m = ids == b
        step = float(tbl.get(int(b), default))
        if mode == "log":
            out[m] = np.exp(np.round(np.log(sigma[m]) / step) * step)
        else:
            out[m] = np.round(sigma[m] / step) * step
    return out


def delta_for_range(s_list, bits: int = 8) -> tuple:
    """由一批 FP32 分数场（正常对比度）定一条标定 scale，供压测集复用。

    这正是 AMCT 的行为：**离线标定一次**，部署时不再变。E2 里 ×2/×3 的
    饱和就是"用正常海的 scale 去装高对比度数据"撞出来的。
    """
    from qcert.qint import quant_params_from_range

    lo = min(float(np.min(s)) for s in s_list)
    hi = max(float(np.max(s)) for s in s_list)
    return quant_params_from_range(lo, hi, bits=bits, symmetric=True)


def make_pair(seed: int = 0, contrast: float = 1.0, delta: float | None = None,
              bits: int = 8, impl_ref: str = "fp32", impl_test: str = "int8",
              stem: str = "s", **scene_kw):
    """同一场景的 (scene, FP32 field, 量化 field) 三元组。"""
    scene = make_scene(seed=seed, contrast=contrast, **scene_kw)
    f32 = to_field(scene, impl=impl_ref, stem=stem)
    q = to_field(scene, impl=impl_test, delta=delta, bits=bits, stem=stem)
    return scene, f32, q
