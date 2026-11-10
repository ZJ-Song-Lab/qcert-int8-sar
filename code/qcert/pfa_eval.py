"""P_fa 双口径评测、分桶、bootstrap CI、P_d@P_fa（paper/04 §2 的报数口径）。

三套集合严格分离（04 §1.3）：
- ``Omega_cal``  ：契约标定集（求 lambda），来自 GT 框外 + 块抽样的**一半**块；
- ``Omega_meas`` ：测量集，来自**另一半互不相交**的块，用于报实测 P_fa；
- 二者都排除了 GT 框（含保护带），因此都是"杂波位置"。

位置单位是**检测器网格中心**（N=8400 个格子），而不是原图像素。这样 FP32 与
INT8 两条链路的 Omega 天然逐位置对齐，`R` 才有可比性。
"""
from __future__ import annotations

import numpy as np


# ------------------------------------------------------------------ 网格几何
def cell_centers(head_sizes, imgsz: int = 640) -> np.ndarray:
    """三个尺度的网格中心（letterbox 输入坐标系），顺序与 concat 一致。"""
    if isinstance(head_sizes, np.ndarray):
        head_sizes = head_sizes.reshape(-1, 2).tolist()
    out = []
    for h, w in head_sizes:
        h, w = int(h), int(w)
        stride = float(imgsz) / h
        yy, xx = np.meshgrid((np.arange(h) + 0.5) * stride,
                             (np.arange(w) + 0.5) * stride, indexing="ij")
        out.append(np.stack([xx.ravel(), yy.ravel()], axis=1))
    return np.concatenate(out, axis=0).astype(np.float32)


def boxes_to_letterbox(boxes_xyxy, meta: dict):
    """原图 xyxy -> letterbox 输入坐标（用 dump 时记录的 ratio / pad）。"""
    b = np.asarray(boxes_xyxy, dtype=np.float64).reshape(-1, 4).copy()
    if b.size == 0:
        return np.zeros((0, 4))
    ratio = float(meta.get("ratio", 1.0))
    dw, dh = meta.get("pad", (0, 0))
    b[:, [0, 2]] = b[:, [0, 2]] * ratio + float(dw)
    b[:, [1, 3]] = b[:, [1, 3]] * ratio + float(dh)
    return b


def points_in_boxes(centers, boxes, guard_px: float = 0.0) -> np.ndarray:
    """每个网格中心是否落在任一 GT 框（外扩 guard_px）内。"""
    c = np.asarray(centers, dtype=np.float64)
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    inside = np.zeros(c.shape[0], dtype=bool)
    for x1, y1, x2, y2 in b:
        inside |= ((c[:, 0] >= x1 - guard_px) & (c[:, 0] <= x2 + guard_px) &
                   (c[:, 1] >= y1 - guard_px) & (c[:, 1] <= y2 + guard_px))
    return inside


def block_ids(centers, block_px: float = 64.0) -> np.ndarray:
    """按绝对坐标分块（默认 64px = 8 个 stride-8 格子）。"""
    c = np.asarray(centers, dtype=np.float64)
    bx = np.floor(c[:, 0] / block_px).astype(np.int64)
    by = np.floor(c[:, 1] / block_px).astype(np.int64)
    return bx * 1000003 + by


def build_omega(centers, boxes=None, guard_px: float = 8.0,
                block_px: float = 64.0, keep: float = 1.0 / 64.0,
                seed: int = 0, disjoint: bool = True):
    """构造 (Omega_cal, Omega_meas)，两者在 disjoint=True 时互不相交。

    Returns
    -------
    mask_cal, mask_meas, info(dict)
    """
    centers = np.asarray(centers)
    inside = (points_in_boxes(centers, boxes, guard_px)
              if boxes is not None and len(boxes) else np.zeros(centers.shape[0], bool))
    valid = ~inside
    ids = block_ids(centers, block_px)
    uniq, inv = np.unique(ids[valid], return_inverse=True)
    if uniq.size < 2:
        raise ValueError("有效杂波块不足 2 个：检查 GT 覆盖或 block_px 是否过大")

    rng = np.random.default_rng(seed)
    order = rng.permutation(uniq.size)
    n_keep = max(1, int(round(uniq.size * float(keep))))
    n_keep = min(n_keep, max(1, uniq.size // 2))
    cal_blocks = order[:n_keep]
    meas_blocks = order[n_keep:2 * n_keep] if disjoint else order[:n_keep]

    cal_ids = np.zeros(uniq.size, dtype=bool)
    cal_ids[cal_blocks] = True
    meas_ids = np.zeros(uniq.size, dtype=bool)
    meas_ids[meas_blocks] = True

    valid_idx = np.flatnonzero(valid)
    mask_cal = np.zeros(centers.shape[0], dtype=bool)
    mask_meas = np.zeros(centers.shape[0], dtype=bool)
    mask_cal[valid_idx] = cal_ids[inv]
    mask_meas[valid_idx] = meas_ids[inv]

    info = {
        "n_cells": int(centers.shape[0]),
        "n_valid": int(valid.sum()),
        "n_blocks": int(uniq.size),
        "n_blocks_cal": int(cal_blocks.size),
        "n_blocks_meas": int(meas_blocks.size),
        "n_cal": int(mask_cal.sum()),
        "n_meas": int(mask_meas.sum()),
        "n_gt_cells": int(inside.sum()),
        "block_px": float(block_px),
        "keep": float(keep),
        "disjoint": bool(disjoint),
    }
    return mask_cal, mask_meas, info


# ------------------------------------------------------------------ 契约指标
def calibrate_lambda(z, mask, alpha: float) -> float:
    """1-alpha 分位数（线性插值，与 np.quantile 默认一致）。"""
    z = np.asarray(z, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    vals = z[m]
    if vals.size == 0:
        raise ValueError("Omega_cal 为空")
    return float(np.quantile(vals, 1.0 - float(alpha)))


def measure_pfa(z, mask, lam: float) -> float:
    """Omega_meas 上的逐位置虚警率。"""
    z = np.asarray(z, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    vals = z[m]
    if vals.size == 0:
        raise ValueError("Omega_meas 为空")
    return float(np.mean(vals > float(lam)))


def scene_metrics(z, mask_cal, mask_meas, alpha: float) -> dict:
    lam = calibrate_lambda(z, mask_cal, alpha)
    pfa = measure_pfa(z, mask_meas, lam)
    return {"lambda": lam, "pfa": pfa, "n_meas": int(np.asarray(mask_meas).sum())}


def pfa_ratio(pfa_int: float, pfa_ref: float) -> float:
    """主指标 R = P_fa(INT8) / P_fa(FP32)。分母为 0 时退化为 inf/1。"""
    if pfa_ref <= 0:
        return float("inf") if pfa_int > 0 else 1.0
    return float(pfa_int / pfa_ref)


def deviation_from_alpha(pfa: float, alpha: float) -> float:
    """|log10(P_fa / alpha)|：一句话表述"契约是否还守"。"""
    if pfa <= 0:
        return float("inf")
    return float(abs(np.log10(pfa / float(alpha))))


def log10_ratio(a: float, b: float) -> float:
    if b <= 0 or a <= 0:
        return float("nan")
    return float(np.log10(a / b))


# ------------------------------------------------------------------ 统计纪律
def bootstrap_ci(values, n_boot: int = 1000, stat=np.mean, ci: float = 0.95,
                 seed: int = 0):
    """场景级 bootstrap：返回 (lo, hi, point)。"""
    v = np.asarray(values, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(stat(v))
    if v.size == 1:
        return point, point, point
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(int(n_boot), v.size))
    stats = stat(v[idx], axis=1) if stat is np.mean else np.array(
        [stat(v[i]) for i in idx])
    lo = float(np.percentile(stats, 100 * (1 - ci) / 2))
    hi = float(np.percentile(stats, 100 * (1 + ci) / 2))
    return lo, hi, point


def bootstrap_ratio_ci(num_values, den_values, n_boot: int = 1000, seed: int = 0):
    """配对 bootstrap 的 R 区间：同场景重采样，保留配对关系。"""
    a = np.asarray(num_values, dtype=np.float64).ravel()
    b = np.asarray(den_values, dtype=np.float64).ravel()
    n = min(a.size, b.size)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    a, b = a[:n], b[:n]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(int(n_boot), n))
    ma = a[idx].mean(axis=1)
    mb = b[idx].mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        stats = np.where(mb > 0, ma / np.where(mb > 0, mb, 1), np.nan)
    stats = stats[np.isfinite(stats)]
    if stats.size == 0:
        return float("nan"), float("nan"), float("nan")
    return (float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5)),
            float(a.mean() / b.mean()) if b.mean() > 0 else float("nan"))


def n_eff(mask, centers=None, block_px: float = 64.0) -> int:
    """块抽样后的有效独立样本数（= 被选中的块数）。"""
    m = np.asarray(mask, dtype=bool)
    if centers is None:
        return int(m.sum())
    ids = block_ids(np.asarray(centers), block_px)[m]
    return int(np.unique(ids).size) if ids.size else 0


# ------------------------------------------------------------------ P_d @ P_fa
def pd_at_pfa(z_h0, z_h1, alphas, n_boot: int = 0, seed: int = 0):
    """扫 alpha 得 (P_fa, P_d) 曲线，并读指定 alpha 处的 P_d。

    阈值始终由 H0 分位数给出（这正是"误警率契约"的定义）。
    """
    z0 = np.asarray(z_h0, dtype=np.float64).ravel()
    z1 = np.asarray(z_h1, dtype=np.float64).ravel()
    if z0.size == 0 or z1.size == 0:
        raise ValueError("H0/H1 样本为空")
    rows = []
    for a in alphas:
        lam = float(np.quantile(z0, 1.0 - float(a)))
        pfa = float(np.mean(z0 > lam))
        pd = float(np.mean(z1 > lam))
        row = {"alpha": float(a), "lambda": lam, "pfa": pfa, "pd": pd,
               "n_h0": int(z0.size), "n_h1": int(z1.size)}
        if n_boot > 0:
            lo, hi, pt = bootstrap_ci((z1 > lam).astype(np.float64), n_boot, seed=seed)
            row.update({"pd_lo": lo, "pd_hi": hi, "pd_point": pt})
        rows.append(row)
    return rows


def dprime(z_h0, z_h1) -> float:
    """检测可分性 d' = (mean(z|H1) - mean(z|H0)) / std(z|H0)。"""
    z0 = np.asarray(z_h0, dtype=np.float64).ravel()
    z1 = np.asarray(z_h1, dtype=np.float64).ravel()
    s = z0.std()
    return float((z1.mean() - z0.mean()) / s) if s > 0 else float("inf")


# ------------------------------------------------------------------ Prop.2 的界
def prop2_delta(lam: float, sigma: float, delta_s: float, delta_mu: float,
                delta_sigma: float) -> float:
    """Prop.2 的 delta = (ds/2 + dmu/2 + lambda*dsigma/2) / sigma。"""
    sigma = float(sigma)
    if sigma <= 0:
        raise ValueError("sigma 必须 > 0")
    return (float(delta_s) / 2 + float(delta_mu) / 2
            + float(lam) * float(delta_sigma) / 2) / sigma


def empirical_density(z, at: float, bw: float | None = None) -> float:
    """高斯核经验密度（用于 Lipschitz 常数 L 的估计）。"""
    z = np.asarray(z, dtype=np.float64).ravel()
    n = z.size
    if n == 0:
        return float("nan")
    if bw is None:
        # Silverman 规则
        sd = z.std(ddof=1) if n > 1 else 1.0
        bw = 0.9 * sd * n ** (-1.0 / 5.0) if sd > 0 else 1.0
    u = (z - float(at)) / bw
    return float(np.exp(-0.5 * u * u).sum() / (n * bw * np.sqrt(2 * np.pi)))


def prop2_bound_pfa(z_clutter, lam: float, sigma: float, delta_s: float,
                    delta_mu: float, delta_sigma: float, at: float | None = None):
    """返回 |ΔP_fa| 的解析上界与所需中间量。"""
    at = float(lam) if at is None else float(at)
    L = empirical_density(z_clutter, at)
    d = prop2_delta(lam, sigma, delta_s, delta_mu, delta_sigma)
    return {"L": L, "delta": d, "bound": float(L * d), "at": at}


def spearman(x, y) -> float:
    """Spearman 秩相关（不引入 scipy）。"""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = min(x.size, y.size)
    if n < 2:
        return float("nan")
    x, y = x[:n], y[:n]

    def rank(v):
        order = np.argsort(v, kind="mergesort")
        r = np.empty(v.size, dtype=np.float64)
        r[order] = np.arange(v.size, dtype=np.float64)
        # 处理并列：取平均秩
        _, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
        sums = np.bincount(inv, weights=r)
        r = (sums / cnt)[inv]
        return r

    rx, ry = rank(x), rank(y)
    rx -= rx.mean()
    ry -= ry.mean()
    den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / den) if den > 0 else float("nan")
