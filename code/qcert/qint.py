"""整数域量化语义与整数分位数（纯 numpy，x86 / 板端 Python 3.10 通用）。

对应 paper/02_创新点技术方案.md §4.2：
- 分位数只需要"计数"，不需要排序 -> 直方图 + 前缀和，O(n) 单遍；
- 桶内线性插值，使 alpha 误差远小于桶宽；
- 判决量保持整数，这里只做 q <-> float 的边界转换。

量化约定（与 AMCT / ATC 的逐层仿射 int8 一致）::

    a_q = clamp(round(a / delta) + zero_point, qmin, qmax)
    a   = (a_q - zero_point) * delta

即 ``delta`` 是"一个整数步"对应的物理量，``zero_point`` 是 0.0 对应的整数码。
对称量化时 zero_point = 0。
"""
from __future__ import annotations

import numpy as np

BITS_TO_DTYPE = {8: np.int8, 16: np.int16, 32: np.int32}


def qmin_qmax(bits: int = 8):
    """返回 (qmin, qmax)，例如 int8 -> (-128, 127)。"""
    return -(2 ** (bits - 1)), 2 ** (bits - 1) - 1


def quant_params_from_range(rmin: float, rmax: float, bits: int = 8,
                            symmetric: bool = True):
    """由物理量的取值范围推出 (delta, zero_point)。

    symmetric=True 时强制 zero_point=0（AMCT 默认的对称激活量化）。
    """
    qmin, qmax = qmin_qmax(bits)
    rmin, rmax = float(rmin), float(rmax)
    if not np.isfinite(rmin) or not np.isfinite(rmax) or rmax <= rmin:
        raise ValueError(f"非法 range: ({rmin}, {rmax})")
    if symmetric:
        amax = max(abs(rmin), abs(rmax))
        delta = amax / qmax if amax > 0 else 1.0
        return float(delta), 0
    delta = (rmax - rmin) / float(qmax - qmin)
    zp = int(round(qmin - rmin / delta))
    zp = int(np.clip(zp, qmin, qmax))
    return float(delta), zp


def quantize(x, delta: float, zero_point: int = 0, bits: int = 8) -> np.ndarray:
    """物理量 -> 整数码（四舍五入后截断到 [qmin, qmax]）。"""
    if delta <= 0:
        raise ValueError("delta 必须 > 0")
    qmin, qmax = qmin_qmax(bits)
    q = np.floor(np.asarray(x, dtype=np.float64) / delta + 0.5) + zero_point
    return np.clip(q, qmin, qmax).astype(BITS_TO_DTYPE[bits])


def dequantize(q, delta: float, zero_point: int = 0) -> np.ndarray:
    """整数码 -> 物理量。"""
    return (np.asarray(q, dtype=np.float64) - float(zero_point)) * float(delta)


def saturation_rate(q, bits: int = 8) -> float:
    """饱和率：落在 qmax 或 qmin 的元素占比。"""
    q = np.asarray(q)
    if q.size == 0:
        return 0.0
    qmin, qmax = qmin_qmax(bits)
    return float(np.mean((q >= qmax) | (q <= qmin)))


def saturation_rate_split(q, bits: int = 8):
    """分别返回 (上饱和率, 下饱和率)。"""
    q = np.asarray(q)
    if q.size == 0:
        return 0.0, 0.0
    qmin, qmax = qmin_qmax(bits)
    return float(np.mean(q >= qmax)), float(np.mean(q <= qmin))


# ---------------------------------------------------------------- 浮点直方图分位
def hist_quantile(x, q: float, bins: int = 4096, vrange=None) -> float:
    """直方图 + 前缀和求分位数（桶内线性插值）。

    与 ``np.quantile(x, q, method="linear")`` 的差别只有桶内插值方式，
    在 B=4096 时 alpha 误差 << 1e-4。

    Parameters
    ----------
    x : array_like
    q : float in [0, 1]
    bins : int, 直方图桶数（论文默认 4096）
    vrange : (lo, hi) or None, 统计范围；None 表示用样本 min/max
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    if n == 0:
        return float("nan")
    if n == 1:
        return float(x[0])
    q = float(np.clip(q, 0.0, 1.0))
    if vrange is None:
        lo, hi = float(x.min()), float(x.max())
    else:
        lo, hi = float(vrange[0]), float(vrange[1])
    if hi <= lo:
        return float(lo)
    bins = int(max(1, min(bins, 1_000_000)))

    # 把等于 hi 的样本并进最后一桶（np.histogram 的右端点行为）
    hist, edges = np.histogram(np.clip(x, lo, np.nextafter(hi, lo)), bins=bins,
                               range=(lo, hi))
    cum = np.cumsum(hist)
    pos = q * (n - 1)                      # 与 numpy linear 的 0-based 位置一致
    b = int(np.searchsorted(cum, pos, side="right"))
    b = min(b, bins - 1)
    before = int(cum[b - 1]) if b > 0 else 0
    cnt = int(hist[b])
    if cnt <= 0:
        return float(edges[b])
    frac = (pos - before + 0.5) / cnt
    frac = float(np.clip(frac, 0.0, 1.0))
    return float(edges[b] + frac * (edges[b + 1] - edges[b]))


def hist_quantile_int(x, q: float, bins: int = 4096) -> int:
    """整数分位数：输入/输出都是 int，桶内线性插值后四舍五入回整数。

    用于 ``d_q = (s_q - mu_q) << k`` 这类整数判决量的标定，
    全程不排序，O(n)。
    """
    xi = np.asarray(x).ravel().astype(np.int64)
    n = xi.size
    if n == 0:
        raise ValueError("空数组无法求分位数")
    lo, hi = int(xi.min()), int(xi.max())
    if hi == lo:
        return lo
    span = hi - lo + 1
    nb = int(max(1, min(bins, span)))
    # 单调的整数桶边界
    edges = lo + (np.arange(nb + 1, dtype=np.int64) * span) // nb
    edges[-1] = hi + 1
    idx = np.searchsorted(edges, xi, side="right") - 1
    np.clip(idx, 0, nb - 1, out=idx)
    hist = np.bincount(idx, minlength=nb)
    cum = np.cumsum(hist)
    pos = float(np.clip(q, 0.0, 1.0)) * (n - 1)
    b = int(np.searchsorted(cum, pos, side="right"))
    b = min(b, nb - 1)
    before = int(cum[b - 1]) if b > 0 else 0
    cnt = int(hist[b])
    if cnt <= 0:
        return int(edges[b])
    frac = (pos - before + 0.5) / cnt
    frac = float(np.clip(frac, 0.0, 1.0))
    val = edges[b] + frac * (edges[b + 1] - edges[b])
    return int(np.floor(val + 0.5))


# ---------------------------------------------------------------- 误差诊断工具
def quant_step_ratio(delta: float, value: float) -> float:
    """eps = delta / (2 * value)，即 paper 中的 epsilon_sigma。"""
    value = float(value)
    if value == 0:
        return float("inf")
    return float(delta) / (2.0 * abs(value))


def round_trip_error(x, delta: float, zero_point: int = 0, bits: int = 8) -> np.ndarray:
    """量化-反量化误差，用于 E9 的 |e_s| P99 统计。"""
    x = np.asarray(x, dtype=np.float64)
    return dequantize(quantize(x, delta, zero_point, bits), delta, zero_point) - x
