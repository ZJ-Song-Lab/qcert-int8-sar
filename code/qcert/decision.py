"""整数域判决层：消除除法的交叉相乘判决。

对应 paper/02_创新点技术方案.md §4.1::

    y_hat(u) = 1{ (s_q(u) - mu_q(u)) * 2^k  >  lambda_int * max(sigma_q(u), sigma_min) }

一般情形下 s / mu / sigma 来自三个不同层，scale 互不相同。把三者的物理量
展开后，判决等价于（全部为整数运算，无除法、无浮点）::

    s_q * 2^K - (dmu/ds * 2^K) * mu_q  >  (lambda * dsigma/ds * 2^K) * sigma_q

其中 K 是固定的定点移位（默认 20）。``decide()`` 实现的就是这个一般式；
``decide_simple()`` 是 ds == dmu 时的特例（与 02 文档逐字一致）。

标定（求 lambda）在 host 上做一次即可，属离线步骤；判决本身是纯整数。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .qint import hist_quantile, hist_quantile_int


@dataclass
class IntQuantSpec:
    """三路量化步长 + 判决所需常量。"""

    delta_s: float
    delta_mu: float
    delta_sigma: float
    sigma_min_q: int = 1        # 整数域防除零（sigma_q == 0 时兜底）
    shift_bits: int = 20        # 定点移位 K

    def check(self):
        for name in ("delta_s", "delta_mu", "delta_sigma"):
            v = float(getattr(self, name))
            if not np.isfinite(v) or v <= 0:
                raise ValueError(f"{name} 必须是正有限数, 得到 {v}")
        return self

    @property
    def shift(self) -> int:
        return int(self.shift_bits)


def lambda_int(spec: IntQuantSpec, lam: float, k: int = 16) -> int:
    """把浮点 lambda 换成整数阈值（ds == dmu 时使用）。

    对应 paper/02 §3.2::

        lambda_int = round( lambda * (dsigma/ds) * 2^k )
    """
    spec.check()
    return int(round(float(lam) * (spec.delta_sigma / spec.delta_s) * (2 ** k)))


def decide_simple(s_q, mu_q, sigma_q, k: int, lam_q: int,
                  sigma_min_q: int = 1) -> np.ndarray:
    """特例：s 与 mu 共用同一 scale。``d << k > lam_q * max(sigma_q, sigma_min)``。"""
    s = np.asarray(s_q, dtype=np.int64)
    mu = np.asarray(mu_q, dtype=np.int64)
    sig = np.maximum(np.asarray(sigma_q, dtype=np.int64), int(sigma_min_q))
    d = (s - mu) << int(k)
    return d > (int(lam_q) * sig)


def decide(s_q, mu_q, sigma_q, lam: float, spec: IntQuantSpec) -> np.ndarray:
    """一般情形下的整数判决（三路 scale 可不同），全程 int64。"""
    spec.check()
    r = 1 << spec.shift
    b = int(round(spec.delta_mu / spec.delta_s * r))
    c = int(round(float(lam) * spec.delta_sigma / spec.delta_s * r))
    lhs = np.int64(r) * np.asarray(s_q, dtype=np.int64) \
        - np.int64(b) * np.asarray(mu_q, dtype=np.int64)
    rhs = np.int64(c) * np.maximum(np.asarray(sigma_q, dtype=np.int64),
                                   int(spec.sigma_min_q))
    return lhs > rhs


def decide_float(z, lam: float) -> np.ndarray:
    """FP32 参照判决。"""
    return np.asarray(z, dtype=np.float64) > float(lam)


def z_from_int(s_q, mu_q, sigma_q, spec: IntQuantSpec,
               sigma_eps: float = 1e-6) -> np.ndarray:
    """由整数码重建浮点 z（仅用于诊断与标定，不参与判决）。

    z = (s - mu) / max(sigma, sigma_eps)
    """
    spec.check()
    s = np.asarray(s_q, dtype=np.float64) * spec.delta_s
    mu = np.asarray(mu_q, dtype=np.float64) * spec.delta_mu
    sig = np.asarray(sigma_q, dtype=np.float64) * spec.delta_sigma
    return (s - mu) / np.maximum(sig, float(sigma_eps))


def calibrate_float(z, mask, alpha: float) -> float:
    """在杂波位置 mask 上取 1-alpha 分位数作为 lambda。"""
    z = np.asarray(z, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    if m.shape != z.shape:
        raise ValueError(f"mask 形状 {m.shape} 与 z {z.shape} 不一致")
    vals = z[m]
    if vals.size == 0:
        raise ValueError("杂波集合为空，无法标定")
    return float(np.quantile(vals, 1.0 - float(alpha)))


def calibrate_int(s_q, mu_q, sigma_q, mask, alpha: float, spec: IntQuantSpec,
                  bins: int = 4096) -> float:
    """整数链路的 lambda（host 上做一次）。

    先用 ``z_from_int`` 重建 z（与判决完全同源），再取分位数。
    这一步不改变"判决在整数域完成"的性质，只是把一次性标定留在 host。
    """
    z = z_from_int(s_q, mu_q, sigma_q, spec)
    return calibrate_float(z, mask, alpha)


def calibrate_int_divfree(s_q, mu_q, sigma_q, mask, alpha: float,
                          spec: IntQuantSpec, k: int = 16,
                          bins: int = 4096) -> int:
    """整数分位数标定（ds == dmu 特例）：返回 lam_q。

    d_q = (s_q - mu_q) << k，直接在整数域上取分位数，不排序。
    用于验证 "整数分位数 vs np.quantile" 的等价性（tests/test_qint.py）。
    """
    s = np.asarray(s_q, dtype=np.int64)
    mu = np.asarray(mu_q, dtype=np.int64)
    sig = np.maximum(np.asarray(sigma_q, dtype=np.int64), int(spec.sigma_min_q))
    d = (s - mu) << int(k)
    ratio = np.floor_divide(d, sig)          # 整数域 z 的定点表示（floor）
    vals = np.asarray(ratio)[np.asarray(mask, dtype=bool)]
    if vals.size == 0:
        raise ValueError("杂波集合为空，无法标定")
    return int(hist_quantile_int(vals, 1.0 - float(alpha), bins=bins))


def compare_decisions(pred_int, pred_float, gt_mask=None) -> dict:
    """比较整数判决与 FP32 判决的一致性（04 §2.2 的"判决不一致率"）。"""
    pi = np.asarray(pred_int, dtype=bool)
    pf = np.asarray(pred_float, dtype=bool)
    if pi.shape != pf.shape:
        raise ValueError("两个判决数组形状不一致")
    n = pi.size
    diff = pi != pf
    out = {
        "n": int(n),
        "n_diff": int(diff.sum()),
        "disagree_rate": float(diff.mean()) if n else 0.0,
        "agree_rate": float(1.0 - diff.mean()) if n else 1.0,
        "n_pos_int": int(pi.sum()),
        "n_pos_float": int(pf.sum()),
    }
    if gt_mask is not None:
        g = np.asarray(gt_mask, dtype=bool)
        if g.shape != pi.shape:
            raise ValueError("gt_mask 形状不一致")
        out["diff_in_gt_frac"] = float((diff & g).sum() / max(int(g.sum()), 1))
    return out


def quantization_error_stats(s_float, s_q, delta: float, zero_point: int = 0) -> dict:
    """|e_s| 的分布（04 §2.2 验收误差）。"""
    s_hat = (np.asarray(s_q, dtype=np.float64) - zero_point) * delta
    e = np.abs(s_hat - np.asarray(s_float, dtype=np.float64))
    return {
        "mean": float(e.mean()),
        "p50": float(np.percentile(e, 50)),
        "p95": float(np.percentile(e, 95)),
        "p99": float(np.percentile(e, 99)),
        "max": float(e.max()),
        "delta": float(delta),
        "bound": float(delta / 2.0),
    }
