"""阈值搬运（模块 6，对应 paper/04 §8 的 E5）。

核心是一个**纯函数**：只依赖两份 scale 表，不碰任何数据::

    lambda_int = round( lambda * (delta_sigma / delta_s) * 2^k )

若已有旧整数阈值 ``lambda_int_old``（旧配置），换到新配置只需::

    lambda_int_new = round( lambda_int_old
                            * (delta_sigma_new / delta_sigma_old)
                            * (delta_s_old / delta_s_new) )

这正是"免数据"的含义：GPU 的 scale 不是编译期常量，这条规则在那里无处定义。

另外给出一阶预测 ``Delta ln P_fa ~= z^2 * eps_sigma`` 与 E5 的判定工具。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .decision import IntQuantSpec, lambda_int as _lambda_int


@dataclass
class TransportSpec:
    """一次量化配置的两个关键步长（判决只用到 sigma 与分数两路）。"""

    delta_s: float
    delta_sigma: float
    shift_bits: int = 16
    name: str = ""

    def check(self):
        for n in ("delta_s", "delta_sigma"):
            v = float(getattr(self, n))
            if not np.isfinite(v) or v <= 0:
                raise ValueError(f"{n} 必须是正有限数, 得到 {v}")
        return self

    @property
    def shift(self) -> int:
        return int(self.shift_bits)

    def to_int_spec(self) -> IntQuantSpec:
        """投影成判决层需要的三路 spec（ds == dmu 是 v3 M1 的无零模型情形）。"""
        return IntQuantSpec(delta_s=self.delta_s, delta_mu=self.delta_s,
                            delta_sigma=self.delta_sigma,
                            shift_bits=self.shift_bits)

    @classmethod
    def from_scale_table(cls, table, score_name: str, sigma_name: str,
                         shift_bits: int = 16) -> "TransportSpec":
        e_s = table.require(score_name)
        e_g = table.require(sigma_name)
        return cls(delta_s=e_s.delta, delta_sigma=e_g.delta,
                   shift_bits=shift_bits, name=f"{score_name}|{sigma_name}")


# ------------------------------------------------------------------ 换算
def lambda_int_from_float(lam: float, spec: TransportSpec) -> int:
    """浮点 lambda（z 域）-> 该配置下的整数阈值。"""
    spec.check()
    return int(_lambda_int(spec.to_int_spec(), float(lam), spec.shift))


def lambda_float_from_int(lam_q: int, spec: TransportSpec) -> float:
    """整数阈值 -> 浮点 lambda（诊断用）。"""
    spec.check()
    return float(lam_q) * spec.delta_s / (spec.delta_sigma * (2 ** spec.shift))


def transport_lambda_int(lambda_int_old: int, old: TransportSpec,
                         new: TransportSpec) -> int:
    """旧整数阈值 -> 新整数阈值（免数据，纯 scale 比值）。"""
    old.check()
    new.check()
    if int(old.shift) != int(new.shift):
        raise ValueError(f"移位不一致: {old.shift} vs {new.shift}（先统一 shift_bits）")
    ratio = (new.delta_sigma / old.delta_sigma) * (old.delta_s / new.delta_s)
    return int(round(float(lambda_int_old) * ratio))


def transport_lambda_float(lam_old: float, old: TransportSpec,
                           new: TransportSpec) -> float:
    """浮点 lambda 的搬运（等价于直接用新 scale 重算，便于自检）。"""
    return lambda_float_from_int(transport_lambda_int(
        lambda_int_from_float(lam_old, old), old, new), new)


# ------------------------------------------------------------------ 一阶预言
def sigma_eps(delta_sigma: float, sigma: float) -> float:
    """epsilon_sigma = Delta_sigma / (2 sigma)。"""
    sigma = float(sigma)
    if sigma <= 0:
        return float("inf")
    return float(delta_sigma) / (2.0 * sigma)


def predicted_dln_pfa(z: float, eps_sigma: float) -> float:
    """一阶近似：Delta ln P_fa ~= z^2 * eps_sigma（02 §2）。"""
    return float(z) ** 2 * float(eps_sigma)


def drift_log10(pfa: float, alpha: float) -> float:
    """|log10(P_fa / alpha)|：契约是否还守。"""
    if pfa <= 0:
        return float("inf")
    return float(abs(np.log10(float(pfa) / float(alpha))))


# ------------------------------------------------------------------ E5 判定
def compare_pfa(pfa_transport: float, pfa_recal: float, pfa_no_convert: float,
                alpha: float, factor: float = 1.5) -> dict:
    """04 §8 验收：搬运后的偏离 <= 重标定上界的 1.5 倍。"""
    d_t = drift_log10(pfa_transport, alpha)
    d_r = drift_log10(pfa_recal, alpha)
    d_n = drift_log10(pfa_no_convert, alpha)
    return {
        "pfa_transport": float(pfa_transport),
        "pfa_recal": float(pfa_recal),
        "pfa_no_convert": float(pfa_no_convert),
        "alpha": float(alpha),
        "drift_transport": d_t,
        "drift_recal": d_r,
        "drift_no_convert": d_n,
        "factor": float(factor),
        "pass": bool(np.isfinite(d_t) and np.isfinite(d_r)
                     and d_t <= factor * max(d_r, 1e-12)),
        "better_than_no_convert": bool(d_t <= d_n),
    }


def e5_row(setting: str, pfa_transport: float, pfa_recal: float,
           pfa_no_convert: float, alpha: float, n_images: int) -> dict:
    """04 §8 空表的一行。"""
    c = compare_pfa(pfa_transport, pfa_recal, pfa_no_convert, alpha)
    return {
        "setting": setting,
        "pfa_transport": c["pfa_transport"],
        "pfa_recal": c["pfa_recal"],
        "pfa_no_convert": c["pfa_no_convert"],
        "alpha": float(alpha),
        "n_images": int(n_images),
        "drift_transport": c["drift_transport"],
        "pass": c["pass"],
        "better_than_no_convert": c["better_than_no_convert"],
    }


# ------------------------------------------------------------------ 自检
def _selfcheck():
    """两条换算路径必须一致（transport 与直接重算）。"""
    old = TransportSpec(1.0 / 128, 1.0 / 512, 16)
    new = TransportSpec(1.0 / 100, 1.0 / 400, 16)
    lam = 3.09
    a = transport_lambda_int(lambda_int_from_float(lam, old), old, new)
    b = lambda_int_from_float(lam, new)
    assert abs(a - b) <= 1, (a, b)
    return a, b


if __name__ == "__main__":  # pragma: no cover
    print(_selfcheck())
