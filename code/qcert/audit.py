"""饱和/格点占用审计（模块 5，对应 paper/04 §5 的 E2）。

数值取证结论（probe_grid）：cv3 终卷积输出是 **int32 累加器网格**（q∈±万级，
严格在 Δ 格点上，sigmoid 前无 int8 重量化），永远不触顶——"分数场 ±127 饱和"
的假设被推翻。饱和的真实发生位置是 **neck 的 int8 特征层**（每个 Conv 前的
AscendQuant 都是 int8），分数场只是这些可能已饱和特征的线性读出。

三件事：
1. **int8 层**（neck 特征）逐层饱和率 ``sat_l``，拆上/下饱和（``layer_saturation``）；
   **int32 网格**（cv3 分数）改报格点占用：uniq 码数、范围利用率（``grid_occupancy``）；
2. 分数场最高 ``frac``（默认 1%）内的**并列数**——同一整数值出现了多少次。
   int8 特征饱和会把邻近位置压成相同特征向量 → 分数码并列，这是"排序塌陷"
   的直接证据（对 int8/int32 码同样成立）；
3. 把饱和率与 ``R = P_fa(INT8)/P_fa(FP32)`` 做 Spearman 相关，回答 E2 的
   Go 判据："压测集上饱和率随对比度单调上升，且与 R 正相关"。

全部是纯 numpy，可在 x86 与板端 Python 3.10 上运行。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .qint import qmin_qmax


# ------------------------------------------------------------------ 单层统计
def saturation_mask(q, bits: int = 8) -> np.ndarray:
    """返回饱和位置掩膜（撞到 qmax 或 qmin）。"""
    q = np.asarray(q)
    qmin, qmax = qmin_qmax(bits)
    return (q >= qmax) | (q <= qmin)


def layer_saturation(q, bits: int = 8) -> dict:
    """单层饱和统计。

    Returns
    -------
    dict: n / sat / sat_up / sat_down / qmin / qmax / p01 / p99
    """
    q = np.asarray(q).ravel()
    qmin, qmax = qmin_qmax(bits)
    n = int(q.size)
    if n == 0:
        return {"n": 0, "sat": 0.0, "sat_up": 0.0, "sat_down": 0.0,
                "qmin": qmin, "qmax": qmax, "p01": float("nan"),
                "p99": float("nan")}
    qf = q.astype(np.int64)
    up = float(np.mean(qf >= qmax))
    down = float(np.mean(qf <= qmin))
    return {
        "n": n,
        "sat": float(up + down),
        "sat_up": up,
        "sat_down": down,
        "qmin": qmin,
        "qmax": qmax,
        "p01": float(np.percentile(qf, 1)),
        "p99": float(np.percentile(qf, 99)),
    }


def grid_occupancy(q, accum_bits: int = 32) -> dict:
    """累加器网格（无固定码界）的占用审计——cv3 分类头的 int32 网格码。

    与 ``layer_saturation`` 互补：后者面向有码界的 int8 层（neck 特征），
    本函数面向 sigmoid 前的终卷积输出——int32 累加器直接 dequant，永不触顶，
    "±127 饱和率"失去意义，改为报告格点占用（uniq 码数）与范围利用率
    （max|q| 相对累加器码界，恒远小于 1，即"不饱和"的量化表述）。

    返回键与 ``layer_saturation`` 兼容（sat/sat_up/sat_down 相对累加器码界，
    理论上恒 0），另加 uniq / uniq_frac / range_util；qmin/qmax 是**观测**范围。
    """
    q = np.asarray(q).ravel().astype(np.int64)
    n = int(q.size)
    lo, hi = -(2 ** (accum_bits - 1)), 2 ** (accum_bits - 1) - 1
    if n == 0:
        return {"n": 0, "sat": 0.0, "sat_up": 0.0, "sat_down": 0.0,
                "qmin": 0, "qmax": 0, "p01": float("nan"), "p99": float("nan"),
                "uniq": 0, "uniq_frac": 0.0, "range_util": 0.0}
    uniq = int(np.unique(q).size)
    return {
        "n": n,
        "sat": float(np.mean((q >= hi) | (q <= lo))),
        "sat_up": float(np.mean(q >= hi)),
        "sat_down": float(np.mean(q <= lo)),
        "qmin": int(q.min()),
        "qmax": int(q.max()),
        "p01": float(np.percentile(q, 1)),
        "p99": float(np.percentile(q, 99)),
        "uniq": uniq,
        "uniq_frac": float(uniq / n),
        "range_util": float(max(int(q.max()), -int(q.min())) / hi),
    }


def tie_stats(codes, select=None, frac: float = 0.01) -> dict:
    """最高 ``frac`` 的并列统计。

    Parameters
    ----------
    codes : array_like
        整数码（同一整数值即"并列"）。
    select : array_like or None
        float 分数，用于挑"最高 frac"。None 时直接按 codes 排序挑。
    frac : float
        取最高的比例，默认 0.01（04 §5）。
    """
    c = np.asarray(codes).ravel().astype(np.int64)
    n = int(c.size)
    if n == 0:
        return {"n_sel": 0, "n_unique": 0, "max_tie": 0, "mean_tie": 0.0,
                "tie_ratio": 0.0, "top_code": 0, "top_count": 0}
    k = max(1, int(round(n * float(frac))))
    if select is None:
        sel = np.sort(c)[-k:]
    else:
        s = np.asarray(select).ravel()
        if s.size != n:
            raise ValueError(f"select 长度 {s.size} != codes 长度 {n}")
        idx = np.argpartition(-s, k - 1)[:k]
        sel = c[idx]
    uniq, cnt = np.unique(sel, return_counts=True)
    top = int(cnt.max())
    return {
        "n_sel": int(sel.size),
        "n_unique": int(uniq.size),
        "max_tie": top,
        "mean_tie": float(sel.size / max(uniq.size, 1)),
        "tie_ratio": float(1.0 - uniq.size / max(sel.size, 1)),
        "top_code": int(uniq[int(cnt.argmax())]),
        "top_count": top,
    }


def score_field_tie_stats(prob, s_q, frac: float = 0.01) -> dict:
    """分数场的并列统计：按 float 分数挑最高 frac，再看整数码的并列。"""
    if s_q is None:
        raise ValueError("缺少 s_q：饱和/并列审计需要整数域原始码")
    return tie_stats(s_q, select=prob, frac=frac)


# ------------------------------------------------------------------ 汇总器
@dataclass
class AuditRow:
    layer: str
    sea: str = "all"
    n: int = 0
    sat: float = 0.0
    sat_up: float = 0.0
    sat_down: float = 0.0
    tie_ratio: float = float("nan")
    max_tie: int = 0
    uniq_frac: float = float("nan")
    range_util: float = float("nan")
    contrast: float = 1.0
    source: str = "layer"

    def to_dict(self) -> dict:
        return {
            "layer": self.layer,
            "sea": self.sea,
            "contrast": self.contrast,
            "n": self.n,
            "sat": self.sat,
            "sat_up": self.sat_up,
            "sat_down": self.sat_down,
            "tie_ratio": self.tie_ratio,
            "max_tie": self.max_tie,
            "uniq_frac": self.uniq_frac,
            "range_util": self.range_util,
            "source": self.source,
        }


@dataclass
class SaturationAuditor:
    """逐层/逐海况累积饱和与并列统计，最后落成 04 §5 的空表行。"""

    bits: int = 8
    frac: float = 0.01
    rows: list = field(default_factory=list)

    def add_layer(self, name, q, sea: str = "all", contrast: float = 1.0,
                  prob=None):
        st = layer_saturation(q, self.bits)
        tie = score_field_tie_stats(prob, q, self.frac) if prob is not None else {}
        self.rows.append(AuditRow(
            layer=name, sea=sea, n=st["n"], sat=st["sat"], sat_up=st["sat_up"],
            sat_down=st["sat_down"], tie_ratio=tie.get("tie_ratio", float("nan")),
            max_tie=tie.get("max_tie", 0), contrast=contrast, source="layer"))
        return st

    def add_scores(self, name, prob, s_q, sea: str = "all", contrast: float = 1.0):
        if s_q is None:
            raise ValueError(f"{name}: 缺少 s_q，不能审计")
        q = np.asarray(s_q)
        # int8 码 → 有码界饱和审计；更宽的码（cv3 int32 网格）→ 格点占用审计
        st = (layer_saturation(q, self.bits) if q.dtype == np.int8
              else grid_occupancy(q))
        tie = score_field_tie_stats(prob, q, self.frac)
        self.rows.append(AuditRow(
            layer=name, sea=sea, n=st["n"], sat=st["sat"], sat_up=st["sat_up"],
            sat_down=st["sat_down"], tie_ratio=tie["tie_ratio"],
            max_tie=tie["max_tie"], uniq_frac=st.get("uniq_frac", float("nan")),
            range_util=st.get("range_util", float("nan")),
            contrast=contrast, source="score"))
        return st

    def add_from_field(self, field, name: str = "score", sea: str = "all",
                       contrast: float = 1.0):
        return self.add_scores(name, field.prob, field.s_q, sea, contrast)

    def table(self) -> list:
        return [r.to_dict() for r in self.rows]

    def summary(self) -> str:
        lines = [f"{'layer':<28s}{'sea':<8s}{'sat':>8s}{'sat_up':>8s}"
                 f"{'tie_ratio':>10s}{'max_tie':>8s}{'uniq':>8s}"]
        for r in self.rows:
            lines.append(f"{r.layer:<28s}{r.sea:<8s}{r.sat:>8.4f}{r.sat_up:>8.4f}"
                         f"{r.tie_ratio:>10.4f}{r.max_tie:>8d}{r.uniq_frac:>8.4f}")
        return "\n".join(lines)


# ------------------------------------------------------------------ E2 判据
def monotonic_increase(values) -> bool:
    """严格单调不降（允许相等）。"""
    v = np.asarray(values, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    return bool(v.size >= 2 and np.all(np.diff(v) >= 0))


def correlation_with_r(sat_by_group: dict, r_by_group: dict):
    """饱和率与 R 的 Spearman 相关（E2 Go 判据：ρ > 0.5）。

    两个 dict 的键必须对应（如对比度档 ×1/×2/×3）。
    """
    from .pfa_eval import spearman

    keys = [k for k in sat_by_group if k in r_by_group]
    if len(keys) < 2:
        return {"n": len(keys), "rho": float("nan"),
                "monotonic_sat": monotonic_increase([sat_by_group[k] for k in keys]),
                "pass": False}
    sat = [sat_by_group[k] for k in keys]
    r = [r_by_group[k] for k in keys]
    rho = spearman(sat, r)
    return {
        "n": len(keys),
        "rho": float(rho),
        "monotonic_sat": monotonic_increase(sat),
        "pass": bool(np.isfinite(rho) and rho > 0.5
                     and monotonic_increase(sat)),
    }


def e2_row(contrast: float, n_scenes: int, score_sat: float, neck_sat: float,
           max_tie: int, r: float, miss_rate: float) -> dict:
    """04 §5 空表的一行。"""
    return {
        "contrast": float(contrast),
        "n_scenes": int(n_scenes),
        "score_sat": float(score_sat),
        "neck_sat": float(neck_sat),
        "top1pct_max_tie": int(max_tie),
        "R": float(r),
        "miss_rate": float(miss_rate),
    }
