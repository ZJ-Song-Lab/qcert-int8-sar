"""热稳态长跑（模块 8，对应 paper/04 §9 的 E6）。

连续跑 ``duration_min`` 分钟，每 ``window_min`` 分钟落一个窗口：
温度 / 频率 / 功率 / P_fa / P99 / T_model，最后判断
"窗口间 P_fa 漂移 |log10| < 0.15、P99 漂移 < 15%"。

设计要点：
- ``frame_fn()`` 返回一帧的指标 dict（至少 ``t_model_ms``，可选 ``pfa``），
  这样离线（x86）与板上（pyACL）都能复用同一个 runner；
- ``npu-smi`` 采样用独立线程，读不到就留空，不中断实验；
- 本模块顶层不 import pyACL / torch。
"""
from __future__ import annotations

import re
import subprocess
import threading
import time
from collections import deque

import numpy as np

SMI_CMD = ["npu-smi", "info"]


# ------------------------------------------------------------------ npu-smi 解析
def parse_npu_smi(text: str) -> dict:
    """从 ``npu-smi info`` 文本里抠可用字段，抠不到就不给键。"""
    out = {}
    if not text:
        return out
    m = re.findall(r"\|\s*(\d+(?:\.\d+)?)\s+\d+\s+\d+\s*/\s*\d+\s*\|", text)
    if m:
        out["power_w"] = float(m[0])
    m = re.findall(r"(\d{3,4})\s*MHz", text)
    if m:
        out["freq_mhz"] = float(m[0])
    m = re.findall(r"(?:Temperature|Temp)\s*[:=]?\s*(\d{1,3})", text, re.I)
    if not m:
        m = re.findall(r"(\d{1,3})\s*(?:C|℃)\b", text)
    if m:
        out["temp_c"] = float(m[0])
    m = re.findall(r"(\d+(?:\.\d+)?)\s*%", text)
    if m:
        out["util_pct"] = float(m[0])
    return out


class SmiSampler(threading.Thread):
    """后台按固定间隔采样 npu-smi；命令不存在时静默跳过。"""

    def __init__(self, interval: float = 1.0, cmd=None):
        super().__init__(daemon=True)
        self.interval = float(interval)
        self.cmd = list(cmd or SMI_CMD)
        self.samples = []          # list[(t, dict)]
        self._stop = threading.Event()
        self._t0 = time.perf_counter()

    def poll_once(self) -> dict:
        try:
            txt = subprocess.run(self.cmd, capture_output=True, text=True,
                                 timeout=3).stdout
        except Exception:
            return {}
        snap = parse_npu_smi(txt)
        if snap:
            self.samples.append((time.perf_counter() - self._t0, snap))
        return snap

    def run(self):
        while not self._stop.is_set():
            self.poll_once()
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()

    def window(self, t0: float, t1: float) -> dict:
        """取 [t0, t1) 区间内的均值。"""
        vals = {}
        for t, s in self.samples:
            if t0 <= t < t1:
                for k, v in s.items():
                    vals.setdefault(k, []).append(v)
        return {k: float(np.mean(v)) for k, v in vals.items()}


# ------------------------------------------------------------------ 窗口统计
def _summ(values, q=None):
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)],
                   dtype=np.float64)
    if v.size == 0:
        return float("nan")
    return float(np.percentile(v, q)) if q is not None else float(v.mean())


class ThermalRunner:
    """按窗口循环执行 ``frame_fn``，返回 04 §9 的空表行。

    两个与"帧数"相关的参数，解决"瞬时帧源在一个窗口内刷出上千万帧"的问题：

    - ``frames_per_window``（默认 200，见 04 §9）：每窗**只保留最后 N 帧**做统计
      （``deque(maxlen=N)`` 环形缓冲）。负载不中断，但统计量与内存都有界；
    - ``pace_s``：帧间最小间隔（秒）。离线自检时设成模拟时延，避免空转。
    """

    def __init__(self, frame_fn, duration_min: float = 40.0,
                 window_min: float = 5.0, sampler=None,
                 clock=time.perf_counter, sleep=time.sleep, verbose=True,
                 frames_per_window: int | None = 200,
                 pace_s: float | None = None):
        self.frame_fn = frame_fn
        self.duration_min = float(duration_min)
        self.window_min = float(window_min)
        self.sampler = sampler
        self.clock = clock
        self.sleep = sleep
        self.verbose = verbose
        self.frames_per_window = (None if not frames_per_window
                                  else int(frames_per_window))
        self.pace_s = (float(pace_s) if pace_s and pace_s > 0 else None)

    def run(self) -> list:
        rows = []
        t_start = self.clock()
        if self.sampler is not None:
            self.sampler._t0 = t_start
            if not self.sampler.is_alive():
                self.sampler.start()
        n_win = max(1, int(round(self.duration_min / self.window_min)))
        for w in range(n_win):
            w0 = t_start + w * self.window_min * 60.0
            w1 = t_start + (w + 1) * self.window_min * 60.0
            lat = deque(maxlen=self.frames_per_window)
            pfa = deque(maxlen=self.frames_per_window)
            n_seen = 0
            t_last = None
            while True:
                now = self.clock()
                if now >= w1:
                    break
                if self.pace_s is not None:
                    due = now if t_last is None else t_last + self.pace_s
                    if due > now:
                        self.sleep(min(due, w1) - now)
                        now = self.clock()
                        if now >= w1:
                            break
                t_last = now
                rec = self.frame_fn()
                n_seen += 1
                if rec is None:
                    self.sleep(0.001)
                    continue
                if "t_model_ms" in rec:
                    lat.append(float(rec["t_model_ms"]))
                if "pfa" in rec and np.isfinite(rec["pfa"]):
                    pfa.append(float(rec["pfa"]))
            smi = self.sampler.window(w0, w1) if self.sampler is not None else {}
            row = {
                "window": f"{w * self.window_min:.0f}-{(w + 1) * self.window_min:.0f} min",
                "t0_min": float(w * self.window_min),
                "n_frames": len(lat),
                "n_seen": n_seen,
                "t_model_ms": _summ(lat),
                "p99_ms": _summ(lat, 99),
                "pfa": _summ(pfa),
                "temp_c": smi.get("temp_c", float("nan")),
                "freq_mhz": smi.get("freq_mhz", float("nan")),
                "power_w": smi.get("power_w", float("nan")),
                "util_pct": smi.get("util_pct", float("nan")),
            }
            rows.append(row)
            if self.verbose:
                print(f"[thermal] {row['window']}: n={row['n_frames']} "
                      f"T={row['t_model_ms']:.2f}ms P99={row['p99_ms']:.2f}ms "
                      f"P_fa={row['pfa']:.3e} temp={row['temp_c']}")
        if self.sampler is not None:
            self.sampler.stop()
        return rows


def thermal_run(frame_fn, duration_min: float = 40.0, window_min: float = 5.0,
                with_smi: bool = True, **kwargs) -> list:
    sampler = SmiSampler() if with_smi else None
    return ThermalRunner(frame_fn, duration_min, window_min, sampler, **kwargs).run()


def window_drift(rows) -> dict:
    """04 §9 验收：P_fa 窗口漂移 |log10| < 0.15；P99 漂移 < 15%。

    空窗（``pfa=nan``，例如该窗内没有跑到一帧）不再污染整张表：
    只丢掉非有限/非正的窗口，并显式报告有效窗口数；有效窗口 < 2 时给出
    ``insufficient=True``，由调用方判成 ``INSUFFICIENT`` 而不是 ``DRIFT``。
    """
    pfa = np.array([r.get("pfa", np.nan) for r in rows], dtype=np.float64)
    p99 = np.array([r.get("p99_ms", np.nan) for r in rows], dtype=np.float64)
    pfa_ok = pfa[np.isfinite(pfa) & (pfa > 0)]
    p99_ok = p99[np.isfinite(p99) & (p99 > 0)]
    d_pfa = (float(np.max(np.abs(np.log10(pfa_ok / pfa_ok[0]))))
             if pfa_ok.size >= 2 else float("nan"))
    d_p99 = (float(np.max(np.abs(p99_ok / p99_ok[0] - 1.0)))
             if p99_ok.size >= 2 else float("nan"))
    return {
        "pfa_drift_log10": d_pfa,
        "p99_drift_rel": d_p99,
        "n_pfa_windows": int(pfa_ok.size),
        "n_p99_windows": int(p99_ok.size),
        "n_windows": int(len(rows)),
        "insufficient": bool(pfa_ok.size < 2 or p99_ok.size < 2),
        "pass_pfa": bool(np.isfinite(d_pfa) and d_pfa < 0.15),
        "pass_p99": bool(np.isfinite(d_p99) and d_p99 < 0.15),
    }


def thermal_table(rows) -> list:
    """整理成 04 §9 的空表结构。"""
    keys = ("window", "temp_c", "freq_mhz", "power_w", "pfa", "p99_ms",
            "t_model_ms", "n_frames")
    return [{k: r.get(k) for k in keys} for r in rows]


# ------------------------------------------------------------------ 离线自检
def synthetic_frame_fn(latency_ms: float = 8.0, jitter_ms: float = 1.0,
                       pfa: float = 1e-3, pfa_drift: float = 0.0,
                       seed: int = 0):
    """造一个可复现的假帧源，用于 E6 脚本的 --dry-run 与单测。"""
    rng = np.random.default_rng(seed)
    state = {"i": 0}

    def fn():
        state["i"] += 1
        drift = 1.0 + pfa_drift * (state["i"] / 1000.0)
        return {
            "t_model_ms": float(latency_ms + rng.normal(0, jitter_ms)),
            "pfa": float(pfa * drift),
        }

    return fn
