"""分数场的落盘格式与读取（.npz）。

一个场景一个文件，字段固定为：

================  ====================================================
``prob``          (N,) float32  逐网格 max_c sigmoid(cls)，即论文的 s(u)
``logit``         (N,) float32  sigmoid 之前的原始值（用于诊断饱和）
``centers``       (N,2) float32 网格中心在 letterbox 输入坐标系下的像素坐标
``head_sizes``    (H,2) int32   三个尺度的 (h, w)，用于还原稠密图
``s_q``           (N,) int32    整数域原始码（cv3 链路是累加器网格码，范围远超
                                ±127；早期 int8 链路文件读出仍为 int8，按值兼容）
``delta``         () float32    输出层量化步长（没有则缺省）
``zero_point``    () int32
``meta``          () str        JSON：imgsz / iw / ih / ratio / pad / impl / contrast
================  ====================================================

``s_q`` + ``delta`` 一起落盘是刻意的：v3 的叙事是"整数域"，所以**原值**必须保住；
``prob`` 只是它的浮点视图，方便与 FP32 链路对齐比较。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class ScoreField:
    prob: np.ndarray
    logit: np.ndarray | None = None
    centers: np.ndarray | None = None
    head_sizes: np.ndarray | None = None
    s_q: np.ndarray | None = None
    delta: float | None = None
    zero_point: int = 0
    meta: dict = field(default_factory=dict)
    stem: str = ""

    @property
    def n(self) -> int:
        return int(self.prob.size)

    @property
    def is_int8(self) -> bool:
        return self.s_q is not None

    def z(self) -> np.ndarray:
        """默认"无零模型"的判决统计量：z = s（μ=0, σ=1）。"""
        return self.prob.astype(np.float64)

    def grid(self, head: int = 0):
        """把某一尺度的分数还原成 (h, w) 稠密图（顺序与 dump 时一致）。"""
        if self.head_sizes is None:
            raise ValueError("缺少 head_sizes，无法还原稠密图")
        sizes = np.asarray(self.head_sizes).reshape(-1, 2)
        off = 0
        for i, (h, w) in enumerate(sizes):
            cnt = int(h) * int(w)
            if i == head:
                return self.prob[off:off + cnt].reshape(int(h), int(w))
            off += cnt
        raise IndexError(f"head 索引越界: {head} / {len(sizes)}")


def save_score_npz(path, field: ScoreField):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "prob": np.asarray(field.prob, dtype=np.float32),
        "meta": np.array(json.dumps(field.meta, ensure_ascii=False)),
    }
    if field.logit is not None:
        payload["logit"] = np.asarray(field.logit, dtype=np.float32)
    if field.centers is not None:
        payload["centers"] = np.asarray(field.centers, dtype=np.float32)
    if field.head_sizes is not None:
        payload["head_sizes"] = np.asarray(field.head_sizes, dtype=np.int32)
    if field.s_q is not None:
        # cv3 网格码是 int32 累加器值（±万级），强转 int8 会毁掉格点
        payload["s_q"] = np.asarray(field.s_q, dtype=np.int32)
    if field.delta is not None:
        payload["delta"] = np.array(float(field.delta), dtype=np.float32)
    payload["zero_point"] = np.array(int(field.zero_point), dtype=np.int32)
    np.savez_compressed(path, **payload)
    return path


def load_score_npz(path) -> ScoreField:
    path = Path(path)
    with np.load(str(path), allow_pickle=True) as z:
        meta = json.loads(str(np.asarray(z["meta"]).ravel()[0])) if "meta" in z else {}
        return ScoreField(
            prob=np.asarray(z["prob"]).ravel().astype(np.float32),
            logit=np.asarray(z["logit"]).ravel().astype(np.float32) if "logit" in z else None,
            centers=np.asarray(z["centers"], dtype=np.float32) if "centers" in z else None,
            head_sizes=np.asarray(z["head_sizes"], dtype=np.int32) if "head_sizes" in z else None,
            s_q=(np.asarray(z["s_q"]).ravel() if "s_q" in z else None),  # 保持落盘 dtype
            delta=float(np.asarray(z["delta"]).ravel()[0]) if "delta" in z else None,
            zero_point=int(np.asarray(z["zero_point"]).ravel()[0])
            if "zero_point" in z else 0,
            meta=meta,
            stem=path.stem,
        )


def load_dir(root, pattern: str = "*.npz") -> dict:
    """读一个目录下的所有分数场 -> {stem: ScoreField}。"""
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(root)
    out = {}
    for p in sorted(root.glob(pattern)):
        try:
            out[p.stem] = load_score_npz(p)
        except Exception as exc:  # noqa: BLE001
            print(f"[scores] 跳过 {p.name}: {exc}")
    if not out:
        raise FileNotFoundError(f"{root} 下没有可读的 {pattern}")
    return out


def field_to_int8(prob, delta: float, zero_point: int = 0):
    """把浮点分数场按给定 delta 量化成 int8（离线仿真用）。"""
    from .qint import quantize
    return quantize(prob, delta, zero_point, bits=8)


def field_from_int8(s_q, delta: float, zero_point: int = 0) -> np.ndarray:
    from .qint import dequantize
    return dequantize(s_q, delta, zero_point).astype(np.float32)
