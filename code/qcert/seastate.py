"""海况分桶（对应 paper/04 §1.2，跑数前固定，检测器独立）。

三条特征：
1. 全图杂波二阶统计 -> 变异系数 ``cv = std/mean``；
2. Weibull 形状参数 ``c`` 的矩估计（``cv(c)`` 单调，二分求逆）；
3. 局部对比度中位数：分块 ``std/mean`` 的中位数，抑制单点强目标的影响。

三聚类（numpy k-means）后按"粗糙度"排序映射到 平静 / 中等 / 粗糙，
并支持把结果落盘（JSON），保证 Fig.1 的横轴在实验前固定、不随分数变化。
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

LABELS = ("calm", "medium", "rough")


# ------------------------------------------------------------------ 图像特征
def to_gray(img) -> np.ndarray:
    """HWC/2D -> float32 灰度（0..255 或原浮点，不做归一化）。"""
    a = np.asarray(img)
    if a.ndim == 3:
        a = a[..., :3].astype(np.float64).mean(axis=2)
    return a.astype(np.float64)


def local_contrast(gray, tile: int = 32) -> float:
    """分块 cv 的中位数：稳健的局部对比度。"""
    g = to_gray(gray)
    h, w = g.shape
    vals = []
    for y in range(0, h - tile + 1, tile):
        for x in range(0, w - tile + 1, tile):
            t = g[y:y + tile, x:x + tile]
            m = t.mean()
            if m > 1e-9:
                vals.append(t.std() / m)
    if not vals:
        return float(g.std() / max(g.mean(), 1e-9))
    return float(np.median(vals))


def weibull_shape(x) -> float:
    """Weibull 形状参数 c 的矩估计（由 cv 反解，二分法，不依赖 scipy）。

    cv(c) = sqrt(Gamma(1+2/c) / Gamma(1+1/c)^2 - 1)，关于 c 单调递减。
    """
    v = np.asarray(x, dtype=np.float64).ravel()
    cv = float(v.std() / max(abs(v.mean()), 1e-12))
    if not np.isfinite(cv) or cv <= 1e-6:
        return float("inf")

    def cv_of(c):
        try:
            g1 = math.gamma(1 + 1 / c)
            g2 = math.gamma(1 + 2 / c)
            return math.sqrt(max(g2 / (g1 * g1) - 1.0, 0.0))
        except (OverflowError, ValueError):   # c 太小时 Gamma 溢出 -> 视为无穷大 cv
            return float("inf")

    lo, hi = 0.05, 50.0
    if cv >= cv_of(lo):
        return lo
    if cv <= cv_of(hi):
        return hi
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if cv_of(mid) > cv:      # c 太小 -> cv 太大
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


def gamma_shape_moments(x) -> float:
    """Gamma/K 强度分布的形状参数矩估计：nu = mean^2 / var。"""
    v = np.asarray(x, dtype=np.float64).ravel()
    var = float(v.var())
    if var <= 0:
        return float("inf")
    return float(v.mean() ** 2 / var)


def clutter_features(img, tile: int = 32) -> dict:
    """单张图（或分数场）的杂波特征。"""
    g = to_gray(img)
    v = g.ravel()
    mean = float(v.mean())
    std = float(v.std())
    return {
        "mean": mean,
        "std": std,
        "cv": float(std / max(abs(mean), 1e-12)),
        "weibull_c": weibull_shape(v),
        "gamma_shape": gamma_shape_moments(v),
        "contrast": local_contrast(g, tile),
    }


def image_features(path, tile: int = 32) -> dict:
    from PIL import Image

    with Image.open(Path(path)) as im:
        im.load()
        arr = np.asarray(im.convert("L"), dtype=np.float64)
    return clutter_features(arr, tile)


# ------------------------------------------------------------------ 聚类
def kmeans(X, k: int = 3, n_iter: int = 100, seed: int = 0, tol: float = 1e-6):
    """最简 k-means（numpy），空簇用最远点重新播种。"""
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    if n < k:
        raise ValueError(f"样本数 {n} < 聚类数 {k}")
    rng = np.random.default_rng(seed)
    centers = X[rng.choice(n, size=k, replace=False)]
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(int(n_iter)):
        d = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = d.argmin(axis=1)
        new = centers.copy()
        for j in range(k):
            m = labels == j
            if m.any():
                new[j] = X[m].mean(axis=0)
            else:
                far = d.min(axis=1).argmax()
                new[j] = X[far]
        if np.allclose(new, centers, atol=tol):
            centers = new
            break
        centers = new
    return centers, labels


FEATURE_KEYS = ("cv", "weibull_c", "contrast")


def feat_matrix(feats) -> np.ndarray:
    """三项特征 -> 标准化前的对数变换矩阵。"""
    X = []
    for f in feats:
        X.append([
            np.log(max(f["cv"], 1e-9)),
            np.log(max(f["weibull_c"], 1e-9)),
            np.log(max(f["contrast"], 1e-9)),
        ])
    return np.asarray(X, dtype=np.float64)


def roughness_score(feats) -> np.ndarray:
    """粗糙度：cv 大、对比度大、Weibull c 小 => 粗糙。

    用中位数 / MAD 稳健标准化后线性组合。
    """
    X = feat_matrix(feats)
    med = np.median(X, axis=0)
    mad = np.median(np.abs(X - med), axis=0)
    mad = np.where(mad > 1e-9, mad, 1.0)
    Z = (X - med) / mad
    return Z[:, 0] - Z[:, 1] + Z[:, 2]


@dataclass
class SeaStateBucketer:
    """三聚类 + 稳健标准化，可保存/加载以固定分桶。"""

    mean: np.ndarray | None = None
    std: np.ndarray | None = None
    centers: np.ndarray | None = None
    order: list = field(default_factory=list)   # cluster id -> label index
    feature_keys: tuple = FEATURE_KEYS

    def _standardize(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if self.mean is None or self.std is None:
            raise RuntimeError("尚未 fit")
        return (X - self.mean) / np.where(self.std > 1e-9, self.std, 1.0)

    def fit(self, feats) -> "SeaStateBucketer":
        X = feat_matrix(feats)
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        Z = self._standardize(X)
        self.centers, labels = kmeans(Z, k=3)
        # 按粗糙度给簇排序
        rough = roughness_score(feats)
        mean_rough = np.array([rough[labels == j].mean() if (labels == j).any()
                               else -np.inf for j in range(len(self.centers))])
        self.order = list(np.argsort(mean_rough))       # 低粗糙度在前
        return self

    def predict(self, feats) -> list:
        Z = self._standardize(feat_matrix(feats))
        d = ((Z[:, None, :] - self.centers[None, :, :]) ** 2).sum(axis=2)
        cluster = d.argmin(axis=1)
        rank = {c: i for i, c in enumerate(self.order)}
        return [LABELS[min(rank[int(c)], 2)] for c in cluster]

    def to_dict(self) -> dict:
        return {
            "mean": np.asarray(self.mean).tolist() if self.mean is not None else None,
            "std": np.asarray(self.std).tolist() if self.std is not None else None,
            "centers": np.asarray(self.centers).tolist() if self.centers is not None else None,
            "order": [int(x) for x in self.order],
            "feature_keys": list(self.feature_keys),
            "labels": list(LABELS),
        }

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2,
                                         ensure_ascii=False), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path) -> "SeaStateBucketer":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        b = cls()
        b.mean = np.asarray(d["mean"], dtype=np.float64) if d.get("mean") else None
        b.std = np.asarray(d["std"], dtype=np.float64) if d.get("std") else None
        b.centers = (np.asarray(d["centers"], dtype=np.float64)
                     if d.get("centers") else None)
        b.order = [int(x) for x in d.get("order", [])]
        b.feature_keys = tuple(d.get("feature_keys", FEATURE_KEYS))
        return b


def assign(feats, bucketer: SeaStateBucketer) -> list:
    """给一组场景特征打桶标签，返回 [{"stem":..., "bucket":...}]。"""
    buckets = bucketer.predict(feats)
    out = []
    for f, b in zip(feats, buckets):
        d = dict(f)
        d["bucket"] = b
        out.append(d)
    return out


def summarize(rows) -> dict:
    """按桶汇总场景数。"""
    out = {k: 0 for k in LABELS}
    for r in rows:
        out[r["bucket"]] = out.get(r["bucket"], 0) + 1
    return out
