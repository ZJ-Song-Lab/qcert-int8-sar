"""CA-CFAR 基线（像素域，E8 的强对照）。

为什么必须有：SAR 舰船检测的传统方法就是恒虚警率检测。审稿人会问
"你的 P_fa 契约和 CFAR 的区别是什么"。因此 E8（paper/04 §11）规定
CA-CFAR 必须作为**同 P_fa 口径下的基线**出现在 Table 6 里，而不是只比
YOLO 系列的 mAP。

实现要点：
- 只用训练窗均值（cell-averaging），滑窗用积分图，O(HW) 与窗宽无关；
- 阈值系数按指数（强度）分布闭式取：``α = N(P_fa^{-1/N} − 1)``；
- 判决图做连通域后处理，给出候选框数与虚警数，便于与网络的分数场逐条对齐；
- 顶层不依赖 torch / acl / cv2，numpy + scipy 即可。

    python -m qcert.cfar            # 自检：纯杂波上实测 P_fa 是否落在 1e-3 附近
"""
from __future__ import annotations

import numpy as np

try:                                     # scipy 只用于连通域，缺了就退化
    from scipy import ndimage as _ndi
except Exception:                        # pragma: no cover
    _ndi = None


# ------------------------------------------------------------------ 阈值系数
def cfar_alpha(n_train: int, pfa: float) -> float:
    """指数分布假设下的 CA-CFAR 阈值系数。

    ``P_fa = (1 + α/N)^{-N}`` ⇒ ``α = N(P_fa^{-1/N} − 1)``。
    """
    n = float(n_train)
    if n <= 0:
        raise ValueError("n_train 必须为正")
    if not (0.0 < pfa < 1.0):
        raise ValueError("pfa 必须在 (0, 1) 内")
    return float(n * (pfa ** (-1.0 / n) - 1.0))


def ring_kernel(guard: int, train: int) -> np.ndarray:
    """参考窗核：外框 ``train`` 宽、内框 ``guard`` 宽的方环填 1。"""
    g, t = int(guard), int(train)
    if g < 0 or t <= 0:
        raise ValueError("guard ≥ 0、train > 0")
    side = 2 * (g + t) + 1
    k = np.ones((side, side), dtype=np.float64)
    k[g:g + 2 * g + 1, g:g + 2 * g + 1] = 0.0
    return k


def _box_sum(a: np.ndarray, k: np.ndarray) -> np.ndarray:
    """与 ``k`` 同尺寸的滑动求和（积分图，O(HW)）。"""
    kh, kw = k.shape
    ph, pw = kh // 2, kw // 2
    ap = np.pad(a, ((ph + 1, ph), (pw + 1, pw)), mode="constant")
    ii = ap.cumsum(0).cumsum(1)
    h, w = a.shape
    s = (ii[kh:kh + h, kw:kw + w] - ii[0:h, kw:kw + w]
         - ii[kh:kh + h, 0:w] + ii[0:h, 0:w])
    return s


# ------------------------------------------------------------------ 主流程
def ca_cfar(image, guard: int = 2, train: int = 8, pfa: float = 1e-3,
            mask=None) -> dict:
    """对一张强度图做 CA-CFAR。

    参数
    ----
    image : (H, W) 非负强度图（SAR 幅度/强度即可，内部不做对数）
    guard : 保护单元半径（像素）
    train : 参考窗宽度（像素）
    pfa   : 目标虚警率
    mask  : 可选 (H, W) bool，True 的位置参与统计/判决（默认全图）

    返回
    ----
    dict(thr=阈值图, det=bool 判决图, alpha=系数, n_train=参考单元数,
         pfa_target=目标虚警率, pfa_emp=None) —— ``pfa_emp`` 由
    :func:`empirical_pfa` 另行计算，避免与"是否在目标上"混淆。
    """
    img = np.asarray(image, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError("image 必须是 (H, W)")
    k = ring_kernel(guard, train)
    n_train = int(k.sum())
    alpha = cfar_alpha(n_train, pfa)
    ref = _box_sum(img, k) / float(n_train)
    thr = alpha * ref
    det = img > thr
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        det = det & m
    return {"thr": thr, "det": det, "alpha": alpha, "n_train": n_train,
            "pfa_target": float(pfa)}


def empirical_pfa(det, mask=None) -> float:
    """纯杂波区域上的实测虚警率。"""
    d = np.asarray(det, dtype=bool)
    if mask is None:
        return float(d.mean())
    m = np.asarray(mask, dtype=bool)
    n = int(m.sum())
    return float((d & m).sum() / n) if n else float("nan")


def detect_boxes(det, min_area: int = 8) -> np.ndarray:
    """连通域 → 候选框 ``(N, 4)``，格式 ``x1, y1, x2, y2``（含端点）。

    没有 scipy 时退化为包围盒全图（并给出单框），保证流程可跑。
    """
    d = np.asarray(det, dtype=bool)
    if not d.any():
        return np.zeros((0, 4), dtype=np.float64)
    if _ndi is None:                     # pragma: no cover
        ys, xs = np.nonzero(d)
        return np.asarray([[xs.min(), ys.min(), xs.max(), ys.max()]],
                          dtype=np.float64)
    lab, n = _ndi.label(d, structure=np.ones((3, 3), dtype=int))
    if n == 0:
        return np.zeros((0, 4), dtype=np.float64)
    objs = _ndi.find_objects(lab)
    area = np.bincount(lab.ravel(), minlength=n + 1)
    out = []
    for i, sl in enumerate(objs, start=1):
        if sl is None or area[i] < min_area:
            continue
        out.append([sl[1].start, sl[0].start, sl[1].stop - 1, sl[0].stop - 1])
    return np.asarray(out, dtype=np.float64).reshape(-1, 4)


def cfar_metrics(det, boxes, guard: int = 2, train: int = 8,
                 pfa: float = 1e-3, min_area: int = 8) -> dict:
    """把 CA-CFAR 整理成与 E8 表一致的字段。"""
    d = np.asarray(det, dtype=bool)
    h, w = d.shape
    pad = guard + train + 1
    clutter = np.ones((h, w), dtype=bool)
    for b in np.asarray(boxes, dtype=np.float64).reshape(-1, 4):
        x1 = max(0, int(np.floor(b[0])) - pad)
        y1 = max(0, int(np.floor(b[1])) - pad)
        x2 = min(w - 1, int(np.ceil(b[2])) + pad)
        y2 = min(h - 1, int(np.ceil(b[3])) + pad)
        clutter[y1:y2 + 1, x1:x2 + 1] = False
    n_tgt = int(np.asarray(boxes).reshape(-1, 4).shape[0])
    n_det_tgt = 0
    for b in np.asarray(boxes, dtype=np.float64).reshape(-1, 4):
        x1, y1 = max(0, int(np.floor(b[0]))), max(0, int(np.floor(b[1])))
        x2, y2 = min(w - 1, int(np.ceil(b[2]))), min(h - 1, int(np.ceil(b[3])))
        n_det_tgt += int(d[y1:y2 + 1, x1:x2 + 1].any())
    cand = detect_boxes(d, min_area=min_area)
    return {
        "pfa_target": float(pfa), "pfa_emp": empirical_pfa(d, clutter),
        "pd": float(n_det_tgt / n_tgt) if n_tgt else float("nan"),
        "n_target": n_tgt, "n_target_detected": n_det_tgt,
        "n_candidates": int(cand.shape[0]),
        "cluster_px": int(d.sum()),
        "guard": int(guard), "train": int(train),
    }


# ------------------------------------------------------------------ 自检
def _selfcheck(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    h = w = 256
    img = rng.exponential(1.0, size=(h, w))       # 单位均值指数杂波
    tgt = np.zeros((h, w), dtype=bool)
    tgt[100:104, 60:66] = True                     # 6×4 亮目标（小于参考窗）
    img[tgt] += rng.exponential(6.0, size=tgt.sum())
    out = ca_cfar(img, guard=2, train=8, pfa=1e-3)   # 不在判决上加 mask
    pfa_emp = empirical_pfa(out["det"], ~tgt)        # 但 P_fa 只在纯杂波上统计
    pd = float((out["det"] & tgt).any())

    # 多目标回归：3 个亮目标必须逐个计数（防止"并集 .any()"把 3 判成 1）
    img2 = rng.exponential(1.0, size=(h, w))
    boxes = np.asarray([[30, 30, 36, 36], [120, 90, 126, 96], [200, 180, 206, 186]],
                       dtype=np.float64)
    for x1, y1, x2, y2 in boxes.astype(int):
        img2[y1:y2 + 1, x1:x2 + 1] += rng.exponential(6.0, size=(y2 - y1 + 1, x2 - x1 + 1))
    det2 = ca_cfar(img2, guard=2, train=8, pfa=1e-3)["det"]
    m3 = cfar_metrics(det2, boxes, guard=2, train=8, pfa=1e-3)
    return {"alpha": out["alpha"], "n_train": out["n_train"],
            "pfa_target": out["pfa_target"], "pfa_emp": pfa_emp, "pd": pd,
            "n_target_multi": m3["n_target"],
            "n_detected_multi": m3["n_target_detected"],
            "pd_multi": m3["pd"]}


if __name__ == "__main__":
    r = _selfcheck()
    print("=== CA-CFAR 自检（纯指数杂波，256×256）===")
    print(f"参考单元 {r['n_train']}  α={r['alpha']:.3f}  "
          f"目标 P_fa={r['pfa_target']:.1e}  实测 P_fa={r['pfa_emp']:.2e}")
    ok = 0.3 * r["pfa_target"] <= r["pfa_emp"] <= 3.0 * r["pfa_target"]
    print(f"量级 {'一致' if ok else '不一致'}；目标检出 {r['pd']:.0f}")
    print(f"多目标计数 {r['n_detected_multi']}/{r['n_target_multi']}  "
          f"P_d={r['pd_multi']:.2f}")
    assert r["n_target_multi"] == 3, "多目标计数按框统计，应为 3"
    assert r["n_detected_multi"] == 3, "3 个亮目标应全部检出（旧 bug 会返回 1）"
