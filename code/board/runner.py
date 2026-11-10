"""板上桥接层的公共件：路径挂载、惰性 pyACL、预处理与结果落盘。

设计约束（paper/05 §2、§8）：

- **本模块顶层不 import acl**。x86 上没有 pyACL，导入 ``board.runner``
  必须成功，否则 E0–E9 的离线自检与 pytest 全都跑不起来；
- 只有真正要建会话时才 :func:`load_model`，失败时给可操作的提示而不是
  一串 ImportError 栈；
- 把 ``code/`` 与 ``asc-sar/deploy`` 挂到 ``sys.path``，让 ``qcert`` 与
  ``acl_infer / postprocess / eval_map`` 都能被 import；
- 结果统一落 ``code/results/<eid>/``，与 ``experiments.common`` 的约定一致。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]      # .../code
ROOT = CODE.parent                              # .../dinghuibianyuan
DEPLOY = ROOT / "asc-sar" / "deploy"
RESULTS = CODE / "results"


def mount_paths() -> tuple:
    """把 ``code/`` 与 ``asc-sar/deploy`` 挂到 ``sys.path``（幂等）。"""
    for p in (CODE, DEPLOY):
        s = str(p)
        if p.is_dir() and s not in sys.path:
            sys.path.insert(0, s)
    return CODE, DEPLOY


mount_paths()          # 导入本模块即挂载：后续 qcert / acl_infer 才 import 得到


# ------------------------------------------------------------------ 文件
def ensure_dir(p) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def json_dump(obj, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def json_load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ------------------------------------------------------------------ pyACL（惰性）
def acl_available() -> bool:
    """pyACL 是否可 import（板上 True，x86 上 False）。"""
    try:
        import acl  # noqa: F401
    except Exception:
        return False
    return True


def load_model(om_path, device_id: int = 0):
    """惰性构造 ``acl_infer.AclModel``——本仓库唯一的 pyACL 会话入口。"""
    mount_paths()
    if not Path(om_path).exists():
        raise FileNotFoundError(f"OM 不存在: {om_path}")
    try:
        from acl_infer import AclModel
    except Exception as exc:                       # pragma: no cover - 板上路径
        raise RuntimeError(
            "无法加载 acl_infer.AclModel。请确认：\n"
            "  1) 在 Atlas 200I DK（或已装 CANN 的机器）上运行；\n"
            "  2) 已 source /usr/local/Ascend/ascend-toolkit/set_env.sh；\n"
            "  3) PYTHONPATH 含 pyACL 的 site-packages。\n"
            f"原始错误：{exc}") from exc
    return AclModel(str(om_path), device_id)


# ------------------------------------------------------------------ 图像
def list_images(root, limit: int = 0):
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(root)
    imgs = sorted(p for p in root.iterdir()
                  if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"))
    if not imgs:
        raise FileNotFoundError(f"图像目录为空: {root}")
    return imgs[:int(limit)] if limit and int(limit) > 0 else imgs


def preprocess_image(path, imgsz: int = 640):
    """原图 -> ``(x, ratio, pad, (ih, iw))``，与 ``acl_infer`` 的 dump 口径一致。"""
    mount_paths()
    from postprocess import load_image_rgb, preprocess
    img = load_image_rgb(Path(path))
    ih, iw = img.shape[:2]
    x, ratio, pad = preprocess(img, imgsz)
    return x, float(ratio), tuple(float(v) for v in pad), (int(ih), int(iw))


def nc_of(model, nc: int = 0) -> int:
    """类别数：显式给了就用给的，否则按输出通道数 − 4 推断。"""
    if int(nc) > 0:
        return int(nc)
    dims = getattr(model, "out_dims", None) or []
    if dims and len(dims[0]) >= 2:
        return max(1, int(dims[0][1]) - 4)
    raise ValueError("无法从模型输出推断 nc，请显式给 --nc")


# ------------------------------------------------------------------ 分数提取
def scores_once(model, x, nc: int, imgsz: int = 640, delta: float | None = None):
    """跑一次并把输出压成 ``(N,)`` 分数向量。

    口径与 ``acl_infer.dump_score_field`` **必须保持一致**：INT8 输出取
    ``ch_first[4:4+nc]`` 的通道 max 作为原始整数码，再按 delta 反量化。
    这里不直接复用它是因为后者会逐帧写 .npz，长跑时不可接受。
    """
    import numpy as np

    mount_paths()
    from postprocess import decode_dense

    raw = model.infer_raw(x)
    if any(np.issubdtype(d, np.integer) for d in (dt for _, dt in raw)):
        i8 = [a for a, d in raw if np.issubdtype(d, np.integer)][0].astype(np.int8)
        nc_axis = 1 if i8.ndim >= 2 else 0
        ch_first = np.moveaxis(i8, nc_axis, 0).reshape(i8.shape[nc_axis], -1)
        s_q = ch_first[4:4 + nc].max(axis=0)
        return s_q.astype(np.float64) * float(delta or 1.0)
    arrs = [a.astype(np.float32) for a, _ in raw]
    return np.asarray(decode_dense(arrs, nc, imgsz)["prob"], dtype=np.float64)


def frame_fn_from_model(model, xs, nc: int, imgsz: int = 640,
                        delta: float | None = None, lam: float | None = None):
    """把模型包成 ``qcert.thermal`` 需要的 ``frame_fn()``。

    返回 ``{"t_model_ms": .., "pfa": ..}``；``lam`` 为 None 时不给 pfa
    （E6 的 ``window_drift`` 会报 insufficient，而不是编一个数出来）。
    """
    import time

    import numpy as np

    state = {"i": 0}

    def fn():
        x = xs[state["i"] % len(xs)]
        state["i"] += 1
        t0 = time.perf_counter()
        s = scores_once(model, x, nc, imgsz, delta)
        ms = (time.perf_counter() - t0) * 1000.0
        rec = {"t_model_ms": ms}
        if lam is not None and s.size:
            rec["pfa"] = float(np.mean(s > float(lam)))
        return rec

    return fn
