"""PyACL 推理 + 评测 + 性能实测 (在 Atlas 200I DK 上运行)。

功能:
- 加载 .om, 对测试集逐图推理, 解码 + NMS, 与 GT 计算 mAP@0.5
- --benchmark: 输出纯推理延迟(mean/p50/p95)与 FPS
- --power: 后台采样 npu-smi 功耗, 输出平均/峰值功耗
- --save-dir: 保存预测 txt (class x1 y1 x2 y2 conf), 供 tools/eval_map.py 复算

运行前: source /usr/local/Ascend/ascend-toolkit/set_env.sh
依赖: numpy + Pillow (无需 cv2/torch)
"""
import argparse
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "tools"))
from postprocess import (decode_dense, decode_outputs, load_image_rgb,  # noqa: E402
                         preprocess, scale_coords)
from eval_map import evaluate, load_yolo_gt  # noqa: E402

try:
    import acl  # pyACL, 需先 source /usr/local/Ascend/ascend-toolkit/set_env.sh
except ImportError:  # 允许离线导入本模块做单元测试
    acl = None

ACL_MEM_MALLOC_HUGE_FIRST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
NUMPY_DTYPE = {0: np.float32, 1: np.float16, 2: np.int8, 3: np.int32,
               4: np.uint8, 9: np.int64}


# ------------------------------------------------------------------ pyACL 兼容层
# 旧版 pyACL: 返回 (值, 错误码); 新版: 直接返回值, 出错抛异常。
def _val(fn, *args):
    r = fn(*args)
    if isinstance(r, tuple):
        code = r[-1]
        if isinstance(code, int) and code != 0:
            raise RuntimeError(f"{getattr(fn, '__name__', fn)} 失败 code={code}")
        return r[0]
    return r


def _status(fn, *args):
    r = fn(*args)
    if isinstance(r, tuple):
        r = r[-1] if isinstance(r[-1], int) else 0
    if isinstance(r, int) and r != 0:
        raise RuntimeError(f"{getattr(fn, '__name__', fn)} 失败 code={r}")


def _get_desc(model_id):
    try:
        return _val(acl.mdl.get_desc, model_id)
    except TypeError:
        desc = _val(acl.mdl.create_desc)
        _status(acl.mdl.get_desc, desc, model_id)
        return desc


def _dims_of(desc, getter, idx):
    """不同 CANN 版本返回 [[..]] / {..'dims'} / ({..}, 0), 统一成 list[int]。"""
    d = _val(getter, desc, idx)
    if isinstance(d, dict):
        d = d.get("dims", [])
    if d and isinstance(d[0], (list, tuple)):
        d = list(d[0])
    return [int(x) for x in d]


def _acl_fn(*cands):
    for root, name in cands:
        fn = getattr(root, name, None)
        if fn is not None:
            return fn
    return None


def _create_data_buffer(ptr, size):
    fn = _acl_fn((acl.mdl, "create_data_buffer"), (acl, "create_data_buffer"))
    if fn is None:
        raise RuntimeError("pyACL 缺少 create_data_buffer")
    try:
        return _val(fn, ptr, size)
    except TypeError:
        return _val(fn, ptr, size, 0)


def _ptr_bytes(ptr, size):
    util = getattr(acl, "util", None)
    if util is not None and hasattr(util, "ptr_to_bytes"):
        return util.ptr_to_bytes(ptr, size)
    import ctypes
    return bytes((ctypes.c_ubyte * size).from_address(ptr))


def _np_ptr(arr):
    """numpy 数组 -> host 指针(直接用 ctypes 取地址, 避免 acl.util.numpy_to_ptr 的弃用告警)。"""
    return arr.ctypes.data


class AclModel:
    """单输入单/多输出 OM 封装(dataset 复用, 适合 benchmark)。"""

    def __init__(self, om_path, device_id=0):
        if acl is None:
            raise RuntimeError("未找到 pyACL: 请先 source /usr/local/Ascend/ascend-toolkit/set_env.sh "
                               "并设置 PYTHONPATH 包含 .../latest/python/site-packages")
        self.device_id = device_id
        r = acl.init()
        if r != 0 and r != 100002:  # 100002 = 重复 init（同进程第二个模型）
            raise RuntimeError(f"acl.init 失败 code={r}")
        _status(acl.rt.set_device, device_id)
        self.context = _val(acl.rt.create_context, device_id)
        self.model_id = _val(acl.mdl.load_from_file, om_path)
        self.desc = _get_desc(self.model_id)

        n_in = _val(acl.mdl.get_num_inputs, self.desc)
        n_out = _val(acl.mdl.get_num_outputs, self.desc)
        self.in_sizes = [_val(acl.mdl.get_input_size_by_index, self.desc, i) for i in range(n_in)]
        self.out_sizes = [_val(acl.mdl.get_output_size_by_index, self.desc, i) for i in range(n_out)]
        self.in_dims = [_dims_of(self.desc, acl.mdl.get_input_dims, i) for i in range(n_in)]
        self.out_dims = [_dims_of(self.desc, acl.mdl.get_output_dims, i) for i in range(n_out)]
        self.out_dtypes = []
        for i in range(n_out):
            try:
                self.out_dtypes.append(NUMPY_DTYPE.get(int(_val(acl.mdl.get_output_data_type,
                                                               self.desc, i)), np.float32))
            except Exception:
                self.out_dtypes.append(np.float32)

        print(f"[acl] 模型 {om_path}")
        print(f"[acl] 输入 {self.in_dims} {self.in_sizes}B")
        print(f"[acl] 输出 {self.out_dims} dtype={[np.dtype(d).name for d in self.out_dtypes]}")

        self.in_dataset, self.in_dev = self._make_dataset(self.in_sizes)
        self.out_dataset, self.out_dev = self._make_dataset(self.out_sizes)

    def _make_dataset(self, sizes):
        ds = _val(acl.mdl.create_dataset)
        ptrs = []
        for sz in sizes:
            ptr = _val(acl.rt.malloc, sz, ACL_MEM_MALLOC_HUGE_FIRST)
            _status(acl.mdl.add_dataset_buffer, ds, _create_data_buffer(ptr, sz))
            ptrs.append(ptr)
        return ds, ptrs

    def infer_raw(self, x):
        """返回 ``[(ndarray, dtype), ...]``，**保留 OM 输出的原始 dtype**。

        INT8 模型必须走这里：``infer()`` 会 astype(float32)，把 int8 原值丢掉，
        而 v3 的整数域叙事要求原值可复现。
        """
        x = np.ascontiguousarray(x, dtype=np.float32)
        if x.nbytes != self.in_sizes[0]:
            raise ValueError(f"输入 {x.shape} 占 {x.nbytes}B, 与模型输入 {self.in_sizes[0]}B 不一致")
        _status(acl.rt.set_context, self.context)  # 多模型共存时必须切回自己的
        _status(acl.rt.memcpy, self.in_dev[0], x.nbytes, _np_ptr(x), x.nbytes,
                ACL_MEMCPY_HOST_TO_DEVICE)
        _status(acl.mdl.execute, self.model_id, self.in_dataset, self.out_dataset)
        outs = []
        for i, size in enumerate(self.out_sizes):
            raw = _ptr_bytes(self.out_dev[i], size)
            arr = np.frombuffer(raw, dtype=self.out_dtypes[i])
            total = int(np.prod(self.out_dims[i])) if self.out_dims[i] else arr.size
            outs.append((arr[:total].reshape(self.out_dims[i]), self.out_dtypes[i]))
        return outs

    def infer(self, x):
        return [a.astype(np.float32) for a, _ in self.infer_raw(x)]

    def describe(self) -> dict:
        """模型的输入/输出元数据（写 JSON 用）。"""
        return {
            "in_dims": self.in_dims,
            "in_sizes": [int(s) for s in self.in_sizes],
            "out_dims": self.out_dims,
            "out_sizes": [int(s) for s in self.out_sizes],
            "out_dtypes": [np.dtype(d).name for d in self.out_dtypes],
        }

    def destroy(self):
        for ds, ptrs in ((getattr(self, "in_dataset", None), getattr(self, "in_dev", [])),
                         (getattr(self, "out_dataset", None), getattr(self, "out_dev", []))):
            if ds is None:
                continue
            try:
                n = _val(acl.mdl.get_dataset_num_buffers, ds)
                for i in range(n):
                    _status(acl.mdl.destroy_data_buffer, _val(acl.mdl.get_dataset_buffer, ds, i))
            except Exception:
                pass
            _status(acl.mdl.destroy_dataset, ds)
            for p in ptrs:
                try:
                    _status(acl.rt.free, p)
                except Exception:
                    pass
        _status(acl.mdl.unload, self.model_id)
        _status(acl.rt.destroy_context, self.context)
        try:
            _status(acl.rt.reset_device, self.device_id)
        except Exception:
            pass
        try:
            acl.finalize()
        except Exception:
            pass


# ------------------------------------------------------------------ v3 新增：dump
def qcert_module():
    """惰性导入 ``code/qcert``（板上没有时把仓库根加进 sys.path）。"""
    root = HERE.parents[1] / "code"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import qcert  # noqa: F401
    return qcert


def load_scale_delta(scale_json, name_candidates=("output0", "output", "scores")):
    """从 scale 表 json 里抠出输出层的 delta / zero_point。"""
    if not scale_json:
        return None, 0
    qcert_module()
    from qcert.scale_table import ScaleTable
    table = ScaleTable.from_json(scale_json)
    for cand in name_candidates:
        for key in table.names():
            if cand in key:
                e = table.get(key)
                return float(e.delta), int(e.zero_point)
    names = table.names()
    if not names:
        return None, 0
    print(f"[dump] scale 表里没有输出层名，取第一项 {names[0]!r}")
    e = table.get(names[0])
    return float(e.delta), int(e.zero_point)


def bit_exact_repeat(model, x, n=1000):
    """同一输入重复 n 次，返回逐 bit 一致率（假设 A1 的实测）。"""
    first = None
    n_ok, n_tot = 0, 0
    for _ in range(int(n)):
        raw = model.infer_raw(x)
        sig = tuple(np.ascontiguousarray(a).tobytes() for a, _ in raw)
        if first is None:
            first = sig
        n_tot += 1
        n_ok += int(sig == first)
    return {"n": n_tot, "n_exact": n_ok, "rate": n_ok / max(n_tot, 1)}


def dump_score_field(model, x, meta, nc, out_path, imgsz=640,
                     delta=None, zero_point=0, impl="fp32"):
    """跑一次推理并把稠密分数场写成 qcert 的 .npz。"""
    qcert_module()
    from qcert.scores import ScoreField, save_score_npz
    raw = model.infer_raw(x)
    dtypes = [d for _, d in raw]
    arrs = [a.astype(np.float32) for a, _ in raw]
    if any(np.issubdtype(d, np.integer) for d in dtypes):
        # INT8 链路：保住原始整数码（只取分数通道），再按 delta 反量化出浮点视图
        i8 = [a for a, d in raw if np.issubdtype(d, np.integer)][0].astype(np.int8)
        nc_axis = 1 if i8.ndim >= 2 else 0
        ch_first = np.moveaxis(i8, nc_axis, 0).reshape(i8.shape[nc_axis], -1)
        s_q = ch_first[4:4 + nc].max(axis=0).astype(np.int8)
        d = float(delta) if delta else 1.0
        hs = decode_dense([i8.astype(np.int32)], nc, imgsz)["head_sizes"]
        sf = ScoreField(prob=s_q.astype(np.float32) * d, logit=None, centers=None,
                        head_sizes=np.asarray(hs, dtype=np.int32), s_q=s_q,
                        delta=d, zero_point=int(zero_point), meta=meta,
                        stem=Path(out_path).stem)
    else:
        field = decode_dense(arrs, nc, imgsz)
        sf = ScoreField(prob=field["prob"], logit=field["logit"],
                        head_sizes=np.asarray(field["head_sizes"], dtype=np.int32),
                        meta=meta, stem=Path(out_path).stem)
    sf.meta.setdefault("impl", impl)
    save_score_npz(out_path, sf)
    return sf


def dump_layers(model, x, out_dir, stem, manifest, keep_idx=None):
    """把每个输出张量原样落成 .npy（中间层审计的最小实现）。

    keep_idx: 只落指定 OM 输出 index（E2 压测集只 dump Neck 三层，控制磁盘）。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, (a, d) in enumerate(model.infer_raw(x)):
        if keep_idx is not None and i not in keep_idx:
            continue
        p = out_dir / f"{stem}__out{i}.npy"
        np.save(p, a)
        manifest.append({"stem": stem, "index": i, "file": p.name,
                         "shape": list(a.shape), "dtype": np.dtype(d).name})


class PowerSampler(threading.Thread):
    """后台采样 npu-smi 功耗(W)。"""

    def __init__(self, interval=0.3):
        super().__init__(daemon=True)
        self.samples, self.interval = [], interval
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(["npu-smi", "info"], capture_output=True,
                                     text=True, timeout=3).stdout
                m = re.findall(r"\|\s*(\d+(?:\.\d+)?)\s+\d+\s+\d+\s*/\s*\d+\s*\|", out)
                if m:
                    self.samples.append(float(m[0]))
            except Exception:
                pass
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="OM 模型")
    ap.add_argument("--images", required=True, help="测试图像目录")
    ap.add_argument("--labels", default=None, help="YOLO 标签目录(GT), 给定则评估 mAP")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--nc", type=int, default=0, help="类别数, 0=按模型输出通道自动推断")
    ap.add_argument("--conf", type=float, default=0.001, help="解码置信度阈值(算 mAP 需取低值)")
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--save-dir", default=None)
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--power", action="store_true")
    # ---- v3（QCert）新增 ----
    ap.add_argument("--dump-scores", default=None,
                    help="把每图的稠密分数场写成 qcert .npz（E0/E1/E4/E5/E9 的输入）")
    ap.add_argument("--dump-layers", default=None,
                    help="把每个输出张量原样落成 .npy（E2 的最小中间层审计）")
    ap.add_argument("--dump-index", default=None,
                    help="只 dump 指定 OM 输出 index(逗号分隔, E2 Neck 层如 48,61,79)")
    ap.add_argument("--scale-json", default=None,
                    help="ScaleTable json：INT8 输出层 delta/zero_point")
    ap.add_argument("--om-meta", default=None, help="写出模型输入/输出元数据 json")
    ap.add_argument("--repeat", type=int, default=0,
                    help="同一输入重复 N 次，检查逐 bit 一致（A1），0=不做")
    ap.add_argument("--repeat-json", default=None,
                    help="把 A1 重复一致率结果写成 json（E3 --repeat 的输入）")
    ap.add_argument("--impl", default="fp32", choices=["fp32", "fp16", "int8"],
                    help="写进分数场 meta 的实现标签")
    ap.add_argument("--contrast", type=float, default=1.0,
                    help="压测集对比度档位(写进分数场 meta, E2)")
    args = ap.parse_args()

    imgs = sorted(p for p in Path(args.images).iterdir()
                  if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"))
    assert imgs, f"图像目录为空: {args.images}"

    model = AclModel(args.model, args.device_id)
    nc = args.nc if args.nc > 0 else max(1, model.out_dims[0][1] - 4)
    print(f"[eval] nc={nc}")
    delta, zero_point = load_scale_delta(args.scale_json)
    if args.om_meta:
        import json
        Path(args.om_meta).parent.mkdir(parents=True, exist_ok=True)
        with open(args.om_meta, "w", encoding="utf-8") as fp:
            json.dump({"model": str(args.model), "nc": nc,
                       "scale_json": args.scale_json, **model.describe()},
                      fp, ensure_ascii=False, indent=2)
        print(f"[meta] {args.om_meta}")
    layer_manifest = []
    try:
        preds, gts = {}, {}
        if args.save_dir:
            Path(args.save_dir).mkdir(parents=True, exist_ok=True)

        print(f"[eval] {len(imgs)} 张图, conf={args.conf}")
        t0 = time.time()
        for n, p in enumerate(imgs, 1):
            img = load_image_rgb(p)
            ih, iw = img.shape[:2]
            x, ratio, pad = preprocess(img, args.imgsz)
            if args.dump_layers:
                # 中间层审计: 只落原始张量, 不解码(debug OM 有 89 个输出, decode 不适用)
                dump_layers(model, x, args.dump_layers, p.stem, layer_manifest,
                            keep_idx=(set(int(v) for v in args.dump_index.split(","))
                                      if args.dump_index else None))
                if n % 50 == 0 or n == len(imgs):
                    print(f"  [{n}/{len(imgs)}] {time.time() - t0:.0f}s")
                continue
            outs = model.infer(x)
            boxes, confs, cls = decode_outputs(outs, nc, args.conf, args.iou)
            boxes = scale_coords(boxes, (ih, iw), (ratio, *pad))
            preds[p.stem] = np.column_stack([cls, boxes, confs]).astype(np.float32) \
                if len(boxes) else np.zeros((0, 6), np.float32)
            if args.labels:
                gts[p.stem] = load_yolo_gt(Path(args.labels) / (p.stem + ".txt"), iw, ih)
            if args.save_dir:
                with open(Path(args.save_dir) / (p.stem + ".txt"), "w") as fp:
                    for row in preds[p.stem]:
                        fp.write(" ".join(f"{v:.4f}" for v in row) + "\n")
            if args.dump_scores:
                meta = {"imgsz": args.imgsz, "iw": iw, "ih": ih, "ratio": ratio,
                        "pad": list(pad), "impl": args.impl, "nc": nc,
                        "model": Path(args.model).name,
                        "contrast": float(args.contrast)}
                if args.labels:
                    meta["boxes"] = load_yolo_gt(
                        Path(args.labels) / (p.stem + ".txt"), iw, ih).tolist()
                dump_score_field(model, x, meta, nc,
                                 Path(args.dump_scores) / (p.stem + ".npz"),
                                 args.imgsz, delta, zero_point, args.impl)
            if n % 50 == 0 or n == len(imgs):
                print(f"  [{n}/{len(imgs)}] {time.time() - t0:.0f}s")

        if args.dump_layers:
            import json
            with open(Path(args.dump_layers) / "manifest.json", "w",
                      encoding="utf-8") as fp:
                json.dump(layer_manifest, fp, ensure_ascii=False, indent=2)

        if args.repeat > 0:
            x = preprocess(load_image_rgb(imgs[0]), args.imgsz)[0]
            st = bit_exact_repeat(model, x, args.repeat)
            st["model"] = Path(args.model).name
            st["img"] = imgs[0].name
            print(f"[A1] 重复 {st['n']} 次逐 bit 一致率 = {st['n_exact']}/{st['n']}")
            if args.repeat_json:
                import json
                Path(args.repeat_json).parent.mkdir(parents=True, exist_ok=True)
                with open(args.repeat_json, "w", encoding="utf-8") as fp:
                    json.dump(st, fp, ensure_ascii=False, indent=2)
                print(f"[A1] 结果写入 {args.repeat_json}")

        if args.labels:
            ap50, prec, rec = evaluate(preds, gts, 0.5)
            print(f"[精度] mAP@0.5={ap50:.4f}  P={prec:.4f}  R={rec:.4f}")

        if args.benchmark:
            print(f"[bench] warmup=10 iters={args.iters}")
            xs = [preprocess(load_image_rgb(p), args.imgsz)[0]
                  for p in imgs[:min(32, len(imgs))]]
            for x in xs[:10]:
                model.infer(x)
            sampler = PowerSampler() if args.power else None
            if sampler:
                sampler.start()
            times = []
            wall0 = time.perf_counter()
            for i in range(args.iters):
                t = time.perf_counter()
                model.infer(xs[i % len(xs)])
                times.append((time.perf_counter() - t) * 1000)
            wall = time.perf_counter() - wall0
            if sampler:
                sampler.stop()
            times = np.array(times)
            print(f"[bench] 延迟 mean={times.mean():.2f}ms p50={np.percentile(times, 50):.2f}ms "
                  f"p95={np.percentile(times, 95):.2f}ms")
            print(f"[bench] 纯推理 FPS={1000 / times.mean():.2f}  端到端 FPS={args.iters / wall:.2f}")
            if sampler and sampler.samples:
                print(f"[功率] 平均 {np.mean(sampler.samples):.2f}W 峰值 {max(sampler.samples):.2f}W"
                      f" (n={len(sampler.samples)})")
    finally:
        model.destroy()


if __name__ == "__main__":
    main()
