"""板上探针 v2：纯 pyACL 单 init（带 dump config），抓 int8 OM 算子输出。

与 v1 的区别：不经 AclModel（它的 acl.init() 二次调用可能冲掉 dump 配置）。
运行（板上）：
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    python3 /root/code/board/acl_dump_probe.py
"""
import json
import sys
from pathlib import Path

import numpy as np

CFG = "/root/acl_dump_cfg.json"
DUMP_DIR = "/root/acl_dump"
MODEL = "/root/asc_sar/models/yolo11n_ssdd_int8.om"
IMG_DIR = "/root/asc_sar/data/images"

cfg = {
    "dump": {
        "dump_path": DUMP_DIR,
        "dump_mode": "output",
        "dump_op_switch": "on",
        "dump_list": [{"model_name": "yolo11n_ssdd_int8"}],
    }
}
Path(CFG).write_text(json.dumps(cfg))
print("[probe] dump config ->", CFG, flush=True)

import acl  # noqa: E402

print("[probe] acl.init:", acl.init(CFG), flush=True)

sys.path.insert(0, "/root/asc_sar/deploy")
from acl_infer import _get_desc, _status, _val  # noqa: E402
from postprocess import load_image_rgb, preprocess  # noqa: E402

_status(acl.rt.set_device, 0)
ctx = _val(acl.rt.create_context, 0)
model_id = _val(acl.mdl.load_from_file, MODEL)
print("[probe] load model_id =", model_id, flush=True)

p = sorted(Path(IMG_DIR).iterdir())[0]
x, _, _ = preprocess(load_image_rgb(p), 640)
x = np.ascontiguousarray(x, dtype=np.float32)
print("[probe] img =", p.name, "x.nbytes =", x.nbytes, flush=True)

in_ds = _val(acl.mdl.create_dataset)
in_ptr = _val(acl.rt.malloc, x.nbytes, 0)
buf = _val(acl.create_data_buffer, in_ptr, x.nbytes)
_status(acl.mdl.add_dataset_buffer, in_ds, buf)
_status(acl.rt.memcpy, in_ptr, x.nbytes, x.ctypes.data, x.nbytes, 1)

desc = _get_desc(model_id)
n_out = _val(acl.mdl.get_num_outputs, desc)
out_ds = _val(acl.mdl.create_dataset)
out_ptrs = []
for i in range(n_out):
    sz = _val(acl.mdl.get_output_size_by_index, desc, i)
    ptr = _val(acl.rt.malloc, sz, 0)
    b = _val(acl.create_data_buffer, ptr, sz)
    _status(acl.mdl.add_dataset_buffer, out_ds, b)
    out_ptrs.append((ptr, sz))

r = acl.mdl.execute(model_id, in_ds, out_ds)
print("[probe] execute:", r, flush=True)
sync = getattr(acl.rt, "synchronize_context", None)
if sync is not None:
    print("[probe] sync:", sync(ctx), flush=True)

acl.mdl.unload(model_id)
acl.rt.destroy_context(ctx)
acl.rt.reset_device(0)
print("[probe] finalize:", acl.finalize(), flush=True)

files = [f for f in sorted(Path(DUMP_DIR).rglob("*")) if f.is_file()]
print(f"[probe] dump 文件数 = {len(files)}")
cv3 = [f for f in files if "cv3" in f.name]
print(f"[probe] 文件名含 cv3 的 = {len(cv3)}")
for f in cv3[:20]:
    print("   ", f.name, f.stat().st_size, "B")
for f in files[:15]:
    print("   [样例]", f.relative_to(DUMP_DIR), f.stat().st_size, "B")
