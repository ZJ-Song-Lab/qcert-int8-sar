"""E6 温度 nan 根因定位：acl 推理进程内并发采样 npu-smi。

两个测试一次跑完：
- T1 主线程在推理循环内直接 poll（带异常打印）：
  失败 -> fork 自 acl 进程的 npu-smi 挂死/超时（进程级问题）；
  成功 -> 线程级问题（daemon 线程抢不到 GIL）。
- T2 标准 SmiSampler daemon 线程并发 15s，看 samples 是否为空。
"""
import subprocess
import sys
import time

sys.path.insert(0, "/root/asc_sar/deploy")
sys.path.insert(0, "/root/asc_sar/code")

from acl_infer import AclModel, preprocess          # noqa: E402
from postprocess import decode_dense                # noqa: E402
from qcert.thermal import SmiSampler                # noqa: E402


def poll_verbose():
    try:
        t0 = time.perf_counter()
        r = subprocess.run(["npu-smi", "info"], capture_output=True, text=True,
                           timeout=3)
        dt = time.perf_counter() - t0
        print(f"[T1 poll] ok in {dt:.2f}s rc={r.returncode} out={len(r.stdout)}B")
        return True
    except Exception as e:
        print(f"[T1 poll] {type(e).__name__}: {e}")
        return False


print("== init model ==")
model = AclModel("/root/asc_sar/models/yolo11n_ssdd_int8.om")
x, ratio, pad = preprocess("/root/asc_sar/data/calib/000955.jpg", imgsz=640)

print("== T1: idle poll (no inference yet) ==")
poll_verbose()

print("== T1: poll inside inference loop ==")
ok_count = 0
for i in range(5):
    raw = [a for a, _ in model.infer_raw(x)]
    p = decode_dense(raw)["prob"]
    if poll_verbose():
        ok_count += 1
print(f"[T1] {ok_count}/5 polls ok inside loop")

print("== T2: SmiSampler daemon during 15s inference ==")
s = SmiSampler(interval=1.0)
s.start()
t0 = time.perf_counter()
n_inf = 0
while time.perf_counter() - t0 < 15.0:
    raw = [a for a, _ in model.infer_raw(x)]
    p = decode_dense(raw)["prob"]
    n_inf += 1
s.stop()
time.sleep(1.5)   # 不 join：SmiSampler._stop 遮蔽了 Thread._stop()，join 会 TypeError
print(f"[T2] {n_inf} frames in 15s; sampler n_samples={len(s.samples)}")
if s.samples:
    print("[T2] first sample:", s.samples[0])
    print("[T2] last  sample:", s.samples[-1])
