"""E6 温度 nan 诊断：板上验证 SmiSampler -> npu-smi -> parse_npu_smi 链路。

用法（板上）::

    cd /root/asc_sar/code && python board/diag_smi.py

逐段暴露被 SmiSampler.poll_once 的 except 静默吞掉的失败原因：
PATH / which / subprocess 编码 / 实际输出行 repr / parse 结果。
"""
import os
import shutil
import subprocess
import sys

print("[env] PATH =", os.environ.get("PATH"))
print("[env] LANG =", os.environ.get("LANG"), "| LC_ALL =", os.environ.get("LC_ALL"))
print("[env] python =", sys.version.split()[0],
      "| default encoding =", sys.getdefaultencoding(),
      "| fs/filesystem =", sys.getfilesystemencoding())
try:
    import locale
    print("[env] preferred =", locale.getpreferredencoding(False))
except Exception as e:  # pragma: no cover
    print("[env] locale ERR:", e)

print("[which] npu-smi =", shutil.which("npu-smi"))

r = None
try:
    r = subprocess.run(["npu-smi", "info"], capture_output=True, text=True,
                       timeout=3)
    print("[run] rc =", r.returncode, "| stderr head:", r.stderr[:120])
    print("[run] stdout lines containing '|':")
    for line in r.stdout.splitlines():
        if "|" in line:
            print("   ", repr(line))
except UnicodeDecodeError as e:
    print("[run] UnicodeDecodeError（text=True 按本地编码解码失败）:", e)
    # 复跑一次按 bytes 看原始输出
    rb = subprocess.run(["npu-smi", "info"], capture_output=True, timeout=3)
    print("[run] raw bytes head:", rb.stdout[:400])
    print("[run] raw bytes repr:", repr(rb.stdout[:400]))
except Exception as e:
    print("[run] ERR:", type(e).__name__, e)
    sys.exit(0)

if r is not None:
    from qcert.thermal import parse_npu_smi, SmiSampler
    print("[parse] parse_npu_smi(stdout) ->", parse_npu_smi(r.stdout))
    s = SmiSampler()
    snap = s.poll_once()
    print("[sampler] poll_once ->", snap, "| n_samples =", len(s.samples))
