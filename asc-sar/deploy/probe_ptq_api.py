"""板上环境检查: onnxoptimizer / onnx / 磁盘空间。"""
import subprocess

for mod in ("onnx", "onnxruntime", "onnxoptimizer"):
    try:
        m = __import__(mod)
        print(f"{mod}: OK {getattr(m, '__version__', '?')}")
    except Exception as exc:
        print(f"{mod}: MISSING ({exc})")

print(subprocess.run(["df", "-h", "/root"], capture_output=True, text=True).stdout)
