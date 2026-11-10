"""验证 310B 表格行温度解析：parse_npu_smi(npu-smi info 输出)。"""
import subprocess
from qcert.thermal import parse_npu_smi

txt = subprocess.run(["npu-smi", "info"], capture_output=True, text=True,
                     timeout=10).stdout
print("---- npu-smi info ----")
print(txt)
print("---- parsed ----")
print(parse_npu_smi(txt))
