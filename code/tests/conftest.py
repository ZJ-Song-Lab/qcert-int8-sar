"""pytest 公共配置：把 ``code/`` 与 ``asc-sar/deploy`` 挂到 sys.path。

这样测试可以直接 ``import qcert`` / ``import board`` / ``import acl_infer``，
而 ``board`` 在 x86 上不 import acl（见 board/__init__.py）。
"""
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
DEPLOY = CODE.parent / "asc-sar" / "deploy"

for p in (CODE, DEPLOY):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
