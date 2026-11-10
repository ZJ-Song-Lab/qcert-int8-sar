"""QCert: 整数域误警率契约（v3）。

全部模块只依赖 numpy（可选 onnx / torch / PIL），可在 Windows x86 与
Atlas 板端 Python 3.10 上导入。pyACL / AMCT 相关代码一律放在
``code/board/`` 下，且顶层不 import，保证本包可离线单测。

模块对应 paper/05_实施路线与代码资产复用.md §2：
    1 分数场 dump      -> board/dump_scores.py + scores.py
    2 scale 表         -> scale_table.py
    3 整数分位数       -> qint.py
    4 整数判决         -> decision.py
    5 饱和审计         -> audit.py
    6 阈值搬运         -> transfer.py
    7 P_fa 评测        -> pfa_eval.py
    8 热稳态           -> thermal.py
    9/10 零模型与损失  -> h0/
"""
from . import (  # noqa: F401
    audit,
    decision,
    pfa_eval,
    qint,
    scale_table,
    scores,
    seastate,
    thermal,
    transfer,
)

__all__ = [
    "audit",
    "decision",
    "pfa_eval",
    "qint",
    "scale_table",
    "scores",
    "seastate",
    "thermal",
    "transfer",
]
__version__ = "0.3.0"
