"""E0–E9 实验驱动脚本（协议见 paper/04_实验设计方案.md）。

约定
----
- 每个 ``run_eN.py`` 都可独立执行：``python -m experiments.run_e0 --help``；
- 所有脚本把结果写到 ``code/results/<eN>/`` 下的 ``.csv`` / ``.md`` / ``.json``；
- 本机（Windows，无 acl）能跑：E1 / E4 / E5 的离线仿真、E0 / E2 / E3 的 dump 复算、
  E6 的 --dry-run、E7–E9 的汇总；
- 必须上板：E2 的逐层 dump、E3 的 1000 次 bit-exact、E6 的热稳态、E7 的功耗。
"""
