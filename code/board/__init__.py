"""板上桥接层（pyACL / npu-smi）。**本包顶层不 import acl**。

对应 paper/05_实施路线与代码资产复用.md §2 的模块 1 / 2 / 8：

- ``runner``        路径挂载、惰性 ``AclModel``、预处理、结果落盘
- ``dump_scores``   模块 1：分数场 dump -> ``qcert`` 的 .npz（E0/E1/E4/E5/E9 输入）
- ``bench_latency`` 模块 8/9：延迟、功率、热稳态（E6/E7 输入）
- ``probe_scales``  模块 2 / M0 硬门：scale 表读出，验证假设 A3

每个脚本都提供 ``--dry-run``，在 x86 上不依赖 pyACL 也能把通路跑通；
dry-run 产物的 ``source`` 字段会标成 ``dry-run``，不可当作板级实测数据。
"""
