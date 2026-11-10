"""E5 probe4（临时）：raw logit（由 prob 反变换）落在哪个量化格点？

候选 A: AMCT int8 格点 quant_delta（head0=0.4185）→ 输出被量化
候选 B: fp16 格点（分段 ulp）→ 输出未量化，量化效应来自上游传导
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qcert.scores import load_dir  # noqa: E402

ROOT = Path(r"D:\dinghuibianyuan\asc-sar")

QD = {0: 0.4184970260827277, 1: 0.660278039711413, 2: 0.4189639255652756}
DD = {0: 0.0006938854930922389, 1: 0.0013238765532150865, 2: 0.0005517006502486765}
SPAN = [(0, 6400), (6400, 8000), (8000, 8400)]

for tag, d in (("int8", "results/scores_ssdd_int8"), ("fp16", "results/scores_ssdd_fp16")):
    fields = load_dir(ROOT / d)
    res_q = []   # 候选A残差
    res_d = []   # 候选A' dequant delta
    res_2m7 = []  # fp16 [-16,-8) 段
    res_2m6 = []  # fp16 [-32,-16) 段
    for st in sorted(fields)[:40]:
        f = fields[st]
        p = f.prob.astype(np.float64)
        raw = np.log(np.clip(p, 1e-300, 1 - 1e-16) / (1 - np.clip(p, 1e-300, 1 - 1e-16)))
        for h, (lo, hi) in enumerate(SPAN):
            r = raw[lo:hi]
            for delta, acc in ((QD[h], res_q), (DD[h], res_d)):
                q = np.abs(r) / delta
                acc.append(np.abs(q - np.round(q)))
            m = (r >= -16) & (r < -8)
            if m.any():
                q = r[m] / 2.0**-7
                res_2m7.append(np.abs(q - np.round(q)))
            m = (r >= -32) & (r < -16)
            if m.any():
                q = r[m] / 2.0**-6
                res_2m6.append(np.abs(q - np.round(q)))
    def stat(a):
        if not a:
            return "n/a"
        a = np.concatenate(a)
        return f"med={np.median(a):.4f} p99={np.quantile(a, .99):.4f} max={a.max():.4f}"
    print(f"[{tag}] A  quant格点(0.4185...): {stat(res_q)}")
    print(f"[{tag}] A' dequant格点(0.0007): {stat(res_d)}")
    print(f"[{tag}] B  fp16 [-16,-8) 2^-7 : {stat(res_2m7)}")
    print(f"[{tag}] B  fp16 [-32,-16) 2^-6: {stat(res_2m6)}")
    print()

# 传导误差量级：int8 vs fp16 同 cell 的 raw logit 差（若 B 成立，这是 dS_new 的实测口径）
f8 = load_dir(ROOT / "results/scores_ssdd_int8")
f16 = load_dir(ROOT / "results/scores_ssdd_fp16")
stems = sorted(set(f8) & set(f16))[:40]
diffs = []
for st in stems:
    p8 = f8[st].prob.astype(np.float64)
    p16 = f16[st].prob.astype(np.float64)
    l8 = np.log(np.clip(p8, 1e-300, 1 - 1e-16) / (1 - np.clip(p8, 1e-300, 1 - 1e-16)))
    l16 = np.log(np.clip(p16, 1e-300, 1 - 1e-16) / (1 - np.clip(p16, 1e-300, 1 - 1e-16)))
    d_ = l8 - l16
    diffs.append(d_)
D = np.concatenate(diffs)
print(f"int8-fp16 raw logit 差: med={np.median(D):+.4f} MAD={np.median(np.abs(D-np.median(D))):.4f} "
      f"p95|.|={np.quantile(np.abs(D), .95):.4f} max|.|={np.abs(D).max():.4f}")
# 背景段（|raw|>8）单独统计
bg = D[np.abs(D) > 1e-9]
print(f"非零差异占比 {(np.abs(D) > 1e-9).mean():.4f}")
