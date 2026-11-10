# -*- coding: utf-8 -*-
"""板上一次性脚本: 对比 FP16 与 INT8 两个 dump 的分数场(数值保真冒烟)。

分数场 = 逐网格 max_c sigmoid(cls)（N=8400）。全场 99% 以上是 ~1e-6 的
背景网格, 直接对全场做 Pearson 会被背景微小值的相对重排主导(度量失真,
首跑实测全场 corr 仅 0.566 而 mean|d| 仅 8.8e-5), 故分口径统计:

  1) 全场绝对误差 mean|d|
  2) 前景子集 (fp16>0.1, 真正会进 NMS 的量级): mean|d| / max|d| / corr
  3) 每图 top-1 候选置信度 + argmax 同网格率(直接对应检测输出)
  4) conf>=0.25 候选总数(默认检测阈值下的检出量)
  5) 全局 max|d| 定位(哪张图/哪个网格/两个值各多少)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/root/code")
from qcert.scores import load_dir

A = Path("/root/asc_sar/results/scores_fp16")
B = Path("/root/asc_sar/results/scores_int8")
FG_TH = 0.1     # 前景口径: FP16 分数阈值
CONF_TH = 0.25  # 默认 conf 阈值

da, db = load_dir(A), load_dir(B)
stems = sorted(set(da) & set(db))
print(f"对齐 {len(stems)} 张 / fp16 {len(da)} / int8 {len(db)}")

all_mad = n_all = 0
fg_mad = n_fg = 0
fg_a, fg_b = [], []
top1_a, top1_b = [], []
argmax_same = n_used = 0
cnt_a = cnt_b = 0
gmax = (0.0, "", -1, 0.0, 0.0)  # diff, stem, idx, fp16, int8
for s in stems:
    a = da[s].prob.astype(np.float64)
    b = db[s].prob.astype(np.float64)
    if a.shape != b.shape or a.size == 0:
        continue
    d = np.abs(a - b)
    all_mad += d.sum()
    n_all += d.size
    m = int(d.argmax())
    if d[m] > gmax[0]:
        gmax = (float(d[m]), s, m, float(a[m]), float(b[m]))
    fg = a > FG_TH
    if fg.any():
        fg_mad += d[fg].sum()
        n_fg += int(fg.sum())
        fg_a.append(a[fg])
        fg_b.append(b[fg])
    top1_a.append(float(a.max()))
    top1_b.append(float(b.max()))
    if int(a.argmax()) == int(b.argmax()):
        argmax_same += 1
    cnt_a += int((a >= CONF_TH).sum())
    cnt_b += int((b >= CONF_TH).sum())
    n_used += 1

ta, tb = np.asarray(top1_a), np.asarray(top1_b)
td = np.abs(ta - tb)
fa = np.concatenate(fg_a) if fg_a else np.zeros(1)
fb = np.concatenate(fg_b) if fg_b else np.zeros(1)
fd = np.abs(fa - fb)
fg_corr = (np.corrcoef(fa, fb)[0, 1] if fa.size > 1 and fa.std() > 0
           else float("nan"))

print(f"[1] 全场 (n={n_all}): mean|d| = {all_mad / max(n_all, 1):.3e}")
print(f"[2] 前景 fp16>{FG_TH} (n={n_fg}): mean|d| = {fg_mad / max(n_fg, 1):.4f}"
      f"  max|d| = {fd.max():.4f}  corr = {fg_corr:.6f}")
print(f"[3] top-1 置信度 (n={n_used}): fp16 mean={ta.mean():.4f}"
      f"  int8 mean={tb.mean():.4f}  mean|d|={td.mean():.4f}"
      f"  max|d|={td.max():.4f}")
print(f"    top-1 argmax 同网格率: {argmax_same}/{n_used} = {argmax_same / n_used:.2%}")
print(f"[4] conf>={CONF_TH} 候选总数: fp16={cnt_a}  int8={cnt_b}")
print(f"[5] 全场 max|d| = {gmax[0]:.4f} @ {gmax[1]} 网格#{gmax[2]}"
      f"  fp16={gmax[3]:.4f}  int8={gmax[4]:.4f}")
