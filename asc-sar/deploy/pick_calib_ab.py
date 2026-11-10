# -*- coding: utf-8 -*-
"""E5 设置 1（校准集 A→B）本地一次性脚本：从 SSDD train 的对比度
calm / rough 单层各抽 256 张，拷到 calib_calm256/ 与 calib_rough256/。

与 pick_calib.py 的区别：那是三层等量混合（calib256，日常 PTQ 用）；
这里刻意制造两个"校准分布错位"的单层校准集——A（calm，低对比度平坦
海面）与 B（rough，高对比度强杂波），供 paper/04 §8 设置 1 验证
"换校准集构建时整数阈值能否靠 scale 比值搬运"。

分层定义与 amct_ptq.stratified_pick 完全一致（全体 train 的 contrast
三分位 q1/q2），保证 calm/rough 层与历史 calib256 的层边界相同。
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import amct_ptq

ROOT = Path("d:/dinghuibianyuan/asc-sar/datasets/ssdd")
N = 256
SEED = 0

imgs = amct_ptq.collect_images(data_yaml=str(ROOT / "ssdd.yaml"))
vals = np.asarray([amct_ptq.contrast_of(p) for p in imgs], dtype=np.float64)
q1, q2 = np.quantile(vals, [1 / 3, 2 / 3])
idx = np.digitize(vals, [q1, q2])
print(f"[calib-ab] train {len(imgs)} 张，三分位 {q1:.4f}/{q2:.4f}，"
      f"各层 {[int((idx == i).sum()) for i in range(3)]}")

rng = np.random.default_rng(SEED)
for stratum, s in (("calm", 0), ("rough", 2)):
    pool = [i for i in range(len(imgs)) if idx[i] == s]
    order = rng.permutation(len(pool))[:N]
    picked = sorted(pool[j] for j in order)
    out = ROOT / f"calib_{stratum}{N}"
    out.mkdir(exist_ok=True)
    for i in picked:
        shutil.copy(imgs[i], out / imgs[i].name)
    meta = {
        "n": len(picked), "stratum": stratum, "seed": SEED,
        "quantiles": [float(q1), float(q2)], "pool_size": len(pool),
        "items": [{"path": imgs[i].name, "stratum": stratum,
                   "contrast": float(vals[i])} for i in picked],
    }
    (ROOT / f"calib_{stratum}{N}.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    cs = [float(vals[i]) for i in picked]
    print(f"[calib-ab] {stratum}{N}: pool {len(pool)} -> 抽 {len(picked)} 张，"
          f"contrast [{min(cs):.4f}, {max(cs):.4f}]，中位 {float(np.median(cs)):.4f}"
          f" -> {out}")
