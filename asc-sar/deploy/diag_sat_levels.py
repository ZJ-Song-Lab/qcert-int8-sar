# -*- coding: utf-8 -*-
"""按对比度档位（c1/c2/c3 前缀）分解 neck_sat_stress.json 的 act/qmax 统计。"""
import collections
import json

import numpy as np

d = json.load(open("/root/asc_sar/results/neck_sat_stress.json"))
by = collections.defaultdict(list)
for s, v in d["stems"].items():
    by[s[:2]].append((v["p3"]["act_max"], v["p4"]["act_max"],
                      v["p5"]["act_max"], v["p5"]["qmax"],
                      v["p5"]["act_p999"], v["p3"]["rail"],
                      v["p4"]["rail"], v["p5"]["rail"]))
for k, a in sorted(by.items()):
    a = np.array(a)
    print(f"{k} n={len(a)}  act_max p3/p4/p5 med = "
          f"{np.median(a[:, 0]):.3f}/{np.median(a[:, 1]):.3f}/{np.median(a[:, 2]):.3f}  "
          f"max = {a[:, 0].max():.3f}/{a[:, 1].max():.3f}/{a[:, 2].max():.3f}")
    print(f"   p5 qmax med={np.median(a[:, 3]):.1f} max={a[:, 3].max():.1f}  "
          f"p5 act_p999 med={np.median(a[:, 4]):.3f}  "
          f"rail p3/p4/p5 med = "
          f"{np.median(a[:, 5]):.4f}/{np.median(a[:, 6]):.4f}/{np.median(a[:, 7]):.4f}")
print("calibrated_range:", d["calibrated_range"])
