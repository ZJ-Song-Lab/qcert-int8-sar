# -*- coding: utf-8 -*-
"""抽查 SSDD dump 的 npz 完整性（qcert ScoreField 格式）。"""
import json
from pathlib import Path

import numpy as np

for sub in ("scores_ssdd_fp16", "scores_ssdd_int8"):
    p = Path("/root/asc_sar/results") / sub / "000969.npz"
    z = np.load(str(p), allow_pickle=True)
    m = json.loads(str(np.asarray(z["meta"]).ravel()[0]))
    hs = z["head_sizes"].tolist() if "head_sizes" in z.files else None
    print(f"{sub}: files={z.files}")
    print(f"  head_sizes={hs}  impl={m.get('impl')}  imgsz={m.get('imgsz')}  "
          f"boxes={len(m.get('boxes', []))}  nc={m.get('nc')}")
    if "s_q" in z.files:
        print(f"  s_q dtype={z['s_q'].dtype} delta={m.get('delta')}")
    n = len(list((Path('/root/asc_sar/results') / sub).glob("*.npz")))
    print(f"  目录文件数 = {n}")
