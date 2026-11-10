# -*- coding: utf-8 -*-
"""本地一次性脚本: 从 SSDD train 分层抽样 256 张校准图, 拷贝到 calib256/ 并记录分层信息。"""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import amct_ptq

ROOT = Path("d:/dinghuibianyuan/asc-sar/datasets/ssdd")

imgs = amct_ptq.collect_images(data_yaml=str(ROOT / "ssdd.yaml"))
picked = amct_ptq.stratified_pick(imgs, 256, 0)

out = ROOT / "calib256"
out.mkdir(exist_ok=True)
for p, _ in picked:
    shutil.copy(p, out / p.name)

meta = {
    "n": len(picked),
    "stratified": True,
    "seed": 0,
    "items": [{"path": p.name, "stratum": s} for p, s in picked],
}
(ROOT / "calib256_stratified.json").write_text(
    json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"train {len(imgs)} -> calib256 {len(list(out.iterdir()))}")
