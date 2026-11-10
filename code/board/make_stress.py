"""板端：从 SSDD 测试集生成对比度压测集（E2 饱和主线的 ×2/×3 档）。

变换：``I' = clip((I - mean) * c + mean, 0, 255)``（mean 为全图均值），
与 experiments/synth.py 的 contrast 语义一致（像素对比度倍增）。
标签文件直接复制（几何不变，GT 框不变）。

用法（板上）::

    python3 /root/code/board/make_stress.py \
        --images /root/asc_sar/data/images --labels /root/asc_sar/data/labels \
        --out /root/asc_sar/data --contrasts 2 3
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--contrasts", type=float, nargs="+", default=[2.0, 3.0])
    args = ap.parse_args(argv)

    from PIL import Image

    imgs = sorted(p for p in Path(args.images).iterdir()
                  if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    assert imgs, f"图像目录为空: {args.images}"
    lab_dir = Path(args.labels)

    for c in args.contrasts:
        tag = f"c{c:g}"
        img_out = Path(args.out) / f"images_{tag}"
        lab_out = Path(args.out) / f"labels_{tag}"
        img_out.mkdir(parents=True, exist_ok=True)
        lab_out.mkdir(parents=True, exist_ok=True)
        n_lab = 0
        for p in imgs:
            a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)
            m = float(a.mean())
            b = np.clip((a - m) * c + m, 0, 255).astype(np.uint8)
            Image.fromarray(b).save(img_out / p.name, quality=95)
            lab = lab_dir / f"{p.stem}.txt"
            if lab.exists():
                shutil.copyfile(lab, lab_out / lab.name)
                n_lab += 1
        print(f"[stress] ×{c:g}: {len(imgs)} 图 -> {img_out}（标签 {n_lab}）")
    return 0


if __name__ == "__main__":
    main()
