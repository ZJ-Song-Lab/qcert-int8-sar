# -*- coding: utf-8 -*-
"""生成 232 张 SSDD 测试图缩略拼版，用于人工判读 远洋/近岸 场景类型。
输出: paper/figs/samples/_sheet_{0..4}.png (每张 48 帧, 8x6)
"""
import os

import cv2
import numpy as np

ROOT = r"D:\dinghuibianyuan"
IMG_DIR = os.path.join(ROOT, "neural networks", "asc-sar", "datasets", "ssdd", "images", "test")
OUT = os.path.join(ROOT, "neural networks", "paper", "figs", "samples")

stems = sorted(f[:-4] for f in os.listdir(IMG_DIR) if f.endswith(".jpg"))
THUMB, COLS = 220, 8
PER = 48

for si in range(0, len(stems), PER):
    chunk = stems[si:si + PER]
    rows = (len(chunk) + COLS - 1) // COLS
    sheet = np.full((rows * (THUMB + 22) + 10, COLS * (THUMB + 6) + 10, 3), 255, np.uint8)
    for k, st in enumerate(chunk):
        img = cv2.imread(os.path.join(IMG_DIR, st + ".jpg"))
        img = cv2.resize(img, (THUMB, THUMB))
        r, c = divmod(k, COLS)
        y, x = 10 + r * (THUMB + 22), 10 + c * (THUMB + 6)
        sheet[y:y + THUMB, x:x + THUMB] = img
        cv2.putText(sheet, st, (x + 4, y + THUMB + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    out = os.path.join(OUT, f"_sheet_{si // PER}.png")
    cv2.imwrite(out, sheet, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    print(out, sheet.shape)
