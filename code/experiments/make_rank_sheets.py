# -*- coding: utf-8 -*-
"""按 coast_score 排序的候选拼版：TOP48 (rank 1-48) 与次高 48 (rank 49-96)。"""
import csv
import os

import cv2
import numpy as np

ROOT = r"D:\dinghuibianyuan"
IMG = os.path.join(ROOT, "neural networks", "asc-sar", "datasets", "ssdd", "images", "test")
CSV = os.path.join(ROOT, "neural networks", "code", "results", "e7", "e7_scene_type.csv")
OUT = os.path.join(ROOT, "neural networks", "paper", "figs", "samples")

rows = sorted(csv.DictReader(open(CSV, encoding="utf-8")), key=lambda r: -float(r["coast_score"]))
THUMB, COLS, PER = 220, 8, 48

for si, chunk in enumerate([rows[:48], rows[48:96]]):
    n = len(chunk)
    r_ = (n + COLS - 1) // COLS
    sheet = np.full((r_ * (THUMB + 22) + 10, COLS * (THUMB + 6) + 10, 3), 255, np.uint8)
    for k, r in enumerate(chunk):
        img = cv2.imread(os.path.join(IMG, r["stem"] + ".jpg"))
        img = cv2.resize(img, (THUMB, THUMB))
        rr, c = divmod(k, COLS)
        y, x = 10 + rr * (THUMB + 22), 10 + c * (THUMB + 6)
        sheet[y:y + THUMB, x:x + THUMB] = img
        cv2.putText(sheet, f"{r['stem']} {float(r['coast_score']):.0f}",
                    (x + 2, y + THUMB + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    p = os.path.join(OUT, f"_rank_{si}.png")
    cv2.imwrite(p, sheet, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    print(p, sheet.shape)
