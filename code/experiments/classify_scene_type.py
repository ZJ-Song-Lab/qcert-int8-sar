# -*- coding: utf-8 -*-
"""远洋/近岸场景分类（SSDD 232 张测试图）
- 人工判读标签（基于缩略拼版目视）：含海岸线/陆地 => nearshore，开阔海域 => offshore
- 陆地各向异性启发式 coast_score = std(行均值)+std(列均值)，仅作交叉参考
输出: code/results/e7/e7_scene_type.csv
"""
import csv
import os

import cv2
import numpy as np

ROOT = r"D:\dinghuibianyuan"
IMG_DIR = os.path.join(ROOT, "neural networks", "asc-sar", "datasets", "ssdd", "images", "test")
OUT = os.path.join(ROOT, "neural networks", "code", "results", "e7", "e7_scene_type.csv")

# 近岸判定：coast_score TOP-40（2026-11-09 拼版目视确认，均含明显海岸线/陆地；rank 49-96 为开阔海面）
# coast_score = std(行均值)+std(列均值)，陆地大块亮/暗结构造成强各向异性
NEARSHORE = {
    "001029", "001159", "001089", "001019", "001069", "000221", "000749", "001039",
    "000241", "001049", "001041", "000231", "001091", "001071", "000349", "000351",
    "000741", "000219", "000229", "000641", "000751", "000251", "000359", "000069",
    "000489", "000759", "000239", "001059", "001079", "000039", "000729", "000841",
    "001151", "001111", "000021", "000639", "000249", "001131", "000409", "001101",
}


def coast_score(path):
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    g = cv2.resize(g, (256, 256))
    return float(np.std(g.mean(axis=1)) + np.std(g.mean(axis=0)))


rows = []
for f in sorted(os.listdir(IMG_DIR)):
    if not f.endswith(".jpg"):
        continue
    stem = f[:-4]
    rows.append({
        "stem": stem,
        "scene_type": "nearshore" if stem in NEARSHORE else "offshore",
        "coast_score": round(coast_score(os.path.join(IMG_DIR, f)), 3),
    })

with open(OUT, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=["stem", "scene_type", "coast_score"])
    w.writeheader()
    w.writerows(rows)

ns = [r for r in rows if r["scene_type"] == "nearshore"]
off = [r for r in rows if r["scene_type"] == "offshore"]
print(f"total={len(rows)} nearshore={len(ns)} offshore={len(off)}")
print("coast_score nearshore median:", np.median([r["coast_score"] for r in ns]))
print("coast_score offshore median:", np.median([r["coast_score"] for r in off]))
print("->", OUT)
