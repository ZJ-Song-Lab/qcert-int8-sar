"""E2 压测集构造（本地，生成后打包推板）。

协议（paper/04 §5）：高杂波子集 100 张 ×3 档人工增益（×1/×2/×3）逼出饱和。
协议偏差（如实记录）：本地无 HRSID，改用 SSDD test 全集（232 张）按
P99 背景分数（FP16 分数场）挑前 100 张高杂波子集作为"强海况"代理。

对比度语义对齐 synth.make_scene 的 `s = s * contrast`（纯增益）：
    img_c = clip(img * c, 0, 255)

产物（--out，默认 asc-sar/data/stress/）：
    c1_<stem>.jpg  ×1（原图重命名拷贝）
    c2_<stem>.jpg  ×2
    c3_<stem>.jpg  ×3
    manifest.json  子集清单 + 每张 P99 + 协议偏差说明

文件名前缀 c{N}_ 保证三档 stem 不冲突，可合入同一 dump 目录；
对比度档同时由 acl_infer --contrast 写进分数场 meta。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", default="../asc-sar/results/scores_ssdd_fp16",
                    help="FP16 分数场目录（挑高杂波子集的依据）")
    ap.add_argument("--images", default="../asc-sar/datasets/ssdd/images/test",
                    help="SSDD test 原图目录")
    ap.add_argument("--out", default="../asc-sar/data/stress", help="压测集输出目录")
    ap.add_argument("--n", type=int, default=100, help="子集张数（协议 100）")
    ap.add_argument("--contrasts", default="1,2,3", help="增益档（逗号分隔）")
    args = ap.parse_args()

    from PIL import Image

    scores = Path(args.scores)
    images = Path(args.images)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    p99 = {}
    for p in sorted(scores.glob("*.npz")):
        z = np.load(p)
        key = "prob" if "prob" in z.files else z.files[0]
        v = np.asarray(z[key], dtype=np.float64).ravel()
        if v.size:
            p99[p.stem] = float(np.quantile(v, 0.99))
    if not p99:
        raise SystemExit(f"[err] 分数场为空: {scores}")

    top = sorted(p99, key=p99.get, reverse=True)[:args.n]
    missing = [s for s in top if not (images / f"{s}.jpg").exists()]
    if missing:
        raise SystemExit(f"[err] 原图缺失: {missing[:5]} ...")

    contrasts = [float(v) for v in args.contrasts.split(",")]
    n_written = 0
    for stem in top:
        img = np.asarray(Image.open(images / f"{stem}.jpg"))
        for c in contrasts:
            tag = f"c{c:g}"
            arr = np.clip(img.astype(np.float64) * c, 0, 255).astype(np.uint8)
            Image.fromarray(arr).save(out / f"{tag}_{stem}.jpg", quality=95)
            n_written += 1

    manifest = {
        "protocol": "paper/04 §5: 100 张高杂波 ×3 档增益（clip 0-255）",
        "deviation": ("本地无 HRSID；改用 SSDD test 全集按 FP16 分数场 P99 "
                      "挑前 100 张作强海况代理（数据驱动，非人工海况标签）"),
        "n": len(top), "contrasts": contrasts,
        "p99_fp16": {s: p99[s] for s in top},
    }
    with open(out / "manifest.json", "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, ensure_ascii=False, indent=1)

    q = np.asarray(sorted(p99.values()))
    sel = np.asarray([p99[s] for s in top])
    print(f"[ok] {n_written} 张 -> {out}")
    print(f"  P99 全集[{q.min():.4f},{q.median() if hasattr(q,'median') else np.median(q):.4f},"
          f"{q.max():.4f}] 子集[{sel.min():.4f},{np.median(sel):.4f},{sel.max():.4f}]")


if __name__ == "__main__":
    main()
