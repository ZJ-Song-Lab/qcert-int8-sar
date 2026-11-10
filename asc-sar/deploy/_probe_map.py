# -*- coding: utf-8 -*-
"""核查 map.json int8=0.9798 与 compare_fp16_int8.log 候选 35->2 的矛盾。

1) 分数场实测: scores_int8 vs scores_fp16 (000001 + 000969, 与 compare log 同源)
2) GT 框内分数: 框映射到 letterbox 坐标后取框内网格 max
"""
import json
from pathlib import Path

import numpy as np

R = Path("/root/asc_sar/results")


def load(stem, sub):
    p = R / sub / f"{stem}.npz"
    if not p.exists():
        return None
    z = np.load(str(p), allow_pickle=True)
    meta = json.loads(str(np.asarray(z["meta"]).ravel()[0])) if "meta" in z else {}
    return {
        "prob": np.asarray(z["prob"]).ravel().astype(np.float32),
        "centers": np.asarray(z["centers"], dtype=np.float32) if "centers" in z else None,
        "meta": meta,
    }


def boxes_in_letterbox(meta):
    """GT 框(原图坐标) -> letterbox 坐标矩形列表。"""
    iw, ih = meta.get("iw", 1), meta.get("ih", 1)
    ratio = meta.get("ratio", 1.0)
    pad = meta.get("pad", [0, 0])  # (pad_x, pad_y)
    out = []
    for b in meta.get("boxes", []):
        x1, y1, x2, y2 = b[:4]
        out.append((x1 * ratio + pad[0], y1 * ratio + pad[1],
                    x2 * ratio + pad[0], y2 * ratio + pad[1]))
    return out


def stats(tag, d):
    prob, centers, meta = d["prob"], d["centers"], d["meta"]
    n25 = int((prob >= 0.25).sum())
    n50 = int((prob >= 0.50).sum())
    top5 = np.sort(prob)[-5:][::-1]
    print(f"  {tag}: n={prob.size} max={prob.max():.4f} "
          f">=0.25: {n25}  >=0.50: {n50}")
    print(f"     top5 = {np.array2string(top5, precision=4)}")
    if centers is not None and meta.get("boxes"):
        rects = boxes_in_letterbox(meta)
        cx, cy = centers[:, 0], centers[:, 1]
        best = []
        for (x1, y1, x2, y2) in rects:
            m = (cx >= x1) & (cx <= x2) & (cy >= y1) & (cy <= y2)
            if m.any():
                best.append(float(prob[m].max()))
        if best:
            print(f"     GT框内max = {np.array2string(np.array(best), precision=4)} "
                  f"(共{len(best)}框)")
    print(f"     meta: impl={meta.get('impl')} model={meta.get('model')} "
          f"nc={meta.get('nc')} boxes={len(meta.get('boxes', []))}")


def main():
    for stem in ("000001", "000969"):
        print("=" * 64)
        print(f"[{stem}]")
        f16 = load(stem, "scores_fp16")
        i8 = load(stem, "scores_int8")
        if f16 is None or i8 is None:
            print(f"  [miss] fp16={'OK' if f16 else 'MISS'} int8={'OK' if i8 else 'MISS'}")
            continue
        stats("fp16", f16)
        stats("int8", i8)
        # 同网格相关系数(全场)
        if f16["prob"].size == i8["prob"].size:
            a, b = f16["prob"].astype(np.float64), i8["prob"].astype(np.float64)
            corr = float(np.corrcoef(a, b)[0, 1])
            print(f"     全场 corr(fp16,int8) = {corr:.4f}")

    # ---- 全量快速盘点: 两目录所有 npz 的 >=0.25 计数 ----
    print("=" * 64)
    print("[全量] conf>=0.25 网格点总数")
    for sub in ("scores_fp16", "scores_int8"):
        tot, files = 0, 0
        for p in sorted((R / sub).glob("*.npz")):
            z = np.load(str(p), allow_pickle=True)
            tot += int((np.asarray(z["prob"]) >= 0.25).sum())
            files += 1
        print(f"  {sub}: {files} 个文件, >=0.25 总点数 = {tot}")


if __name__ == "__main__":
    main()
