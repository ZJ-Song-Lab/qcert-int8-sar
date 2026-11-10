"""诊断 Neck 饱和率 0.98 异常：单层 fp vs scale.json delta 的分布对账。"""
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path("/root/asc_sar")


def load_scale(conv):
    ent = json.load(open(ROOT / "models/yolo11n_ssdd_int8_scale.json"))["entries"]
    pat = re.compile(re.escape(conv) + r"_dequant\[(\d+)\]$")
    picks = {int(m.group(1)): x for x in ent if (m := pat.match(x["name"]))}
    ks = sorted(picks)
    delta = np.array([picks[k]["delta"] for k in ks])
    return delta


def main():
    man = json.load(open(ROOT / "results/layers_stress/manifest.json"))
    for idx, conv in ((48, "/model.16/cv2/conv/Conv_quant"),
                      (61, "/model.19/cv2/conv/Conv_quant"),
                      (79, "/model.22/cv2/conv/Conv_quant")):
        e = [m for m in man if m["index"] == idx][0]
        a = np.load(ROOT / "results/layers_stress" / e["file"]).astype(np.float64)
        delta = load_scale(conv)
        c = a.shape[1]
        print(f"\n=== idx {idx} {conv} ===")
        print("file", e["file"], "shape", a.shape, "n_delta", len(delta))
        print("delta min/med/max: %.3e %.3e %.3e"
              % (delta.min(), np.median(delta), delta.max()))
        d = delta[:c].reshape(1, c, 1, 1)
        q = np.rint(a / d)
        print("q: |q| med=%.1f  frac>=127=%.4f  frac<=-128=%.4f"
              % (np.median(np.abs(q)), np.mean(q >= 127), np.mean(q <= -128)))
        # 每通道 |q| 中位数与 delta 对账：找 sat 最重的通道
        per_ch = np.median(np.abs(q[0]), axis=(1, 2))
        worst = np.argsort(-per_ch)[:5]
        for ch in worst:
            print("  ch %d: delta=%.3e  |q|med=%.1f  fp_p99=%.3e"
                  % (ch, delta[ch], per_ch[ch], np.percentile(a[0, ch], 99)))
        # 全局 fp 分布
        print("fp: p1=%.3e p50=%.3e p99=%.3e max=%.3e"
              % (np.percentile(a, 1), np.percentile(a, 50),
                 np.percentile(a, 99), np.abs(a).max()))
        # 反过来：如果 fp/delta 全体 >> 127，delta 相对多大才合理
        print("implied delta (p99 fp / 127): %.3e"
              % (np.percentile(np.abs(a), 99) / 127.0))


if __name__ == "__main__":
    main()
