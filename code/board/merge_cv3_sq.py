"""把 acl.dump 抓到的 cv3 int32 网格码（s_q）/ fp16 原值（s_hw）注入 score dump npz。

背景：int8 OM 的最终输出是 AscendDequant 后的 fp32，``acl_infer.dump_score_field``
落盘的 npz 没有 s_q，``SaturationAuditor`` 无法审计。``acl_dump_cv3.py`` 抓到的
cv3 终卷积输出（``s_q = rint(v/Δ)`` int32，``s_hw`` fp16 硬件真值）正是判决网格
的整数域原值，按 stem 合并进 score dump 后即可跑 run_e2 / run_e3 --hw。

用法::

    python code/board/merge_cv3_sq.py \
        --cv3-dir asc-sar/results/cv3_stress \
        --in-dir  asc-sar/results/dumps_stress/int8 \
        --out-dir asc-sar/results/dumps_stress/int8_sq
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cv3-dir", required=True, help="acl_dump_cv3.py 的输出目录")
    ap.add_argument("--in-dir", required=True, help="原 score dump 目录（int8）")
    ap.add_argument("--out-dir", required=True, help="合并后输出目录")
    args = ap.parse_args()

    cv3_dir, in_dir = Path(args.cv3_dir), Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_ok, n_skip, diffs = 0, 0, []
    for cv3_path in sorted(cv3_dir.glob("*.npz")):
        stem = cv3_path.stem
        in_path = in_dir / f"{stem}.npz"
        if not in_path.exists():
            print(f"[skip] {stem}: in-dir 无对应 npz")
            n_skip += 1
            continue
        with np.load(str(cv3_path)) as c, np.load(str(in_path)) as z:
            payload = {k: z[k] for k in z.files}          # 保留原键（prob/logit/meta/...）
            payload["s_q"] = c["s_q"].astype(np.int32)     # cv3 int32 网格码
            payload["s_hw"] = c["s_hw"]                    # fp16 硬件原值
            # 一致性检查：sigmoid(cv3 logit) vs OM 输出 prob（同一 int8.om 同输入）
            p_cv3 = 1.0 / (1.0 + np.exp(-np.clip(
                c["logit"].astype(np.float64), -60, 60)))
            diffs.append(float(np.max(np.abs(p_cv3 - z["prob"].astype(np.float64)))))
            # meta 里标注 s_q 来源
            meta = json.loads(str(np.asarray(z["meta"]).ravel()[0]))
            meta["s_q_source"] = "cv3 acl.dump int32 grid (s_q=rint(v/Δ), s_hw=fp16)"
            payload["meta"] = np.array(json.dumps(meta, ensure_ascii=False))
        np.savez_compressed(out_dir / f"{stem}.npz", **payload)
        n_ok += 1

    print(f"[merge] 合并 {n_ok} 个 -> {out_dir}；跳过 {n_skip}")
    if diffs:
        diffs = np.asarray(diffs)
        print(f"[check] sigmoid(cv3_logit) vs OM prob: max|Δ| mean={diffs.mean():.2e} "
              f"max={diffs.max():.2e}")


if __name__ == "__main__":
    main()
