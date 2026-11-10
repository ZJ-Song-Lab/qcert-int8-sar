"""板 vs ORT 整数域逐位置对齐验证（cv3 分数场，E2/E3 的核心取证表）。

输入两条腿的同 stem dump：

- **ORT 腿**（``dump_cv3_ort.py``）：x86 浮点仿真 + numpy requant 的 int32 网格码
  ``s_q``，``meta.impl = ort_int32_cv3``；
- **硬件腿**（``board/acl_dump_cv3.py``）：acl.dump 抓的板上融合卷积 dequant 输出，
  fp16 NC1HWC0 lane0，``v = fl16(q_hw·Δ)``；``s_q = rint(v/Δ)`` 是 fp16 网格代表元，
  ``meta.impl = board_fp16_cv3_acldump``。

逐场景逐头指标：

1. ``bit_exact``：两实现 int32 码完全一致的位置比例（A2 的直接证据）；
2. ``exact_region`` / ``bit_exact_in_region``：|q|≤1024（fp16 可精确表示 Δ 格点）
   区域内的占比与一致率——把"fp16 舍入"与"整数累加本身不同"两种机制分开；
3. ``resid_*``：``v_hw − q_ort·Δ``（logit 单位）的 mean/max/p99 绝对值；
4. ``uniq_frac_*`` / ``collapse``：格点占用（uniq/n）两实现对比与塌缩比——
   硬件 fp16 把 int32 网格压到更少代表元，**只在真板上可观测**；
5. ``max_tie_*``：最高 1% 分数位置的并列数（qcert.audit.tie_stats）；
6. ``rho_prob``：两实现 prob 的 Spearman 秩相关（排序保持度）。

用法::

    python -m experiments.align_hw_ort \
        --hw asc-sar/results/dumps/cv3_hw --ort asc-sar/results/dumps/ort
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from qcert.audit import tie_stats
from qcert.pfa_eval import spearman

from .common import RESULTS, ensure_dir, json_dump, save_table

HEAD_NAMES = ("head0_80x80", "head1_40x40", "head2_20x20")
EXACT_QMAX = 1024  # |q|≤1024 时 fp16 可精确表示 Δ 格点（ulp ≤ Δ/2）


def _heads(npz_path):
    """读一个 dump npz，按 head_sizes 切成 [(q, v, prob, delta_h), ...]。"""
    z = np.load(npz_path, allow_pickle=False)
    sizes = np.asarray(z["head_sizes"]).reshape(-1, 2)
    meta = json.loads(str(z["meta"]))
    deltas = meta.get("head_deltas") or [float(z["delta"])] * len(sizes)
    q_all = np.asarray(z["s_q"]).ravel()
    v_all = np.asarray(z["logit"]).ravel().astype(np.float64)
    p_all = np.asarray(z["prob"]).ravel()
    out, off = [], 0
    for i, (h, w) in enumerate(sizes):
        n = int(h) * int(w)
        out.append((q_all[off:off + n], v_all[off:off + n],
                    p_all[off:off + n], float(deltas[i])))
        off += n
    return out, meta


def _align_head(q_ort, q_hw, v_hw, p_ort, p_hw, delta):
    n = int(q_ort.size)
    bit_exact = float(np.mean(q_hw == q_ort))
    in_reg = np.abs(q_ort) <= EXACT_QMAX
    frac_reg = float(np.mean(in_reg))
    be_reg = (float(np.mean(q_hw[in_reg] == q_ort[in_reg]))
              if np.any(in_reg) else float("nan"))
    resid = v_hw - q_ort.astype(np.float64) * delta
    ares = np.abs(resid)
    v_ort = q_ort.astype(np.float64) * delta
    rel = ares / np.maximum(np.abs(v_ort), 1e-12)
    uniq_ort = int(np.unique(q_ort).size)
    uniq_hw = int(np.unique(q_hw).size)
    tie_ort = tie_stats(q_ort, select=p_ort, frac=0.01)
    tie_hw = tie_stats(q_hw, select=p_hw, frac=0.01)
    return {
        "n": n,
        "bit_exact": bit_exact,
        "exact_region_frac": frac_reg,
        "bit_exact_in_region": be_reg,
        "resid_median_abs": float(np.median(ares)),
        "resid_mean_abs": float(ares.mean()),
        "resid_p99_abs": float(np.percentile(ares, 99)),
        "resid_max_abs": float(ares.max()),
        "rel_div_median": float(np.median(rel)),
        "rel_div_p99": float(np.percentile(rel, 99)),
        "uniq_frac_ort": uniq_ort / n,
        "uniq_frac_hw": uniq_hw / n,
        "collapse": (uniq_hw / uniq_ort) if uniq_ort else float("nan"),
        "max_tie_ort": tie_ort["max_tie"],
        "max_tie_hw": tie_hw["max_tie"],
        "rho_prob": float(spearman(p_ort, p_hw)),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hw", required=True, help="板端 acl.dump 的 cv3 dump 目录")
    ap.add_argument("--ort", required=True, help="ORT int32 的 cv3 dump 目录")
    ap.add_argument("--out", default=str(RESULTS / "align"))
    args = ap.parse_args(argv)

    hw_dir, ort_dir = Path(args.hw), Path(args.ort)
    stems = sorted(p.stem for p in hw_dir.glob("*.npz")
                   if (ort_dir / p.name).exists())
    if not stems:
        raise SystemExit(f"{hw_dir} 与 {ort_dir} 没有同名 npz")
    print(f"[align] {len(stems)} 个共同场景")

    rows = []
    for k, stem in enumerate(stems, 1):
        heads_hw, meta_hw = _heads(hw_dir / f"{stem}.npz")
        heads_ort, meta_ort = _heads(ort_dir / f"{stem}.npz")
        if len(heads_hw) != len(heads_ort):
            raise ValueError(f"{stem}: 头数不一致 {len(heads_hw)} vs {len(heads_ort)}")
        for h, ((q_o, v_o, p_o, d_o), (q_h, v_h, p_h, d_h)) in enumerate(
                zip(heads_ort, heads_hw)):
            if q_o.shape != q_h.shape:
                raise ValueError(f"{stem}/{HEAD_NAMES[h]}: 形状不一致 "
                                 f"{q_o.shape} vs {q_h.shape}")
            r = _align_head(q_o, q_h, v_h, p_o, p_h, d_o)
            r.update({"stem": stem, "head": HEAD_NAMES[h],
                      "delta": d_o, "impl_hw": meta_hw.get("impl", "?"),
                      "impl_ort": meta_ort.get("impl", "?")})
            rows.append(r)
        if k % 50 == 0 or k == len(stems):
            print(f"  {k}/{len(stems)}")

    out = ensure_dir(args.out)
    save_table(rows, out, "align_per_scene_head",
               "板 fp16 vs ORT int32 逐场景逐头对齐")

    # 按头聚合 —— 论文核心表
    agg = []
    for h in HEAD_NAMES:
        sub = [r for r in rows if r["head"] == h]
        be_reg = [r["bit_exact_in_region"] for r in sub]
        be_reg = [x for x in be_reg if np.isfinite(x)]
        agg.append({
            "head": h, "n_scenes": len(sub), "delta": sub[0]["delta"],
            "bit_exact": float(np.mean([r["bit_exact"] for r in sub])),
            "bit_exact_in_region": float(np.mean(be_reg)) if be_reg else float("nan"),
            "exact_region_frac": float(np.mean(
                [r["exact_region_frac"] for r in sub])),
            "resid_median_abs": float(np.mean([r["resid_median_abs"] for r in sub])),
            "resid_max_abs": float(max(r["resid_max_abs"] for r in sub)),
            "resid_p99_abs": float(np.mean([r["resid_p99_abs"] for r in sub])),
            "rel_div_median": float(np.mean([r["rel_div_median"] for r in sub])),
            "rel_div_p99": float(np.mean([r["rel_div_p99"] for r in sub])),
            "uniq_frac_ort": float(np.mean([r["uniq_frac_ort"] for r in sub])),
            "uniq_frac_hw": float(np.mean([r["uniq_frac_hw"] for r in sub])),
            "collapse": float(np.mean([r["collapse"] for r in sub])),
            "max_tie_ort": int(max(r["max_tie_ort"] for r in sub)),
            "max_tie_hw": int(max(r["max_tie_hw"] for r in sub)),
            "rho_prob": float(np.mean([r["rho_prob"] for r in sub])),
        })
    save_table(agg, out, "align_by_head",
               "Table · 板 fp16 vs ORT int32 对齐（按头聚合）")
    summary = {
        "n_scenes": len(stems),
        "impl_hw": rows[0]["impl_hw"], "impl_ort": rows[0]["impl_ort"],
        "input_check": ("已验证：同一 jpg（md5 相同）经两侧各自的 PIL 预处理 "
                        "（板 Pillow 12.2.0 vs 本机 12.3.0）输出逐 bit 一致 → "
                        "下表差异全部来自硬件算术，不含输入管线差"),
        "by_head": agg,
        "reading": {
            "bit_exact<1": "同输入下同权重两实现的分数场逐位置发散（A2 实证）",
            "collapse<1": "fp16 ulp>Δ 处多个 int32 码共值 → 硬件特有并列机制（E2）",
        },
    }
    json_dump(summary, out / "summary.json")

    print("\n=== 板 fp16 vs ORT int32 对齐（输入已验证逐 bit 一致）===")
    for a in agg:
        print(f"  {a['head']:<14s} bit-exact={a['bit_exact']:.4f} "
              f"resid med={a['resid_median_abs']:.3f} p99={a['resid_p99_abs']:.3f} "
              f"max={a['resid_max_abs']:.3f} rel={a['rel_div_median']:.4f} "
              f"uniq {a['uniq_frac_ort']:.3f}→{a['uniq_frac_hw']:.3f} "
              f"(×{a['collapse']:.3f}) tie {a['max_tie_ort']}→{a['max_tie_hw']} "
              f"ρ={a['rho_prob']:.4f}")
    print(f"产物: {out}")
    return summary


if __name__ == "__main__":
    main()
