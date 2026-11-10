"""E2 · 饱和审计：逐层/逐场饱和率、最高 1% 的并列数、与漏检的相关。

协议依据 paper/04 §5。核心主张（02 §4.2）：高对比度杂波把分数推到 int8 上界
→ 大量并列 → 排序塌陷 → 分位数阈值被推到上界 → P_fa 爆炸或 P_d.
**FP16 没有这个机制**，所以这一组只有定点板能观测。

两种跑法：

板端（真数据）::

    python -m experiments.run_e2 --fp32 dumps/fp32 --int8 dumps/int8 \
        --labels asc-sar/datasets/ssdd/labels/test --layers dumps/layers

离线（无板子，M0 自检 / 走通代码路径）::

    python -m experiments.run_e2 --synth --n-scenes 12

``--neck-sat-json`` 是 ``neck_sat.py`` 的输出（``{"stems": {stem: {p3/p4/p5:
{sat,...}}}``），提供 Neck 三支路 int8 特征层的饱和率；不给就留空。

压测集 v2（make_stress_set_v2.py）剂量由文件名前缀 ``c{1,2,3}_`` 承载
（meta.contrast 恒 1.0），用 ``--dose-from stem`` 按前缀取剂量档。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from qcert.audit import SaturationAuditor, correlation_with_r, e2_row
from qcert.pfa_eval import boxes_to_letterbox, points_in_boxes

from .common import (RESULTS, ensure_dir, group_of, gt_for, json_dump,
                     load_pairs, save_table, scene_pfa_pair)
from .synth import make_pair

CONTRASTS = (1.0, 2.0, 3.0)
SEAS = ("calm", "moderate", "rough")


def _gt_mask(field, boxes):
    if boxes is None or len(boxes) == 0:
        return None
    lb = boxes_to_letterbox(boxes, field.meta)
    centers = field.centers
    if centers is None:  # dump 未落盘 centers 时按网格重建（与 omega_for 同口径）
        from .common import cell_centers
        centers = cell_centers(field.head_sizes,
                               int(field.meta.get("imgsz", 640)))
    return points_in_boxes(centers, lb, 0.0)


def _miss(prob, lam, mask):
    if mask is None or not np.any(mask):
        return float("nan")
    return float(1.0 - np.mean(np.asarray(prob)[mask] > float(lam)))


# ------------------------------------------------------------------ 板端
def _dose(args, field, stem):
    """剂量档：默认取 meta.contrast；--dose-from stem 时按 c{N}_ 前缀解析
    （压测集 v2 协议，meta.contrast 恒 1.0）。"""
    if args.dose_from == "stem":
        return float(stem.split("_", 1)[0][1:])
    return float(field.meta.get("contrast", 1.0))


def _load_neck_json(path):
    if not path:
        return None
    with open(path, encoding="utf-8") as fp:
        return json.load(fp).get("stems", {})


def _neck_sat(neck_stems, stem):
    """Neck 三支路 int8 饱和率 {p3,p4,p5}；缺数据返回 None。"""
    if not neck_stems:
        return None
    per = neck_stems.get(stem)
    if not per:
        return None
    return {tag: float(per[tag]["sat"]) for tag in ("p3", "p4", "p5")
            if tag in per}


def run_dumps(args, out):
    pairs = load_pairs(args.fp32, args.int8)
    aud = SaturationAuditor(bits=8, frac=0.01)
    neck_stems = _load_neck_json(args.neck_sat_json)
    rows, by_contrast, r_by_contrast, miss_by_contrast = [], {}, {}, {}
    neck_lc = {}  # (layer, contrast) -> [sat]
    for stem, ref, test in pairs:
        boxes = gt_for(stem, args.labels, ref.meta) if args.labels else None
        gm = _gt_mask(test, boxes)
        r = scene_pfa_pair(ref, test, boxes, args.alpha, args.guard_px,
                           args.block_px, args.keep, args.seed)
        con = _dose(args, ref, stem)
        r["contrast"], r["sea"] = con, group_of(ref, "sea")
        r["miss_test"] = _miss(test.prob, r["lambda_test"], gm)
        rows.append(r)
        aud.add_from_field(test, name="score", sea=r["sea"], contrast=con)
        r_by_contrast.setdefault(con, []).append(r["R"])
        miss_by_contrast.setdefault(con, []).append(r["miss_test"])

        neck = _neck_sat(neck_stems, stem)
        if neck is not None:
            by_contrast.setdefault(con, []).append(neck.get("p5", np.nan))
            for tag, sat in neck.items():
                neck_lc.setdefault((tag, con), []).append(sat)

    # 从审计明细里按对比度汇总
    sat = {}
    for row in aud.table():
        sat.setdefault(float(row["contrast"]), []).append(row["sat"])
    sat_by_c = {c: float(np.nanmean(v)) for c, v in sat.items()}
    tie_by_c = {}
    for row in aud.table():
        tie_by_c.setdefault(float(row["contrast"]), []).append(row["max_tie"])

    table7 = [
        e2_row(c, sum(1 for r in rows if r["contrast"] == c),
               sat_by_c.get(c, float("nan")),
               float(np.nanmean(by_contrast.get(c, [np.nan]))),
               int(max(tie_by_c.get(c, [0]))),
               float(np.nanmean([x for x in r_by_contrast.get(c, [np.nan])])),
               float(np.nanmean(miss_by_contrast.get(c, [np.nan]))))
        for c in sorted(sat_by_c) or sorted(r_by_contrast)
    ]
    neck_layers = {tag: {c: float(np.nanmean(v))
                         for (t, c), v in neck_lc.items() if t == tag}
                   for tag in ("p3", "p4", "p5")}
    return rows, aud.table(), table7, sat_by_c, r_by_contrast, (out, neck_layers)


# ------------------------------------------------------------------ 离线
def run_synth(args, out):
    seas = SEAS if args.sea_states is None else tuple(args.sea_states)
    # 标定 scale 只在正常对比度上取一次，之后固定（AMCT 语义）
    cal = [make_pair(seed=9000 + i, contrast=1.0, n_target=3)[0]["s"]
           for i in range(args.n_cal)]
    from qcert.qint import quant_params_from_range

    lo = min(float(np.min(s)) for s in cal)
    hi = max(float(np.max(s)) for s in cal)
    delta, zp = quant_params_from_range(lo, hi, bits=args.bits, symmetric=True)

    aud = SaturationAuditor(bits=args.bits, frac=0.01)
    rows, r_by_c, miss_by_c = [], {}, {}
    for ci, con in enumerate(CONTRASTS):
        for i in range(args.n_scenes):
            sea = seas[i % len(seas)]
            stem = f"c{con:g}_{i:03d}"
            scene, f32, q = make_pair(
                seed=1000 + 100 * ci + i, contrast=con, delta=delta,
                bits=args.bits, stem=stem, n_target=3, sea_state=sea)
            r = scene_pfa_pair(f32, q, scene["boxes"], args.alpha,
                               args.guard_px, args.block_px, args.keep, args.seed)
            r["contrast"], r["sea"] = con, sea
            r["miss_test"] = _miss(q.prob, r["lambda_test"], scene["target_idx"])
            rows.append(r)
            aud.add_from_field(q, name="score", sea=sea, contrast=con)
            r_by_c.setdefault(con, []).append(r["R"])
            miss_by_c.setdefault(con, []).append(r["miss_test"])

    sat_by_c, tie_by_c = {}, {}
    for row in aud.table():
        c = float(row["contrast"])
        sat_by_c.setdefault(c, []).append(row["sat"])
        tie_by_c.setdefault(c, []).append(row["max_tie"])
    sat_by_c = {c: float(np.nanmean(v)) for c, v in sat_by_c.items()}
    tie_by_c = {c: int(max(v)) for c, v in tie_by_c.items()}

    table7 = [
        e2_row(c, args.n_scenes, sat_by_c[c], float("nan"), tie_by_c[c],
               float(np.nanmean(r_by_c[c])), float(np.nanmean(miss_by_c[c])))
        for c in CONTRASTS
    ]
    json_dump({"delta": delta, "zero_point": zp, "bits": args.bits,
               "note": "scale 由正常对比度标定一次，压测集复用"},
              Path(out) / "scale.json")
    return rows, aud.table(), table7, sat_by_c, r_by_c, (out, {})


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fp32", default=None)
    ap.add_argument("--int8", default=None)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--layers", default=None, help="（已废弃，用 --neck-sat-json）")
    ap.add_argument("--neck-sat-json", default=None,
                    help="neck_sat.py 的输出 json（Neck 三支路 sat）")
    ap.add_argument("--dose-from", default="meta", choices=["meta", "stem"],
                    help="剂量档来源：meta.contrast 或 stem 前缀 c{N}_（压测集 v2）")
    ap.add_argument("--synth", action="store_true", help="无板子离线跑通")
    ap.add_argument("--n-scenes", type=int, default=12, help="每档对比度的场景数")
    ap.add_argument("--n-cal", type=int, default=8, help="定 scale 用的正常场景数")
    ap.add_argument("--sea-states", nargs="*", default=None)
    ap.add_argument("--alpha", type=float, default=1e-3)
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--guard-px", type=float, default=8.0)
    ap.add_argument("--block-px", type=float, default=64.0)
    ap.add_argument("--keep", type=float, default=0.5,
                    help="块抽样比例（alpha=1e-3 需要足够的 Omega_meas 分辨率）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(RESULTS / "e2"))
    args = ap.parse_args(argv)

    out = ensure_dir(args.out)
    if args.synth or not (args.fp32 and args.int8):
        if not args.synth:
            print("[warn] 没给 --fp32/--int8，自动转 --synth 离线模式")
        rows, detail, table7, sat_by_c, r_by_c, _ = run_synth(args, out)
        neck_layers = {}
    else:
        rows, detail, table7, sat_by_c, r_by_c, (_, neck_layers) = \
            run_dumps(args, out)

    corr = correlation_with_r(sat_by_c, {c: float(np.nanmean(v))
                                        for c, v in r_by_c.items()})
    save_table(rows, out, "e2_per_scene", "E2 逐场景 R / 饱和 / 漏检")
    save_table(detail, out, "e2_audit", "E2 逐场饱和与并列审计")
    save_table(table7, out, "table7_e2", "Table 7 · E2 饱和审计")
    summary = {"alpha": args.alpha, "n_scenes": len(rows),
               "keep": args.keep,
               "source": "synth" if not (args.fp32 and args.int8) else "dump",
               "dose_from": args.dose_from,
               "sat_by_contrast": sat_by_c,
               "neck_sat_by_layer_contrast": neck_layers,
               "R_by_contrast": {c: float(np.nanmean(v)) for c, v in r_by_c.items()},
               "spearman": corr, "verdict": "GO" if corr["pass"] else "WEAK"}
    json_dump(summary, out / "summary.json")

    print("\n=== E2 · 饱和审计 ===")
    print(f"场景 {len(rows)}  alpha={args.alpha}")
    for r in table7:
        print(f"  ×{r['contrast']:.0f}: sat={r['score_sat']:.4f} "
              f"neck_sat={r['neck_sat']:.5f} max_tie={r['top1pct_max_tie']} "
              f"R={r['R']:.3f} miss={r['miss_rate']:.3f}")
    if neck_layers:
        for tag, per in neck_layers.items():
            s = "  ".join(f"×{c:g}:{v:.5f}" for c, v in sorted(per.items()))
            print(f"  neck[{tag}] {s}")
    print(f"Spearman(饱和, R) = {corr['rho']:.3f}  "
          f"单调={corr['monotonic_sat']}  → {summary['verdict']}")
    print(f"产物: {out}")
    return summary


if __name__ == "__main__":
    main()
