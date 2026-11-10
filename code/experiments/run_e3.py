"""E3 · 三实现判决差异 + 同 .om 重复 1000 次的 bit-exact。

协议依据 paper/04 §6。目的不是比精度，而是回答两个只有定点板能回答的问题：

1. **同一份权重量化成 INT8，板端 ACL 与 onnxruntime 的判决会不会不一样？**
   会，而且差异集中在尾部（P_fa 那一端）。
2. **同一个 .om、同一张图、连跑 1000 次，输出是否逐 bit 相同？**
   这是 A1 假设（``qcert/deploy`` 的一切可复现性论证的前提），也是
   「P99 = mean」的来源。

板端::

    python -m experiments.run_e3 --fp32 dumps/fp32 --int8 dumps/board \
        --ort dumps/ort --repeat results/board/repeat.json

离线自检::

    python -m experiments.run_e3 --synth
"""
from __future__ import annotations

import argparse

import numpy as np

from qcert.decision import compare_decisions, decide_float
from qcert.pfa_eval import calibrate_lambda, measure_pfa

from .common import (RESULTS, ensure_dir, json_dump, json_load, load_pairs,
                     save_table, scene_pfa_pair)
from .synth import delta_for_range, make_pair, to_field

IMPLS = ("fp32", "int8_board", "int8_ort")


def _cal(alpha, keep, block_px, guard_px, seed, ref, boxes):
    from .common import omega_for

    (mc, mm, info), _ = omega_for(ref, boxes, guard_px, block_px, keep, seed)
    lam = calibrate_lambda(ref.z(), mc, alpha)
    return lam, mc, mm, info


def _row(stem, fields, lam_fp, mm, alpha):
    z = {k: v.z() for k, v in fields.items()}
    d = {k: decide_float(v, lam_fp) for k, v in z.items()}
    row = {"stem": stem, "alpha": alpha, "lambda_fp32": lam_fp}
    for k in z:
        row[f"pfa_fixed_{k}"] = measure_pfa(z[k], mm, lam_fp)
        row[f"pfa_own_{k}"] = measure_pfa(z[k], mm, calibrate_lambda(z[k], mm, alpha))
    base = d["fp32"]
    for k in z:
        if k == "fp32":
            continue
        c = compare_decisions(d[k], base)
        row[f"disagree_{k}"] = c["disagree_rate"]
        row[f"n_diff_{k}"] = c["n_diff"]
    row["R_board"] = (row["pfa_own_int8_board"] / row["pfa_own_fp32"]
                      if row["pfa_own_fp32"] > 0 else np.inf)
    if "int8_ort" in z:
        row["R_ort"] = (row["pfa_own_int8_ort"] / row["pfa_own_fp32"]
                        if row["pfa_own_fp32"] > 0 else np.inf)
    return row


def _repeat_summary(path):
    if not path:
        return {}
    d = json_load(path)
    if isinstance(d, dict) and "n" in d:
        return {"bit_exact_rate": d.get("rate", d.get("n_exact", 0) / max(1, d.get("n", 1))),
                "bit_exact_n": d.get("n")}
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, dict) and "n" in v:
                out[k] = v.get("rate", v.get("n_exact", 0) / max(1, v["n"]))
    return {"bit_exact_rate": float(np.nanmean(list(out.values()))) if out else float("nan"),
            "bit_exact_n": sum(v.get("n", 0) for v in d.values()
                               if isinstance(v, dict)) if isinstance(d, dict) else 0}


def run_dumps(args, out):
    pairs = load_pairs(args.fp32, args.int8)
    rows = []
    for stem, ref, board in pairs:
        from .common import gt_for

        boxes = gt_for(stem, args.labels, ref.meta) if args.labels else None
        lam, mc, mm, info = _cal(args.alpha, args.keep, args.block_px,
                                 args.guard_px, args.seed, ref, boxes)
        fields = {"fp32": ref, "int8_board": board}
        if args.ort:
            from qcert.scores import load_dir

            o = load_dir(args.ort)
            if stem in o:
                fields["int8_ort"] = o[stem]
        rows.append(_row(stem, fields, lam, mm, args.alpha))
    return rows


def run_synth(args, out):
    n = args.n_scenes
    scenes = [make_pair(seed=5000 + i, n_target=3, stem=f"s{i:03d}")
              for i in range(n)]
    delta, _ = delta_for_range([s[0]["s"] for s in scenes], bits=8)
    rows = []
    for i, (scene, fp32, _) in enumerate(scenes):
        # 板端：AMCT 的 scale；ORT：同一权重但 scale 略不同（真实会发生）
        board = to_field(scene, impl="int8", delta=delta, bits=8, stem=f"s{i:03d}")
        ort = to_field(scene, impl="int8", delta=delta * (1.0 + args.scale_gap),
                       bits=8, stem=f"s{i:03d}")
        lam, mc, mm, info = _cal(args.alpha, args.keep, args.block_px,
                                 args.guard_px, args.seed, fp32, scene["boxes"])
        rows.append(_row(f"s{i:03d}",
                         {"fp32": fp32, "int8_board": board, "int8_ort": ort},
                         lam, mm, args.alpha))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fp32", default=None)
    ap.add_argument("--int8", default=None, help="板端 ACL 的 dump 目录")
    ap.add_argument("--ort", default=None, help="onnxruntime INT8 的 dump 目录")
    ap.add_argument("--labels", default=None)
    ap.add_argument("--repeat", default=None,
                    help="acl_infer --repeat 写出的 json（A1 证据）")
    ap.add_argument("--synth", action="store_true")
    ap.add_argument("--n-scenes", type=int, default=8)
    ap.add_argument("--scale-gap", type=float, default=0.03,
                    help="合成模式下模拟 ORT 与板的 scale 差（相对量）")
    ap.add_argument("--alpha", type=float, default=1e-3)
    ap.add_argument("--guard-px", type=float, default=8.0)
    ap.add_argument("--block-px", type=float, default=64.0)
    ap.add_argument("--keep", type=float, default=0.5,
                    help="块抽样比例（alpha=1e-3 需要足够的 Omega_meas 分辨率）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(RESULTS / "e3"))
    args = ap.parse_args(argv)

    out = ensure_dir(args.out)
    if args.synth or not (args.fp32 and args.int8):
        if not args.synth:
            print("[warn] 缺 --fp32/--int8，转 --synth")
        rows = run_synth(args, out)
    else:
        rows = run_dumps(args, out)

    def _m(key):
        v = np.asarray([r.get(key, np.nan) for r in rows], dtype=np.float64)
        v = v[np.isfinite(v)]
        return float(v.mean()) if v.size else float("nan")

    rep = _repeat_summary(args.repeat)
    table2 = [{
        "impl": "INT8 · ACL(板)",
        "n_scenes": len(rows),
        "disagree_vs_fp32": _m("disagree_int8_board"),
        "pfa_own": _m("pfa_own_int8_board"),
        "R_vs_fp32": _m("R_board"),
        "bit_exact_rate": rep.get("bit_exact_rate", float("nan")),
    }]
    if any("disagree_int8_ort" in r for r in rows):
        table2.append({
            "impl": "INT8 · ORT",
            "n_scenes": len(rows),
            "disagree_vs_fp32": _m("disagree_int8_ort"),
            "pfa_own": _m("pfa_own_int8_ort"),
            "R_vs_fp32": _m("R_ort"),
            "bit_exact_rate": float("nan"),
        })
    table2.append({"impl": "FP32", "n_scenes": len(rows),
                   "disagree_vs_fp32": 0.0, "pfa_own": _m("pfa_own_fp32"),
                   "R_vs_fp32": 1.0, "bit_exact_rate": float("nan")})

    a1 = rep.get("bit_exact_rate", float("nan"))
    summary = {
        "alpha": args.alpha, "n_scenes": len(rows),
        "disagree_board": _m("disagree_int8_board"),
        "disagree_ort": _m("disagree_int8_ort"),
        "R_board": _m("R_board"), "R_ort": _m("R_ort"),
        "bit_exact_rate": a1,
        "A1_holds": bool(np.isfinite(a1) and a1 >= 1.0 - 1e-12),
        "note": ("A1 未验证：用 asc-sar/deploy/acl_infer.py --repeat 1000 生成 json "
                 "后加 --repeat 再跑" if not np.isfinite(a1) else ""),
    }
    save_table(rows, out, "e3_per_scene", "E3 逐场景三实现判决差异")
    save_table(table2, out, "table2_e3", "Table 2 · E3 三实现")
    json_dump(summary, out / "summary.json")

    print("\n=== E3 · 三实现 ===")
    for r in table2:
        print(f"  {r['impl']:<16s} disagree={r['disagree_vs_fp32']:.4f} "
              f"P_fa={r['pfa_own']:.3e} R={r['R_vs_fp32']:.3f} "
              f"bit-exact={r['bit_exact_rate']}")
    print(f"A1（逐 bit 可复现）: {summary['A1_holds']}")
    if summary["note"]:
        print("  " + summary["note"])
    print(f"产物: {out}")
    return summary


if __name__ == "__main__":
    main()
