"""E5 · 阈值搬运：换量化配置时，整数阈值能不能"搬"过去。

协议依据 paper/04 §8。命题（02 §3.3）::

    lambda_int_new = round( lambda_int_old * (dSigma_new/dSigma_old) * (dS_old/dS_new) )

三列对照：

``pfa_transport``
    用上式搬运后的整数阈值直接部署（**不碰标定集**）。
``pfa_recal``
    在新配置上重新标定的上界（理想情况）。
``pfa_no_convert``
    偷懒直接复用旧 λ_int（错误的配置），应当是三者里最差。

四个 setting：同传感器换构建（A→B）、FP16→INT8、跨传感器、重标定上界。

    python -m experiments.run_e5 --synth
    python -m experiments.run_e5 --fp32 dumps/fp32 --old old_spec.json --new new_spec.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from qcert.decision import IntQuantSpec
from qcert.pfa_eval import block_ids, calibrate_lambda
from qcert.transfer import (TransportSpec, compare_pfa, lambda_int_from_float,
                            lambda_float_from_int, transport_lambda_int)

from .common import (RESULTS, ensure_dir, json_dump, load_pairs, save_table,
                     scene_pfa_pair)
from .synth import make_pair

SETTINGS = {
    "A→B（同传感器换构建）": (0.02, 0.05),
    "FP16→INT8": (1e-4, 0.05),
    "跨传感器": (0.02, 0.10),
    "重标定上界": (0.05, 0.05),
}


def _spec(delta_s, delta_sigma, name: str = "") -> TransportSpec:
    return TransportSpec(delta_s=float(delta_s), delta_sigma=float(delta_sigma),
                         shift_bits=16, name=name)


def _mu_sigma(field, block_px=64.0):
    """由分数场估出逐位置的 (mu, sigma) 代理：块内 median / std。

    真值链路里 μ/σ 来自杂波估计器；这里只是给"分母量化"一个合理载体，
    板端有真实估计器时替换掉即可。
    """
    s = field.z()
    centers = field.centers
    if centers is None:
        # 板端 dump 不落 centers：由 head_sizes 现算（与 omega_for 同一来源）
        from qcert.pfa_eval import cell_centers

        centers = cell_centers(field.head_sizes,
                               int(field.meta.get("imgsz", 640)))
    ids = block_ids(centers, block_px)
    mu = np.empty_like(s)
    sig = np.empty_like(s)
    for bid in np.unique(ids):
        m = ids == bid
        v = s[m]
        mu[m] = np.median(v)
        sig[m] = max(float(np.std(v)), 1e-3 * max(float(np.median(np.abs(v))), 1e-6))
    return s, mu, sig


def _counts(z, mask, lam) -> dict:
    """命中数与样本量（P_fa 必须按 hits/n 池化，不能对逐场景比值求均值）。"""
    v = np.asarray(z, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    hits = int((v > float(lam)).sum())
    return {"hits": hits, "n": int(v.size), "pfa": float(hits / v.size) if v.size else float("nan")}


def _pfa_pair(scene_or_field, alpha, eps_old, eps_new, guard_px, block_px,
              keep, seed, boxes):
    """一个场景下三种阈值（搬运 / 重标定 / 不换算）的命中计数。"""
    from .common import omega_for

    field = scene_or_field
    (mc, mm, info), centers = omega_for(field, boxes, guard_px, block_px, keep, seed)
    s, mu, sig = _mu_sigma(field, block_px)
    z_fp = (s - mu) / sig
    lam_fp = calibrate_lambda(z_fp, mc, alpha)

    ref = float(np.median(sig))
    sig_new = np.round(sig / (2 * eps_new * ref)) * (2 * eps_new * ref)
    z_new = (s - mu) / np.maximum(sig_new, 1e-6)

    old = _spec(ref * 0.02, 2 * eps_old * ref)
    new = _spec(ref * 0.02, 2 * eps_new * ref)
    lam_int_old = lambda_int_from_float(lam_fp, old)
    lam_int_new = transport_lambda_int(lam_int_old, old, new)

    lam_z_transport = lambda_float_from_int(lam_int_new, new)
    lam_z_noconv = lambda_float_from_int(lam_int_old, new)
    lam_z_recal = calibrate_lambda(z_new, mc, alpha)

    row = {"stem": getattr(field, "stem", "?"),
           "lambda_z_transport": float(lam_z_transport),
           "lambda_z_recal": float(lam_z_recal),
           "lambda_z_no_convert": float(lam_z_noconv)}
    for tag, lam in (("transport", lam_z_transport), ("recal", lam_z_recal),
                     ("no_convert", lam_z_noconv)):
        c = _counts(z_new, mm, lam)
        row[f"pfa_{tag}"], row[f"hits_{tag}"], row["n_meas"] = c["pfa"], c["hits"], c["n"]
    return row


def run_synth(args):
    rows = []
    for i in range(args.n_scenes):
        scene, fp32, _ = make_pair(seed=7000 + i, n_target=3, stem=f"t{i:03d}")
        for name, (eps_old, eps_new) in SETTINGS.items():
            r = _pfa_pair(fp32, args.alpha, eps_old, eps_new,
                          args.guard_px, args.block_px, args.keep,
                          args.seed, scene["boxes"])
            rows.append({"setting": name, "n_images": 1, **r})
    return rows


class _FieldView:
    """让 z() 返回指定键（如 logit）的分数场视图（其余属性透传）。"""

    def __init__(self, f, key):
        self._f, self._key = f, key

    def __getattr__(self, k):
        return getattr(self._f, k)

    def z(self):
        if self._key == "logit" and self._f.logit is not None:
            return self._f.logit.astype(np.float64)
        return self._f.z()


def run_dumps(args):
    pairs = load_pairs(args.fp32, args.fp32, verbose=False)
    old = _load_spec(args.old)
    new = _load_spec(args.new)
    rows = []
    for stem, ref, _ in pairs:
        ref = _FieldView(ref, args.score_key)
        from .common import gt_for

        boxes = gt_for(stem, args.labels, ref.meta) if args.labels else None
        from .common import omega_for

        (mc, mm, info), centers = omega_for(ref, boxes, args.guard_px,
                                            args.block_px, args.keep, args.seed)
        s, mu, sig = _mu_sigma(ref, args.block_px)
        z_fp = (s - mu) / sig
        lam_fp = calibrate_lambda(z_fp, mc, args.alpha)
        z_new = (s - mu) / np.maximum(
            np.round(sig / new.delta_sigma) * new.delta_sigma, 1e-6)
        lam_int_old = lambda_int_from_float(lam_fp, old)
        lam_int_new = transport_lambda_int(lam_int_old, old, new)

        row = {"stem": stem, "setting": args.setting or stem, "n_images": 1}
        for tag, lam in (("transport", lambda_float_from_int(lam_int_new, new)),
                         ("recal", calibrate_lambda(z_new, mc, args.alpha)),
                         ("no_convert", lambda_float_from_int(lam_int_old, new))):
            c = _counts(z_new, mm, lam)
            row[f"pfa_{tag}"], row[f"hits_{tag}"], row["n_meas"] = c["pfa"], c["hits"], c["n"]
        rows.append(row)
    return rows


def _load_spec(path):
    if not path:
        raise SystemExit("需要 --old/--new 量化配置 json")
    d = json.loads(open(path, encoding="utf-8").read())

    def pick(*keys, default=None):
        for k in keys:
            if k in d:
                return d[k]
        if default is None:
            raise SystemExit(f"{path} 缺少 {keys}")
        return default

    return _spec(pick("delta_s", "dS", "ds", default=1.0 / 128),
                 pick("delta_sigma", "dSigma", "dsigma", default=1.0 / 256))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fp32", default=None)
    ap.add_argument("--old", default=None)
    ap.add_argument("--new", default=None)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--synth", action="store_true")
    ap.add_argument("--n-scenes", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=1e-3)
    ap.add_argument("--guard-px", type=float, default=8.0)
    ap.add_argument("--block-px", type=float, default=64.0)
    ap.add_argument("--keep", type=float, default=0.5,
                    help="块抽样比例（0.5 = 全部块对半分给 Omega_cal / Omega_meas）。"
                         "alpha=1e-3 要求 n_meas >= 1e4，keep 太小会把 P_fa 卡在 1/n 分辨率上")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--score-key", default="prob", choices=["prob", "logit"],
                    help="分数场键：prob 是 sigmoid 输出（偏斜），logit 是 cv3 原值"
                         "（与量化步长 ΔS 同域，物理正确）")
    ap.add_argument("--setting", default=None,
                    help="固定 setting 分组名（缺省逐场景分组，n_meas*alpha 易不足）")
    ap.add_argument("--out", default=str(RESULTS / "e5"))
    args = ap.parse_args(argv)

    out = ensure_dir(args.out)
    if args.synth or not args.fp32:
        if not args.synth:
            print("[warn] 缺 --fp32，转 --synth")
        rows = run_synth(args)
    else:
        rows = run_dumps(args)

    agg = []
    for name in list(dict.fromkeys(r["setting"] for r in rows)):
        sub = [r for r in rows if r["setting"] == name]
        n = int(sum(r["n_meas"] for r in sub))
        # 池化 P_fa：hits/n。逐场景比值求均值在 P_fa << 1/n 时会被 0 命中支配。
        pfa = {t: (sum(r[f"hits_{t}"] for r in sub) / n if n else float("nan"))
               for t in ("transport", "recal", "no_convert")}
        c = compare_pfa(pfa["transport"], pfa["recal"], pfa["no_convert"], args.alpha)
        # 分辨率：n*alpha < 10 时连"1e-3 是否守住"都数不出来，判为数据不足。
        insufficient = bool(n * float(args.alpha) < 10.0)
        agg.append({
            "setting": name,
            "n_images": len(sub),
            "n_meas": n,
            "pfa_transport": pfa["transport"],
            "pfa_recal": pfa["recal"],
            "pfa_no_convert": pfa["no_convert"],
            "alpha": args.alpha,
            "drift_transport": c["drift_transport"],
            "drift_recal": c["drift_recal"],
            "drift_no_convert": c["drift_no_convert"],
            "insufficient": insufficient,
            "pass": bool(c["pass"] and not insufficient),
            "better_than_no_convert": bool(c["better_than_no_convert"]),
        })
    source = "synth" if (args.synth or not args.fp32) else "dump"
    summary = {"alpha": args.alpha, "n_rows": len(rows), "rows": agg,
               "keep": args.keep, "guard_px": args.guard_px,
               "block_px": args.block_px, "source": source,
               "all_pass": bool(all(r["pass"] for r in agg))}
    save_table(rows, out, "e5_per_scene", "E5 逐场景搬运")
    save_table(agg, out, "table4_e5", "Table 4 · E5 阈值搬运")
    json_dump(summary, out / "summary.json")

    print("\n=== E5 · 阈值搬运 ===")
    print(f"{'setting':<22s}{'n_meas':>9}{'P_fa(搬)':>12}{'P_fa(重标)':>12}"
          f"{'P_fa(不换)':>12}{'d(搬)':>7}{'d(重标)':>8}{'pass':>7}")
    for r in agg:
        print(f"{r['setting']:<22s}{r['n_meas']:>9d}{r['pfa_transport']:>12.3e}"
              f"{r['pfa_recal']:>12.3e}{r['pfa_no_convert']:>12.3e}"
              f"{r['drift_transport']:>7.3f}{r['drift_recal']:>8.3f}"
              f"{str(r['pass']):>7}")
    if any(r["insufficient"] for r in agg):
        print("[warn] 有 setting 的 n_meas*alpha < 10：P_fa 分辨率不足，"
              "判 pass=False（加大 --keep / --n-scenes 或换更小的 alpha）")
    print(f"产物: {out}")
    return summary


if __name__ == "__main__":
    main()
