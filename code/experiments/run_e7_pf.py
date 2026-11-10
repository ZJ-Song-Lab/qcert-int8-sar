"""E7 补充实验：逐帧 P_fa 分布（F3 证据）+ 透明海况分桶重算 E7。

背景（审计遗留）：E7 原表（results/e7/table5a_e7.csv）的海况标签系一次性
外部注入，分桶口径已不可考（n_rough=77、rough 候选 P90=28.0/27.6 均无法从
现有产物复现）。本脚本：

1. 用 qcert.seastate.SeaStateBucketer（kmeans k=3；特征 cv/weibull_c/contrast；
   按粗糙度定序 calm/medium/rough）对 232 张 test 图重新分桶，标签与分桶器
   一并落盘——任何人用同一工具即可复现；
2. 在固定分桶下重算 E7 系统矩阵（t_model、功率、判决层口径与原表一致），
   产物 table5a_e7_v2.*，不覆盖原表；
3. 逐帧 P_fa 分布：对每帧计算
     P_fa^global(f) = mean(z_f > λ_global)      （λ 全帧池化，跨帧固定）
     P_fa^qcert(f)  = mean(z_f > Q_{1-α}(z_f))  （逐帧各自分位）
   给出两条机制下逐帧违约（> c·α）比例与尾部统计——全局机制"逐帧虚警率
   不受控"的定量证据（F3）。

    python -m experiments.run_e7_pf
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image

from qcert.scores import load_dir
from qcert.seastate import SeaStateBucketer, clutter_features

from .common import RESULTS, ensure_dir, json_dump
from .run_e7 import _eval_config, _pooled_lambda

ALPHA = 1e-3
FP16_DIR = Path(__file__).resolve().parents[2] / "asc-sar" / "results" / "scores_ssdd_fp16"
INT8_DIR = Path(__file__).resolve().parents[2] / "asc-sar" / "results" / "scores_ssdd_int8"
IMG_DIR = Path(__file__).resolve().parents[2] / "asc-sar" / "datasets" / "ssdd" / "images" / "test"

# 与原表一致的板级输入（results/e7/table5a_e7.csv / summary.json）
T_MODEL = {"fp16": 45.5, "int8": 34.68}
POWER_W = 9.31


def bucket_frames(stems):
    """对 test 集 232 图重新分桶；返回 {stem: label} 与分桶器。"""
    feats = {}
    for s in stems:
        img = np.asarray(Image.open(IMG_DIR / f"{s}.jpg").convert("L"),
                         dtype=np.float64) / 255.0
        fd = clutter_features(img)
        feats[s] = fd
    bucketer = SeaStateBucketer().fit(list(feats.values()))
    labels = dict(zip(stems, bucketer.predict(list(feats.values()))))
    return labels, bucketer, feats


def main():
    d16 = load_dir(FP16_DIR)
    d8 = load_dir(INT8_DIR)
    stems = sorted(d16.keys())
    assert stems == sorted(d8.keys()), "fp16/int8 dump 帧名不一致"

    out = ensure_dir(RESULTS / "e7")

    # ---------- 1) 透明分桶 ----------
    labels, bucketer, feats = bucket_frames(stems)
    from collections import Counter
    counts = Counter(labels.values())
    bucketer.save(out / "e7_bucketer.json")
    with open(out / "e7_labels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["stem", "sea_state", "cv", "weibull_c", "contrast"])
        for s in stems:
            w.writerow([s, labels[s], feats[s]["cv"], feats[s]["weibull_c"],
                        feats[s]["contrast"]])
    print(f"分桶: {dict(counts)}  (bucketer -> {out / 'e7_bucketer.json'})")

    # ---------- 2) 重算 E7 系统矩阵 ----------
    rows = []
    for impl, dd in (("fp16", d16), ("int8", d8)):
        frames = [{"stem": s, "z": np.asarray(dd[s].z(), dtype=np.float64),
                   "sea": labels[s]} for s in stems]
        lam_g = _pooled_lambda(frames, ALPHA)
        for thr in ("global", "qcert"):
            r = _eval_config(frames, thr, ALPHA, lam_g,
                             T_MODEL[impl], POWER_W, float("nan"))
            r["impl"] = impl
            r["source"] = "dump+transparent-bucket"
            r["t_model_ms"] = T_MODEL[impl]
            rows.append(r)
    with open(out / "table5a_e7_v2.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    verdicts = {}
    for impl in ("fp16", "int8"):
        g = next(r for r in rows if r["impl"] == impl and r["thr"] == "global")
        q = next(r for r in rows if r["impl"] == impl and r["thr"] == "qcert")
        ratio = g["cand_p90_rough"] / q["cand_p90_rough"]
        verdicts[impl] = {
            "cand_p90_rough_global": g["cand_p90_rough"],
            "cand_p90_rough_qcert": q["cand_p90_rough"],
            "ratio_global_over_qcert": float(ratio),
            "pass_cand": bool(ratio >= 2.0),
            "n_rough": g["n_rough"],
        }

    # ---------- 3) 逐帧 P_fa 分布 ----------
    lam_g = {}
    for impl, dd in (("fp16", d16), ("int8", d8)):
        pool = np.concatenate([np.asarray(dd[s].z(), dtype=np.float64) for s in stems])
        lam_g[impl] = float(np.quantile(pool, 1.0 - ALPHA))

    pf_rows = []
    for s in stems:
        row = {"stem": s, "sea_state": labels[s]}
        for impl, dd in (("fp16", d16), ("int8", d8)):
            z = np.asarray(dd[s].z(), dtype=np.float64)
            lam_q = float(np.quantile(z, 1.0 - ALPHA))
            row[f"pfa_global_{impl}"] = float(np.mean(z > lam_g[impl]))
            row[f"pfa_qcert_{impl}"] = float(np.mean(z > lam_q))
            row[f"n_cand_global_{impl}"] = int((z > lam_g[impl]).sum())
            row[f"n_cand_qcert_{impl}"] = int((z > lam_q).sum())
        pf_rows.append(row)
    with open(out / "e7_per_frame_pfa.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(pf_rows[0].keys()))
        w.writeheader()
        w.writerows(pf_rows)

    def _stats(key):
        """逐帧 P_fa 的违约统计（分 impl、分 all/rough）。"""
        res = {}
        for impl in ("fp16", "int8"):
            for scope in ("all", "rough"):
                vals = [r[f"pfa_{key}_{impl}"] for r in pf_rows
                        if scope == "all" or r["sea_state"] == "rough"]
                v = np.asarray(vals)
                res[f"{key}_{impl}_{scope}"] = {
                    "n": int(v.size),
                    "mean": float(v.mean()), "median": float(np.median(v)),
                    "p90": float(np.percentile(v, 90)), "max": float(v.max()),
                    "min": float(v.min()),
                    "frac_gt_2alpha": float(np.mean(v > 2 * ALPHA)),
                    "frac_gt_5alpha": float(np.mean(v > 5 * ALPHA)),
                    "frac_gt_10alpha": float(np.mean(v > 10 * ALPHA)),
                    "frac_lt_half_alpha": float(np.mean(v < 0.5 * ALPHA)),
                }
        return res

    pf_stats = {**_stats("global"), **_stats("qcert")}
    json_dump({
        "alpha": ALPHA,
        "n_frames": len(stems),
        "bucket_counts": dict(counts),
        "bucket_method": "SeaStateBucketer(kmeans k=3, features=cv/weibull_c/contrast)",
        "t_model_ms": T_MODEL, "power_w": POWER_W,
        "verdicts": verdicts,
        "per_frame_pfa_stats": pf_stats,
        "note": "逐帧 P_fa：global=跨帧固定λ（池化分位），qcert=逐帧分位；"
                "违约定义为逐帧 P_fa > c·α",
    }, out / "e7_pf_summary.json")

    # ---------- 打印 ----------
    print("\n=== E7 v2 · 透明分桶系统矩阵 ===")
    print(f"{'impl':>6}{'thr':>8}{'cand_med':>10}{'cand_p90':>10}"
          f"{'rgh_n':>7}{'rgh_p90':>9}{'dec_ms':>9}{'pfa_meas':>12}")
    for r in rows:
        print(f"{r['impl']:>6}{r['thr']:>8}{r['cand_median']:>10.0f}"
              f"{r['cand_p90']:>10.0f}{r['n_rough']:>7}{r['cand_p90_rough']:>9.0f}"
              f"{r['dec_ms']:>9.3f}{r['pfa_meas']:>12.6f}")
    for impl, v in verdicts.items():
        print(f"  {impl}: rough 候选 P90 全局/QCert = {v['ratio_global_over_qcert']:.2f}× "
              f"(≥2 → {v['pass_cand']}, n_rough={v['n_rough']})")

    print("\n=== 逐帧 P_fa 分布 ===")
    print(f"{'机制':>16}{'impl':>6}{'范围':>6}{'mean':>9}{'median':>9}"
          f"{'p90':>9}{'max':>9}{'>2α':>7}{'>5α':>7}{'>10α':>7}{'<α/2':>7}")
    for key in ("global", "qcert"):
        for impl in ("fp16", "int8"):
            for scope in ("all", "rough"):
                s = pf_stats[f"{key}_{impl}_{scope}"]
                print(f"{key:>16}{impl:>6}{scope:>6}{s['mean']:>9.5f}"
                      f"{s['median']:>9.5f}{s['p90']:>9.5f}{s['max']:>9.5f}"
                      f"{s['frac_gt_2alpha']:>7.3f}{s['frac_gt_5alpha']:>7.3f}"
                      f"{s['frac_gt_10alpha']:>7.3f}{s['frac_lt_half_alpha']:>7.3f}")
    print(f"产物: {out}")


if __name__ == "__main__":
    main()
