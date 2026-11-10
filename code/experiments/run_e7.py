"""E7 · 板级系统矩阵：{FP16, INT8} × {全局阈值, QCert}。

协议依据 paper/04 §10。要证明的不是"更快"，而是**同虚警率下更省**：
粗糙海况里全局阈值会产生大量候选，QCert 的逐场景 λ 把候选压下来，
于是 J/艘（每检出一艘船消耗的能量）和 P99/mean 一起下降。

四配置 × 九列（Table 5 上）：候选数 median/P90、判决层耗时、T_e2e 均值、
P99、P99/mean、功率、J/艘、CPU。

    # 真板（先跑 acl_infer.py --dump-scores 出两套 dump）
    python -m experiments.run_e7 --fp16 dumps/fp16 --int8 dumps/int8 \
        --labels asc-sar/datasets/ssdd/labels/test \
        --latency-json '{"fp16":8.6,"int8":4.2}' --power-w 12

    # 离线自检（合成场，时延/功率为假设值，只验证代码路径与判定）
    python -m experiments.run_e7 --synth --n 40 --power-w 12

注意：离线模式的时延与功率都是**输入假设**，产物里 ``source`` 字段会标明，
不能当作板级实测数据写进论文。
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from qcert.decision import decide_simple
from qcert.qint import quant_params_from_range, quantize
from qcert.scores import load_dir

from .common import RESULTS, ensure_dir, json_dump, json_load, save_table

try:
    from scipy import ndimage as _ndi
except Exception:                                     # pragma: no cover
    _ndi = None

THRESHOLDS = ("global", "qcert")
IMPLS = ("fp16", "int8")
COARSE = 80                                           # head-0 网格边长


def _quant_pack(z, bits=8):
    """把分数场打成整数判决层需要的三路整数（s_q / mu_q / sigma_q 代理）。"""
    z = np.asarray(z, dtype=np.float64)
    lo, hi = np.percentile(z, 0.5), np.percentile(z, 99.5)
    delta, zp = quant_params_from_range(lo, hi, bits=bits, symmetric=True)
    s_q = quantize(z, delta, zp, bits=bits)
    med = float(np.median(z))
    mad = float(np.median(np.abs(z - med))) + 1e-9
    sigma_q = np.maximum(np.round((np.abs(z - med) + mad) / delta), 1).astype(np.int32)
    mu_q = np.full_like(s_q, int(round((med - zp * delta) / delta)))
    return s_q, mu_q, sigma_q, delta


def _lambda_int(lam, delta, k=16):
    return int(round(float(lam) / delta * (1 << k)))


def _decide_ms(s_q, mu_q, sigma_q, lam_q, reps=20):
    """判决层耗时（ms）：中位数与 P99，逐次单独计时。"""
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        decide_simple(s_q, mu_q, sigma_q, 16, lam_q)
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts = np.asarray(ts)
    return float(np.median(ts)), float(np.percentile(ts, 99))


def _n_regions(center_idx, n_cells):
    """head-0 的 80×80 网格上，候选单元的连通块数（候选"目标"数）。"""
    m = np.zeros(COARSE * COARSE, dtype=bool)
    m[np.asarray(center_idx, dtype=int)] = True
    if _ndi is None or not m.any():                   # pragma: no cover
        return int(m.sum())
    lab, n = _ndi.label(m.reshape(COARSE, COARSE), structure=np.ones((3, 3), int))
    return int(n)


def _frames_from_dumps(root):
    fields = load_dir(root)
    frames = []
    for f in fields:
        z = np.asarray(f.z(), dtype=np.float64)
        if z.size == 0:
            continue
        frames.append({"stem": f.stem, "z": z,
                       "sea": (f.meta or {}).get("sea_state", "unknown"),
                       "n_head0": min(z.size, COARSE * COARSE)})
    return frames


def _frames_from_synth(n, seed0=7000, sea_cycle=("calm", "moderate", "rough"), n_target=3):
    from .synth import make_scene, to_field
    out = []
    for i in range(n):
        sea = sea_cycle[i % len(sea_cycle)]
        sc = make_scene(seed=seed0 + i, n_target=n_target, sea_state=sea,
                        contrast=(1.0 if sea != "rough" else 1.4))
        f = to_field(sc, impl="synth")
        out.append({"stem": f"synth_{i:04d}", "z": np.asarray(f.z(), dtype=np.float64),
                    "sea": sea, "n_head0": COARSE * COARSE})
    return out


def _pooled_lambda(frames, alpha):
    """全局阈值：把所有帧的分数混在一起取 1−α 分位（"一套参数打天下"）。"""
    pool = np.concatenate([f["z"] for f in frames])
    return float(np.quantile(pool, 1.0 - alpha))


def _eval_config(frames, thr, alpha, lam_global, t_model_ms, power_w, cpu_pct):
    n_cand, n_reg, dec, e2e, cand_rough = [], [], [], [], []
    for f in frames:
        z = f["z"]
        lam = lam_global if thr == "global" else float(np.quantile(z, 1.0 - alpha))
        det = z > lam
        n_cand.append(int(det.sum()))
        if f.get("sea") == "rough":
            cand_rough.append(int(det.sum()))
        idx = np.nonzero(det)[0]
        n_reg.append(_n_regions(idx[idx < COARSE * COARSE], z.size))

        s_q, mu_q, sigma_q, delta = _quant_pack(z)
        lam_q = _lambda_int(lam, delta)
        d_med, d_p99 = _decide_ms(s_q, mu_q, sigma_q, lam_q)
        dec.append(d_med)
        e2e.append(t_model_ms + d_med)

    pfa = float(np.mean([np.mean(f["z"] > (lam_global if thr == "global"
                                           else np.quantile(f["z"], 1.0 - alpha)))
                         for f in frames]))
    e2e = np.asarray(e2e)
    n_ship = float(np.mean(n_reg)) if n_reg else float("nan")
    j_per_ship = (power_w * float(e2e.mean()) / 1000.0 / n_ship
                  if n_ship and np.isfinite(n_ship) and n_ship > 0 else float("nan"))
    return {
        "impl": None, "thr": thr,
        "cand_median": float(np.median(n_cand)),
        "cand_p90": float(np.percentile(n_cand, 90)),
        # 04 §10 的验收只在粗糙海况上看：那里全局阈值最吃亏
        "cand_p90_rough": (float(np.percentile(cand_rough, 90))
                           if cand_rough else float("nan")),
        "n_rough": len(cand_rough),
        "regions_median": float(np.median(n_reg)),
        "dec_ms": float(np.median(dec)),
        "dec_ms_p99": float(np.percentile(dec, 99)),
        "t_e2e_mean_ms": float(e2e.mean()),
        "t_e2e_p99_ms": float(np.percentile(e2e, 99)),
        "p99_over_mean": (float(np.percentile(e2e, 99) / e2e.mean())
                          if e2e.mean() > 0 else float("nan")),
        "pfa_meas": pfa,
        "power_w": power_w, "j_per_ship": j_per_ship,
        "cpu_pct": cpu_pct,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fp16", default=None, help="FP16 链路的分数场 dump 目录")
    ap.add_argument("--int8", default=None, help="INT8 链路的分数场 dump 目录")
    ap.add_argument("--labels", default=None, help="YOLO 标签目录（可选，仅记录）")
    ap.add_argument("--synth", action="store_true", help="用合成场离线自检")
    ap.add_argument("--n", type=int, default=40, help="--synth 的帧数")
    ap.add_argument("--alpha", type=float, default=1e-3)
    ap.add_argument("--latency-json", default=None,
                    help='板级实测时延 {"fp16":.., "int8":..} ms')
    ap.add_argument("--latency-ms", type=float, default=8.0,
                    help="未给 latency-json 时的统一时延假设（离线自检用）")
    ap.add_argument("--power-w", type=float, default=12.0, help="整板功率（W）")
    ap.add_argument("--cpu-pct", type=float, default=float("nan"))
    ap.add_argument("--out", default=str(RESULTS / "e7"))
    args = ap.parse_args(argv)

    dirs = {"fp16": args.fp16, "int8": args.int8}
    lat = json_load(args.latency_json) if args.latency_json else {}
    rows, meta = [], {"alpha": args.alpha, "power_w": args.power_w}

    for impl in IMPLS:
        if args.synth or not dirs[impl]:
            frames = _frames_from_synth(args.n if impl == "fp16" else args.n,
                                        seed0=7000 + (0 if impl == "fp16" else 97))
            source = "synth"
        else:
            frames = _frames_from_dumps(Path(dirs[impl]))
            source = "dump"
        if not frames:
            continue
        t_model = float(lat.get(impl, args.latency_ms))
        lam_g = _pooled_lambda(frames, args.alpha)
        for thr in THRESHOLDS:
            r = _eval_config(frames, thr, args.alpha, lam_g, t_model,
                             args.power_w, args.cpu_pct)
            r["impl"] = impl
            r["source"] = source
            r["t_model_ms"] = t_model
            rows.append(r)

    # 04 §10 验收：粗糙海况候选 P90 至少减半；判决层 < 3 ms
    verdicts = {}
    for impl in IMPLS:
        g = next((r for r in rows if r["impl"] == impl and r["thr"] == "global"), None)
        q = next((r for r in rows if r["impl"] == impl and r["thr"] == "qcert"), None)
        if not (g and q):
            continue
        ratio = (g["cand_p90_rough"] / q["cand_p90_rough"]
                 if q["cand_p90_rough"] and np.isfinite(q["cand_p90_rough"])
                 and q["cand_p90_rough"] > 0 else float("inf"))
        verdicts[impl] = {
            "cand_p90_rough_global": g["cand_p90_rough"],
            "cand_p90_rough_qcert": q["cand_p90_rough"],
            "cand_p90_ratio_global_over_qcert": float(ratio),
            "pass_cand": bool(ratio >= 2.0),
            "pass_dec_ms": bool(q["dec_ms"] < 3.0),
            "j_per_ship_drop": (float(g["j_per_ship"] / q["j_per_ship"])
                                if q["j_per_ship"] and np.isfinite(q["j_per_ship"]) else float("nan")),
        }

    out = ensure_dir(args.out)
    save_table(rows, out, "table5a_e7", "Table 5 上 · E7 板级系统矩阵")
    sources = sorted({r["source"] for r in rows})
    json_dump({**meta, "n_frames": args.n, "verdicts": verdicts,
               "source": sources[0] if len(sources) == 1 else sources,
               "note": "offline 时延/功率为输入假设" if args.synth else "板级实测"},
              out / "summary.json")

    print("\n=== E7 · 系统矩阵 ===")
    hdr = (f"{'impl':>6}{'thr':>8}{'cand_med':>10}{'cand_p90':>10}"
           f"{'rgh_p90':>9}{'dec_ms':>9}{'e2e_mean':>10}{'e2e_p99':>9}"
           f"{'p99/mean':>10}{'J/艘':>9}")
    print(hdr)
    for r in rows:
        print(f"{r['impl']:>6}{r['thr']:>8}{r['cand_median']:>10.0f}"
              f"{r['cand_p90']:>10.0f}{r['cand_p90_rough']:>9.0f}{r['dec_ms']:>9.3f}"
              f"{r['t_e2e_mean_ms']:>10.2f}{r['t_e2e_p99_ms']:>9.2f}"
              f"{r['p99_over_mean']:>10.3f}{r['j_per_ship']:>9.3f}")
    for impl, v in verdicts.items():
        print(f"  {impl}: 候选 P90 全局/QCert = {v['cand_p90_ratio_global_over_qcert']:.2f}× "
              f"(≥2 → {v['pass_cand']})；判决层 {v['pass_dec_ms']}；"
              f"J/艘 降 {v['j_per_ship_drop']:.2f}×")
    print(f"产物: {out}")
    return {"rows": rows, "verdicts": verdicts}


if __name__ == "__main__":
    main()
