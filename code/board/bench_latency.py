"""模块 8/9 的板级入口：延迟、功率与热稳态（E6 / E7 的输入）。

两种模式：

1. ``bench`` —— 纯推理延迟 + 整板功率，产出 E7 的 ``--latency-json``::

       python -m board.bench_latency bench --model yolov8n_int8.om \\
           --images data/ssdd/images/test --iters 200 --impl int8 \\
           --merge-into results/m0/latency.json

2. ``thermal`` —— 热稳态长跑，产出 E6 的窗口表（paper/04 §9：
   40 min、每 5 min 一窗、每窗 200 帧、``|log10(P_fa)| < 0.15``）::

       python -m board.bench_latency thermal --model yolov8n_int8.om \\
           --images data/ssdd/images/test --minutes 40 --window-min 5 \\
           --lam-from-dump dumps/int8 --scale-json results/m0/scale.json \\
           --out-dir results/e6

x86 自检（不碰 pyACL，用假时钟把 40 min 压缩到亚秒）::

       python -m board.bench_latency bench --dry-run --impl int8
       python -m board.bench_latency thermal --dry-run

``--dry-run`` 的时延/功率是**假设值**，产物里 ``source="dry-run"``，
不能当作板级实测数据写进论文。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

try:                                   # python -m board.bench_latency
    from .runner import (RESULTS, ensure_dir, frame_fn_from_model, json_dump,
                         json_load, list_images, load_model, nc_of,
                         preprocess_image)
except ImportError:                    # 直接 python board/bench_latency.py
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from runner import (RESULTS, ensure_dir, frame_fn_from_model, json_dump,  # noqa: F401
                        json_load, list_images, load_model, nc_of,
                        preprocess_image)


# ------------------------------------------------------------------ bench
def _bench_stats(ts, source, impl, model=None, pw=(), desc=None) -> dict:
    import numpy as np

    ts = np.asarray(ts, dtype=np.float64)
    mean = float(ts.mean())
    return {
        "impl": impl, "source": source, "model": model,
        "n": int(ts.size),
        "mean_ms": mean,
        "p50_ms": float(np.percentile(ts, 50)),
        "p95_ms": float(np.percentile(ts, 95)),
        "p99_ms": float(np.percentile(ts, 99)),
        "p99_over_mean": float(np.percentile(ts, 99) / mean) if mean > 0 else float("nan"),
        "fps": float(1000.0 / mean) if mean > 0 else float("nan"),
        "power_mean_w": float(np.mean(pw)) if len(pw) else float("nan"),
        "power_peak_w": float(max(pw)) if len(pw) else float("nan"),
        "power_n": int(len(pw)),
        "desc": desc,
    }


def _bench_real(args) -> dict:
    from qcert.thermal import SmiSampler

    imgs = list_images(args.images, args.limit)
    model = load_model(args.model, args.device_id)
    ts, pw, sampler = [], [], None
    try:
        desc = model.describe()
        xs = [preprocess_image(p, args.imgsz)[0]
              for p in imgs[:min(16, len(imgs))]]
        print(f"[bench] warmup={args.warmup} iters={args.iters} 图池={len(xs)}")
        for i in range(int(args.warmup)):
            model.infer_raw(xs[i % len(xs)])
        if args.power:
            sampler = SmiSampler(interval=0.2)
            sampler.start()
        wall0 = time.perf_counter()
        for i in range(int(args.iters)):
            t0 = time.perf_counter()
            model.infer_raw(xs[i % len(xs)])
            ts.append((time.perf_counter() - t0) * 1000.0)
        wall = time.perf_counter() - wall0
        if sampler is not None:
            sampler.stop()
            pw = [s["power_w"] for _, s in sampler.samples if "power_w" in s]
        print(f"[bench] 端到端 {args.iters / wall:.2f} FPS")
    finally:
        model.destroy()
    out = _bench_stats(ts, "board", args.impl, str(args.model), pw, desc)
    out["e2e_fps"] = float(ts.size / wall) if wall > 0 else float("nan")
    return out


def _bench_dry(args) -> dict:
    import numpy as np

    base = (float(args.dry_latency) if args.dry_latency > 0
            else (8.0 if args.impl == "fp16" else 4.0))
    rng = np.random.default_rng(0)
    ts = np.clip(rng.normal(base, base * 0.06, int(args.iters)), 0.1, None)
    out = _bench_stats(ts, "dry-run", args.impl, None, [args.dry_power])
    out["e2e_fps"] = float("nan")
    out["note"] = "dry-run：时延/功率为假设值，不可写入论文"
    return out


# ------------------------------------------------------------------ thermal
def _fake_clock(step_s: float = 0.02):
    """假时钟：每次调用推进 ``step_s``，让 40 min 的长跑压缩到亚秒。"""
    state = {"t": 0.0}

    def clock():
        state["t"] += step_s
        return state["t"]

    def sleep(sec):
        state["t"] += max(0.0, float(sec))

    return clock, sleep


def _lambda_from_dump(root, alpha: float):
    """全局 λ：把标定 dump 的分数汇总后取 1−α 分位（与 run_e7 的全局阈值同口径）。"""
    import numpy as np

    from qcert.scores import load_dir

    fields = load_dir(root)
    pool = np.concatenate([np.asarray(f.prob, dtype=np.float64)
                           for f in fields.values()])
    return float(np.quantile(pool, 1.0 - float(alpha))), int(pool.size)


def _thermal_real(args) -> list:
    from qcert.thermal import SmiSampler, ThermalRunner

    imgs = list_images(args.images, args.limit)
    model = load_model(args.model, args.device_id)
    try:
        nc = nc_of(model, args.nc)
        delta = float(args.delta) if args.delta else None
        if delta is None and args.scale_json:
            from acl_infer import load_scale_delta
            delta, _zp = load_scale_delta(args.scale_json)

        lam = float(args.lam) if args.lam is not None else None
        if lam is None and args.lam_from_dump:
            lam, n_pool = _lambda_from_dump(args.lam_from_dump, args.alpha)
            print(f"[thermal] λ={lam:.6g}（{n_pool} 个标定格，α={args.alpha}）")
        if lam is None:
            print("[thermal] 未给 --lam / --lam-from-dump：只测时延，P_fa 记为 nan")

        xs = [preprocess_image(p, args.imgsz)[0]
              for p in imgs[:min(32, len(imgs))]]
        fn = frame_fn_from_model(model, xs, nc, args.imgsz, delta, lam)
        runner = ThermalRunner(fn, args.minutes, args.window_min, SmiSampler(),
                              frames_per_window=args.frames_per_window)
        return runner.run()
    finally:
        model.destroy()


def _thermal_dry(args) -> list:
    from qcert.thermal import ThermalRunner, synthetic_frame_fn

    base = (float(args.dry_latency) if args.dry_latency > 0
            else (8.0 if args.impl == "fp16" else 4.0))
    fn = synthetic_frame_fn(latency_ms=base, pfa=1e-3, pfa_drift=args.dry_pfa_drift)
    clock, sleep = _fake_clock(step_s=0.02)
    runner = ThermalRunner(fn, args.minutes, args.window_min, None,
                           clock=clock, sleep=sleep, pace_s=0.02,
                           frames_per_window=args.frames_per_window)
    return runner.run()


# ------------------------------------------------------------------ CLI
def _add_common(p):
    p.add_argument("--impl", default="int8", choices=["fp32", "fp16", "int8"])
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--nc", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--dry-run", action="store_true", help="不碰 pyACL，用假设值自检")
    p.add_argument("--dry-latency", type=float, default=0.0,
                   help="dry-run 的时延假设（ms），0=按 impl 取默认")


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    b = sub.add_parser("bench", help="纯推理延迟 + 整板功率（E7 输入）")
    _add_common(b)
    b.add_argument("--model", default=None)
    b.add_argument("--images", default=None)
    b.add_argument("--device-id", type=int, default=0)
    b.add_argument("--iters", type=int, default=200)
    b.add_argument("--warmup", type=int, default=10)
    b.add_argument("--power", action="store_true", default=True,
                   help="采样 npu-smi 功率（默认开）")
    b.add_argument("--no-power", dest="power", action="store_false")
    b.add_argument("--dry-power", type=float, default=12.0)
    b.add_argument("--merge-into", default=None,
                   help="把 mean_ms 写进 {impl: ms} 的 json（run_e7 --latency-json）")
    b.add_argument("--out-json", default=None)

    t = sub.add_parser("thermal", help="热稳态长跑（E6）")
    _add_common(t)
    t.add_argument("--model", default=None)
    t.add_argument("--images", default=None)
    t.add_argument("--device-id", type=int, default=0)
    t.add_argument("--minutes", type=float, default=40.0)
    t.add_argument("--window-min", type=float, default=5.0)
    t.add_argument("--frames-per-window", type=int, default=200)
    t.add_argument("--alpha", type=float, default=1e-3)
    t.add_argument("--lam", type=float, default=None, help="直接给阈值 λ")
    t.add_argument("--lam-from-dump", default=None,
                   help="用标定 dump 汇总取 1−α 分位作为 λ")
    t.add_argument("--scale-json", default=None, help="INT8 输出层 delta")
    t.add_argument("--delta", type=float, default=None)
    t.add_argument("--dry-pfa-drift", type=float, default=0.0,
                   help="dry-run 的 P_fa 漂移强度（0=稳定）")
    t.add_argument("--out-dir", default=str(RESULTS / "e6"))
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.mode == "bench":
        if args.dry_run:
            info = _bench_dry(args)
        else:
            if not (args.model and args.images):
                raise SystemExit("板上模式需要 --model 与 --images；自检请加 --dry-run")
            info = _bench_real(args)

        print(f"[bench] {info['impl']} mean={info['mean_ms']:.2f}ms "
              f"p50={info['p50_ms']:.2f} p95={info['p95_ms']:.2f} "
              f"p99={info['p99_ms']:.2f} P99/mean={info['p99_over_mean']:.3f} "
              f"功率={info['power_mean_w']:.2f}W ({info['source']})")
        if args.out_json:
            json_dump(info, args.out_json)
            print(f"[bench] -> {args.out_json}")
        if args.merge_into:
            try:
                lat = json_load(args.merge_into)
            except FileNotFoundError:
                lat = {}
            lat[info["impl"]] = info["mean_ms"]
            json_dump(lat, args.merge_into)
            print(f"[bench] {args.merge_into} += {{{info['impl']!r}: "
                  f"{info['mean_ms']:.3f}}}")
        return info

    # ---- thermal
    from experiments.common import save_table
    from qcert.thermal import thermal_table, window_drift

    rows = _thermal_dry(args) if args.dry_run else _thermal_real(args)
    drift = window_drift(rows)
    out = ensure_dir(args.out_dir)
    source = "dry-run" if args.dry_run else "board"
    save_table(thermal_table(rows), out, "table5b_e6",
               f"Table 5 下 · E6 热稳态（{source}）")
    json_dump({"source": source, "minutes": args.minutes,
               "window_min": args.window_min,
               "frames_per_window": args.frames_per_window,
               "alpha": args.alpha, "impl": args.impl,
               "drift": drift,
               "verdict": ("PASS" if (drift["pass_pfa"] and drift["pass_p99"])
                           else ("INSUFFICIENT" if drift["insufficient"] else "FAIL")),
               "note": ("dry-run：假时钟 + 假帧源，只验证代码通路" if args.dry_run
                        else "板级实测")},
              out / "summary.json")

    print(f"\n=== E6 · 热稳态（{source}）===")
    print(f"  窗口 {drift['n_windows']}  有效 P_fa 窗 {drift['n_pfa_windows']}")
    print(f"  P_fa 漂移 |log10| = {drift['pfa_drift_log10']:.3f} "
          f"(<0.15 → {drift['pass_pfa']})")
    print(f"  P99 漂移 = {drift['p99_drift_rel']:.3%} (<15% → {drift['pass_p99']})")
    print(f"产物: {out}")
    return {"rows": rows, "drift": drift}


if __name__ == "__main__":
    main()
