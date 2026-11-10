"""E6 · 热稳态 40 min：每 5 min 一窗，P_fa 漂移 |log10| < 0.15。

协议依据 paper/04 §9。这块板是**无风扇被动散热**，NPU 降频会改变
算子的执行路径（不是数值）→ 若 P_fa 随温度漂，说明"契约"在长时间运行下守不住。
只有真板能测（E6 是五组板级独有实验之一）。

真板::

    python -m experiments.run_e6 --om models/yolov8n_int8.om \
        --image data/frame.png --minutes 40 --window 5

离线自检（不用板子，验证代码路径与判定逻辑）::

    python -m experiments.run_e6 --dry-run --minutes 1 --window 0.5
"""
from __future__ import annotations

import argparse

import numpy as np

from qcert.thermal import (SmiSampler, ThermalRunner, synthetic_frame_fn,
                           thermal_table, window_drift)

from .common import RESULTS, ensure_dir, json_dump, save_table


def _board_frame_fn(om, image, alpha, imgsz=640):
    """真板帧源：一次前向 -> 分数场 -> 固定 λ 下的瞬时 P_fa。

    ``acl`` 只在真正调用时才 import；本文件的顶层不依赖板端环境。
    """
    import sys
    import time
    from pathlib import Path

    deploy = Path(__file__).resolve().parents[2] / "asc-sar" / "deploy"
    sys.path.insert(0, str(deploy))
    from acl_infer import AclModel, preprocess          # noqa: WPS433
    from qcert.pfa_eval import build_omega, cell_centers, calibrate_lambda, measure_pfa

    model = AclModel(om)
    x, ratio, pad = preprocess(image, imgsz=imgsz)
    meta = {"ratio": ratio, "pad": pad, "imgsz": imgsz}
    raw = [a for a, _ in model.infer_raw(x)]
    from postprocess import decode_dense

    prob = decode_dense(raw)["prob"].astype(np.float64)
    from qcert.scores import ScoreField

    field = ScoreField(prob=prob.astype(np.float32), centers=None,
                       meta=dict(meta, imgsz=imgsz))
    # 只用网格几何，不依赖 head_sizes（板端 decode 已给出顺序）
    centers = cell_centers(np.asarray([[imgsz // 8, imgsz // 8],
                                       [imgsz // 16, imgsz // 16],
                                       [imgsz // 32, imgsz // 32]]), imgsz)
    _, mask_meas, info = build_omega(centers, None, guard_px=0.0,
                                     block_px=64.0, keep=1.0, seed=0)
    lam = float(np.quantile(prob[mask_meas], 1.0 - alpha))

    def fn():
        t0 = time.perf_counter()
        r = [a for a, _ in model.infer_raw(x)]
        dt = (time.perf_counter() - t0) * 1000.0
        p = decode_dense(r)["prob"].astype(np.float64)
        return {"t_model_ms": dt, "pfa": measure_pfa(p, mask_meas, lam)}

    return fn


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--om", default=None)
    ap.add_argument("--image", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="用合成帧源验证流程（不碰板子）")
    ap.add_argument("--minutes", type=float, default=40.0)
    ap.add_argument("--window", type=float, default=5.0)
    ap.add_argument("--alpha", type=float, default=1e-3)
    ap.add_argument("--latency-ms", type=float, default=8.0,
                    help="dry-run 的模拟时延")
    ap.add_argument("--pfa-drift", type=float, default=0.0,
                    help="dry-run 的模拟漂移（1.0 = 窗内 +100%）")
    ap.add_argument("--no-smi", action="store_true", help="板外环境跳过 npu-smi")
    ap.add_argument("--frames-per-window", type=int, default=200,
                    help="每窗统计帧数（04 §9 规定 200；0 = 不限）")
    ap.add_argument("--out", default=str(RESULTS / "e6"))
    args = ap.parse_args(argv)

    out = ensure_dir(args.out)
    if args.dry_run or not (args.om and args.image):
        if not args.dry_run:
            print("[warn] 缺 --om/--image，转 --dry-run")
        frame_fn = synthetic_frame_fn(latency_ms=args.latency_ms,
                                      pfa=args.alpha,
                                      pfa_drift=args.pfa_drift)
        with_smi = False
        source = "dry-run"
        # 合成帧源是瞬时的：按模拟时延节流，否则一个窗口能刷出上千万帧
        pace_s = max(args.latency_ms, 0.1) / 1000.0
    else:
        frame_fn = _board_frame_fn(args.om, args.image, args.alpha)
        with_smi = not args.no_smi
        source = "board"
        pace_s = None

    sampler = SmiSampler() if with_smi else None
    rows = ThermalRunner(frame_fn, args.minutes, args.window,
                         sampler=sampler, pace_s=pace_s,
                         frames_per_window=args.frames_per_window or None).run()
    drift = window_drift(rows)
    verdict = ("INSUFFICIENT" if drift["insufficient"]
               else ("PASS" if (drift["pass_pfa"] and drift["pass_p99"]) else "DRIFT"))
    summary = {
        "minutes": args.minutes, "window_min": args.window, "alpha": args.alpha,
        "frames_per_window": args.frames_per_window, "source": source,
        **drift, "verdict": verdict,
    }
    save_table(thermal_table(rows), out, "e6_windows", "E6 逐窗热稳态")
    save_table([{"pfa_drift_log10": drift["pfa_drift_log10"],
                 "p99_drift_rel": drift["p99_drift_rel"],
                 "n_pfa_windows": drift["n_pfa_windows"],
                 "pass_pfa": drift["pass_pfa"], "pass_p99": drift["pass_p99"],
                 "verdict": verdict, "source": source}],
               out, "table5b_e6", "Table 5 下 · E6 热稳态漂移")
    json_dump(summary, out / "summary.json")

    print("\n=== E6 · 热稳态 ===")
    print(f"窗口数 {len(rows)}  时长 {args.minutes} min  "
          f"有效 P_fa 窗 {drift['n_pfa_windows']}  来源={source}")
    print(f"P_fa 漂移 |log10| = {drift['pfa_drift_log10']:.3f} "
          f"(<0.15 → {drift['pass_pfa']})")
    print(f"P99 漂移 = {drift['p99_drift_rel']:.3f} (<0.15 → {drift['pass_p99']})")
    print(f"判决: {verdict}  产物: {out}")
    return summary


if __name__ == "__main__":
    main()
