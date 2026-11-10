"""模块 1：把稠密分数场落成 ``qcert`` 的 .npz（E0/E1/E4/E5/E9 的输入）。

两条通路：

- **板上**：``--model model.om``，走 pyACL，底层是
  ``acl_infer.dump_score_field``。INT8 时用 ``infer_raw`` 保住原始整数码
  ``s_q``，并把 ``delta / zero_point`` 一起落盘 —— v3 的叙事是"整数域"，
  原值必须可复现，浮点视图 ``prob`` 只是它的 dequantized 影子；
- **离线自检**：``--dry-run``，用 ``experiments.synth`` 造合成分数场，
  验证"下游 E0–E9 能直接吃这个目录"，不产生任何论文数据。

示例::

    # 板上：FP16 与 INT8 两套 dump（E7/E3 的输入）
    python -m board.dump_scores --model yolov8n_fp16.om \\
        --images data/ssdd/images/test --labels data/ssdd/labels/test \\
        --out dumps/fp16 --impl fp16
    python -m board.dump_scores --model yolov8n_int8.om \\
        --images data/ssdd/images/test --labels data/ssdd/labels/test \\
        --scale-json results/m0/scale.json --out dumps/int8 --impl int8

    # 离线：造一批 demo dump（含压测对比度 ×2，供 E2 走通）
    python -m board.dump_scores --dry-run --n 12 --out dumps/demo_fp32 --impl fp32
    python -m board.dump_scores --dry-run --n 12 --out dumps/demo_int8 --impl int8
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:                                   # python -m board.dump_scores
    from .runner import (ensure_dir, json_dump, list_images, load_model,
                         nc_of, preprocess_image)
except ImportError:                    # 直接 python board/dump_scores.py
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from runner import (ensure_dir, json_dump, list_images, load_model,  # noqa: F401
                        nc_of, preprocess_image)


# ------------------------------------------------------------------ 离线自检
def _dry_run(args, out: Path) -> dict:
    """写合成 dump：只验证通路，不是实验数据（meta 里 ``synth=True``）。"""
    from experiments.synth import delta_for_range, make_scene, to_field
    from qcert.scores import save_score_npz

    n = int(args.n)
    seas = ("calm", "moderate", "rough")
    scenes = [make_scene(seed=int(args.seed) + i, n_target=3,
                         sea_state=seas[i % len(seas)],
                         contrast=float(args.contrast))
              for i in range(n)]

    delta = None
    if args.impl == "int8":
        # 与 AMCT 一致：在"正常对比度"的一小批场景上标定一次 scale，之后不再变。
        delta = (float(args.delta) if args.delta
                 else float(delta_for_range([s["s"] for s in scenes[:min(8, n)]],
                                            bits=8)[0]))

    for i, sc in enumerate(scenes):
        f = to_field(sc, impl=args.impl, delta=delta, bits=8, stem=f"demo_{i:03d}")
        if args.impl == "fp16":        # 只有输出张量按 FP16 舍入，权重/激活不管
            f.prob = f.prob.astype("float16").astype("float32")
        save_score_npz(out / f"{f.stem}.npz", f)
    return {"source": "dry-run", "impl": args.impl, "n": n, "delta": delta,
            "zero_point": 0, "nc": 1, "contrast": float(args.contrast)}


# ------------------------------------------------------------------ 板上
def _dump_real(args, out: Path) -> dict:
    from acl_infer import bit_exact_repeat, dump_score_field, load_scale_delta
    from eval_map import load_yolo_gt

    imgs = list_images(args.images, args.limit)
    model = load_model(args.model, args.device_id)
    delta, zp = load_scale_delta(args.scale_json)
    rep = None
    try:
        nc = nc_of(model, args.nc)
        if args.impl == "int8" and delta is None:
            print("[dump] 警告：INT8 但未给 --scale-json，delta 退化为 1.0；"
                  "整数码 s_q 仍保真，浮点视图 prob 不可用于横向比较。")
        print(f"[dump] {len(imgs)} 张图 nc={nc} delta={delta} zp={zp} impl={args.impl}")
        for i, p in enumerate(imgs, 1):
            x, ratio, pad, (ih, iw) = preprocess_image(p, args.imgsz)
            meta = {"imgsz": args.imgsz, "iw": iw, "ih": ih, "ratio": ratio,
                    "pad": list(pad), "impl": args.impl, "nc": nc,
                    "contrast": float(args.contrast),
                    "model": Path(args.model).name}
            if args.labels:
                lb = Path(args.labels) / f"{p.stem}.txt"
                meta["boxes"] = load_yolo_gt(lb, iw, ih).tolist()
            dump_score_field(model, x, meta, nc, out / f"{p.stem}.npz",
                             args.imgsz, delta, zp, args.impl)
            if i % 20 == 0 or i == len(imgs):
                print(f"  [{i}/{len(imgs)}]")
        if args.repeat > 0:
            x = preprocess_image(imgs[0], args.imgsz)[0]
            rep = bit_exact_repeat(model, x, args.repeat)
            print(f"[A1] 同一输入重复 {rep['n']} 次：逐 bit 一致 "
                  f"{rep['n_exact']}/{rep['n']} = {rep['rate']:.4f}")
    finally:
        model.destroy()
    return {"source": "board", "impl": args.impl, "n": len(imgs), "nc": nc,
            "delta": delta, "zero_point": zp, "bit_exact": rep,
            "model": str(args.model)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="npz 输出目录")
    ap.add_argument("--impl", default="fp32", choices=["fp32", "fp16", "int8"])
    ap.add_argument("--imgsz", type=int, default=640)
    # 板上
    ap.add_argument("--model", default=None, help=".om 模型（板上）")
    ap.add_argument("--images", default=None, help="测试图像目录（板上）")
    ap.add_argument("--labels", default=None, help="YOLO 标签目录（可选）")
    ap.add_argument("--scale-json", default=None, help="ScaleTable json（INT8 必需）")
    ap.add_argument("--nc", type=int, default=0, help="类别数，0=按输出通道推断")
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 张（0=全部）")
    ap.add_argument("--repeat", type=int, default=0,
                    help="同一输入重复 N 次查逐 bit 一致（A1），0=不做")
    # 离线
    ap.add_argument("--dry-run", action="store_true", help="合成 dump，不碰 pyACL")
    ap.add_argument("--n", type=int, default=12, help="--dry-run 的场景数")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--contrast", type=float, default=1.0,
                    help="对比度标签/增益：板上模式写进 meta（E2 分组）；--dry-run 作压测增益")
    ap.add_argument("--delta", type=float, default=None,
                    help="--dry-run --impl int8 时强制指定 delta")
    args = ap.parse_args(argv)

    out = ensure_dir(args.out)
    if args.dry_run:
        info = _dry_run(args, out)
    else:
        if not (args.model and args.images):
            ap.error("板上模式需要 --model 与 --images；离线自检请加 --dry-run")
        info = _dump_real(args, out)

    info["out"] = str(out)
    json_dump(info, out / "manifest.json")
    print(f"[dump] source={info['source']} impl={info['impl']} "
          f"n={info['n']} delta={info['delta']} -> {out}")
    return info


if __name__ == "__main__":
    main()
