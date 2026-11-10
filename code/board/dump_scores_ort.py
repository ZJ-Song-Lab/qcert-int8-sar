"""模块 1c：主机端 onnxruntime INT8 dump（E3 的 ORT 腿）。

与板上 ``dump_scores.py`` 同构：同一份 letterbox 预处理（``asc-sar/deploy``
的 ``postprocess``）、同一个 ``decode_dense``、同一个 .npz 格式，只是推理
后端从 pyACL(Ascend310B) 换成 onnxruntime(CPU)。

模型用 AMCT 产出的 ``*_sim.onnx``（标准 Q/DQ 仿真图）——与 atc 编进 .om 的
AscendQuant 版**同源同 scale**（MD5 已核对的是 AscendQuant 版，sim 版由同一
次量化产出），正好构成 E3 要的"同一权重、不同后端"对照。

用法（在 code/ 目录下）::

    python -m board.dump_scores_ort --model ../asc-sar/models/yolo11n_ssdd_int8_sim.onnx \\
        --images ../asc-sar/datasets/ssdd/images/test \\
        --labels ../asc-sar/datasets/ssdd/labels/test \\
        --out ../asc-sar/results/scores_ssdd_int8_ort
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:                                   # python -m board.dump_scores_ort
    from .runner import list_images
except ImportError:                    # 直接 python board/dump_scores_ort.py
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from runner import list_images  # noqa: F401


def _bootstrap():
    """把 code/ 根与 asc-sar/deploy 放进 sys.path（qcert / postprocess）。"""
    here = Path(__file__).resolve()
    root = here.parents[1]                 # code/
    deploy = here.parents[2] / "asc-sar" / "deploy"
    for p in (str(root), str(deploy)):
        if p not in sys.path:
            sys.path.insert(0, p)


def main(argv=None):
    _bootstrap()
    import onnxruntime as ort

    from postprocess import decode_dense, load_image_rgb, preprocess
    from qcert.scores import ScoreField, save_score_npz

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="AMCT *_sim.onnx（Q/DQ 仿真图）")
    ap.add_argument("--images", required=True, help="测试图像目录")
    ap.add_argument("--labels", default=None, help="YOLO 标签目录（写进 meta）")
    ap.add_argument("--out", required=True, help="npz 输出目录")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--nc", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]

    from experiments.common import gt_for

    imgs = list_images(args.images, args.limit)
    print(f"[ort-dump] {len(imgs)} 张图 model={Path(args.model).name} "
          f"providers={sess.get_providers()}")
    for i, p in enumerate(imgs, 1):
        img = load_image_rgb(p)
        ih, iw = img.shape[:2]
        x, ratio, pad = preprocess(img, args.imgsz)
        meta = {"imgsz": args.imgsz, "iw": int(iw), "ih": int(ih),
                "ratio": float(ratio), "pad": [int(v) for v in pad],
                "impl": "int8_ort", "nc": args.nc,
                "model": Path(args.model).name}
        if args.labels:
            b = gt_for(p.stem, args.labels, meta)
            meta["boxes"] = (b.tolist() if b is not None and len(b) else [])
        ys = sess.run(out_names, {in_name: x})
        field = decode_dense([np.asarray(y, dtype=np.float32) for y in ys],
                             args.nc, args.imgsz)
        sf = ScoreField(prob=field["prob"], logit=field["logit"],
                        head_sizes=np.asarray(field["head_sizes"], dtype=np.int32),
                        meta=meta, stem=p.stem)
        save_score_npz(out / f"{p.stem}.npz", sf)
        if i % 20 == 0 or i == len(imgs):
            print(f"  [{i}/{len(imgs)}]")

    info = {"source": "ort", "impl": "int8_ort", "n": len(imgs), "nc": args.nc,
            "model": str(args.model), "providers": sess.get_providers(),
            "out": str(out)}
    (out / "manifest.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ort-dump] n={info['n']} -> {out}")
    return info


if __name__ == "__main__":
    main()
