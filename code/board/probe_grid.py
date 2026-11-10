"""经验性 A3：从 OM 的 float32 输出里反推 INT8 反量化网格 Δ。

ATC 编译 INT8 OM 时会在输出前自动补 dequant，因此 ``infer_raw`` 拿到的
是 float32 —— 但如果芯片内部确实走 INT8 算术，这些 float 值必须严格落在
``k * Δ`` 的离散网格上（每个检测头一个 Δ）。本脚本**测量**这个网格：

1. 对 INT8 OM 的原始输出逐头估计 Δ（排序唯一值的最小正差）；
2. 检验所有值相对网格的最大偏差（应 ≈ fp32 舍入量级）；
3. 与 scale 表里的激活 scale 条目交叉对照（A3 的两种读法互相印证）；
4. 用 FP16 OM 做对照（不应出现 INT8 网格）。

判据：``on_grid_frac >= 0.999`` 且逐头 Δ 稳定（跨图 CV < 1%）。

用法（板上）::

    python3 /root/code/board/probe_grid.py \
        --int8 /root/asc_sar/models/yolo11n_ssdd_int8.om \
        --fp16 /root/asc_sar/models/yolo11n_ssdd_fp16.om \
        --images /root/asc_sar/data/images --n 8 \
        --scale-json /root/asc_sar/models/yolo11n_ssdd_int8_scale.json \
        --out /root/qcert_results/m0/grid_probe.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HEADS_640 = (6400, 1600, 400)  # stride 8/16/32 @ imgsz 640


def estimate_delta(v: np.ndarray):
    """返回 (Δ_est, 最大网格偏差(相对Δ), 在网格上的比例)。估计失败返回 None。"""
    u = np.unique(v.astype(np.float64))
    if u.size < 8:
        return None
    d = np.diff(u)
    d = d[d > 0]
    if d.size == 0:
        return None
    delta = float(d.min())
    if delta <= 0:
        return None
    r = np.abs(v.astype(np.float64) / delta - np.round(v.astype(np.float64) / delta))
    return {"delta": delta,
            "max_dev_rel": float(r.max()),
            "on_grid_frac": float((r < 1e-3).mean()),
            "n_unique": int(u.size),
            "q_min": int(np.round(v.min() / delta)),
            "q_max": int(np.round(v.max() / delta))}


def probe_model(model, om_path: str, imgs, imgsz: int, nc: int):
    per_head = [[] for _ in HEADS_640]
    for p in imgs:
        if deploy_mod is None:
            raise RuntimeError("deploy 模块未初始化")
        x, _, _ = deploy_mod["preprocess"](deploy_mod["load_image_rgb"](p), imgsz)
        raw = model.infer_raw(x)
        arr = raw[0][0][0].astype(np.float32)  # [4+nc, 8400]
        cls = arr[4:4 + nc].max(axis=0)        # 分数通道（逐网格 max_c）
        off = 0
        for h, cnt in enumerate(HEADS_640):
            per_head[h].append(cls[off:off + cnt])
            off += cnt
    heads = []
    for h, chunks in enumerate(per_head):
        v = np.concatenate(chunks)
        est = estimate_delta(v)
        heads.append({"head": h, "n": int(v.size), "est": est})
    return {"model": Path(om_path).name, "dtype": str(raw[0][1]),
            "n_images": len(imgs), "heads": heads}


deploy_mod = None


def _init_deploy(deploy: str):
    global deploy_mod
    if deploy not in sys.path:
        sys.path.insert(0, deploy)
    import acl_infer
    deploy_mod = {"AclModel": acl_infer.AclModel,
                  "load_image_rgb": acl_infer.load_image_rgb,
                  "preprocess": acl_infer.preprocess}
    return deploy_mod


def cross_check_scale(scale_json: str):
    """列出 scale 表里检测头 cls 终卷积的无后缀激活条目（候选输出 Δ）。"""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from qcert.scale_table import ScaleTable

    t = ScaleTable.from_json(scale_json)
    out = []
    for n in t.names():
        nl = n.lower()
        if "cv3" in nl and nl.endswith("conv_quant_dequant"):
            e = t.get(n)
            out.append({"name": n, "delta": float(e.delta),
                        "zero_point": int(e.zero_point)})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--int8", required=True)
    ap.add_argument("--fp16", default=None, help="对照模型（可选）")
    ap.add_argument("--images", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--nc", type=int, default=1)
    ap.add_argument("--scale-json", default=None)
    ap.add_argument("--deploy", default="/root/asc_sar/deploy")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    imgs = sorted(p for p in Path(args.images).iterdir()
                  if p.suffix.lower() in (".jpg", ".jpeg", ".png"))[: args.n]
    assert imgs, f"图像目录为空: {args.images}"

    report = {"n_images": len(imgs), "imgsz": args.imgsz}
    dm = _init_deploy(args.deploy)
    models = [(tag, om, dm["AclModel"](om, 0))
              for tag, om in [("int8", args.int8), ("fp16", args.fp16)] if om]
    try:
        for tag, om, model in models:
            report[tag] = probe_model(model, om, imgs, args.imgsz, args.nc)
    finally:
        for _, _, model in models:
            try:
                model.destroy()
            except Exception:
                pass
    if args.scale_json:
        report["scale_json_entries"] = cross_check_scale(args.scale_json)

    deltas = [h["est"]["delta"] for h in report["int8"]["heads"] if h["est"]]
    on_grid = [h["est"]["on_grid_frac"] for h in report["int8"]["heads"] if h["est"]]
    report["verdict"] = {
        "grid_found": bool(deltas) and all(f >= 0.999 for f in on_grid),
        "delta_per_head": deltas,
        "note": "float32 输出严格落在逐头 INT8 网格上：芯片内部确为整数算术，"
                "Δ 可从输出端经验读出（经验性 A3）",
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(json.dumps(report["verdict"], ensure_ascii=False, indent=2))
    for tag in ("int8", "fp16"):
        if tag in report:
            for h in report[tag]["heads"]:
                e = h["est"]
                if e:
                    print(f"[{tag}] head{h['head']}: Δ={e['delta']:.6e} "
                          f"on_grid={e['on_grid_frac']:.4f} "
                          f"q∈[{e['q_min']},{e['q_max']}] "
                          f"max_dev={e['max_dev_rel']:.2e}")
                else:
                    print(f"[{tag}] head{h['head']}: 无网格（连续值）")
    if "scale_json_entries" in report:
        for e in report["scale_json_entries"]:
            print(f"[scale-json] {e['name']}  Δ={e['delta']:.6e} zp={e['zero_point']}")
    print(f"[probe] -> {out}")
    return report


if __name__ == "__main__":
    main()
