"""板上全量 cv3 dump：acl.dump 抓三个 cv3 终卷积输出（fp16 NC1HWC0）→ 分数场 npz。

背景：ATC 无法在 3.5GB 板上编译 cv3 多输出 OM（独占仍被 OOM Kill）；CANN 7.0 支持
``acl.init(config_path)`` 传入 dump 配置，对**现有 int8 OM** 推理时抓取任意算子输出，
无需重编译。dump 出的张量是融合卷积 dequant 后的 fp16 值（NC1HWC0 [1,1,H,W,16]，
lane0 有效、lane1..15 全零），即板上部署算术的真实分数场 ``v = fl16(q·Δ)``。

与 ORT int32 dump（dump_cv3_ort.py）的关系：
  * Δ 取自 scale json 无后缀激活条目（与 ORT 侧相同的三个值），zp=0；
  * ``q_hat = rint(v/Δ)``：|v|≤Δ·1024 区域与 int32 码逐位一致，大 |q| 区域因 fp16
    ulp > Δ 而塌缩到同一代表元——这是硬件真值的并列结构（E2/E5 输入）；
  * npz 键与 ORT 侧完全一致（prob/logit/head_sizes/s_q/delta/zero_point/meta），
    另加 ``s_hw``（fp16 原值），可按 stem 直接配对喂 experiments.run_e3 --hw。

用法（板上）::

    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    python3 /root/code/board/acl_dump_cv3.py --limit 2        # 冒烟
    python3 /root/code/board/acl_dump_cv3.py                  # 全量 232
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

CFG = "/root/acl_dump_cfg.json"
DUMP_DIR = Path("/dev/shm/acl_dump")  # tmpfs：362 个算子文件/图 ~160MB，写内存盘快 10 倍
MODEL = "/root/asc_sar/models/yolo11n_ssdd_int8.om"
IMG_DIR = Path("/root/asc_sar/data/images")
OUT_DIR = Path("/root/qcert_dumps/cv3_hw")
REPORT = Path("/root/qcert_dumps/cv3_hw_report.json")

# scale json 无后缀激活条目 /model.23/cv3.{0,1,2}/cv3.{0,1,2}.2/Conv_quant_dequant
HEAD_DELTAS = [6.938855e-4, 1.323877e-3, 5.517007e-4]
HEAD_HW = [80, 40, 20]
LAYER_NAMES = [f"/model.23/cv3.{i}/cv3.{i}.2/Conv_quant" for i in range(3)]
FILE_PREFIX = [f"Conv2D._model_23_cv3_{i}_cv3_{i}_2_Conv_quant" for i in range(3)]


def parse_head(fp: Path, hw: int) -> tuple[np.ndarray, int]:
    """dump 文件 -> (lane0 fp16 (n,), lane1..15 非零数)。尾长不足抛异常。"""
    b = fp.read_bytes()
    n = hw * hw
    off = len(b) - n * 32  # NC1HWC0 fp16: 每像素 16 lane × 2B
    if off < 0:
        raise ValueError(f"{fp.name} 太小: {len(b)} < {n * 32}")
    data = np.frombuffer(b, dtype="<f2", offset=off, count=n * 16).reshape(n, 16)
    return data[:, 0].copy(), int(np.count_nonzero(data[:, 1:]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--no-layer", action="store_true",
                    help="dump config 不加 layer 过滤（全算子 dump，慢但兜底）")
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--img-dir", default=str(IMG_DIR),
                    help="图像目录（默认 SSDD 232；压测集用 /root/asc_sar/data/stress_v2）")
    ap.add_argument("--report", default=str(REPORT))
    args = ap.parse_args()
    img_dir = Path(args.img_dir)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    DUMP_DIR.mkdir(parents=True, exist_ok=True)  # dump_path 必须预先存在

    item = {"model_name": "yolo11n_ssdd_int8"}
    if not args.no_layer:
        item["layer"] = LAYER_NAMES
    # layer 过滤与 dump_op_switch 互斥：过滤模式必须 op_switch=off
    cfg = {"dump": {"dump_path": str(DUMP_DIR), "dump_mode": "output",
                    "dump_op_switch": "off" if not args.no_layer else "on",
                    "dump_list": [item]}}
    Path(CFG).write_text(json.dumps(cfg))
    print("[dump] config ->", CFG, "layer_filter =", not args.no_layer, flush=True)

    import acl
    r = acl.init(CFG)
    print("[dump] acl.init:", r, flush=True)

    sys.path.insert(0, "/root/asc_sar/deploy")
    from acl_infer import _status, _val  # noqa: E402
    from postprocess import load_image_rgb, preprocess  # noqa: E402

    _status(acl.rt.set_device, 0)
    ctx = _val(acl.rt.create_context, 0)
    model_id = _val(acl.mdl.load_from_file, MODEL)
    print("[dump] model_id =", model_id, flush=True)

    imgs = sorted(p for p in img_dir.iterdir()
                  if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"))
    if args.limit:
        imgs = imgs[: args.limit]
    print(f"[dump] {len(imgs)} 张待跑 -> {out_dir}", flush=True)

    # 输入/输出 dataset 只建一次（静态 shape）
    x0, _, _ = preprocess(load_image_rgb(imgs[0]), args.imgsz)
    x0 = np.ascontiguousarray(x0, dtype=np.float32)
    in_ds = _val(acl.mdl.create_dataset)
    in_ptr = _val(acl.rt.malloc, x0.nbytes, 0)
    _status(acl.mdl.add_dataset_buffer, in_ds,
            _val(acl.create_data_buffer, in_ptr, x0.nbytes))
    from acl_infer import _get_desc
    desc = _get_desc(model_id)
    n_out = _val(acl.mdl.get_num_outputs, desc)
    out_ds = _val(acl.mdl.create_dataset)
    for i in range(n_out):
        sz = _val(acl.mdl.get_output_size_by_index, desc, i)
        ptr = _val(acl.rt.malloc, sz, 0)
        _status(acl.mdl.add_dataset_buffer, out_ds,
                _val(acl.create_data_buffer, ptr, sz))

    stats = {h: {"uniq_v": [], "uniq_q": [], "lane_nz": 0} for h in range(3)}
    n_files_log, t0 = [], time.time()
    for n, p in enumerate(imgs, 1):
        stem = p.stem
        # 清上一次 dump 产物
        for d in DUMP_DIR.iterdir():
            if d.is_dir():
                shutil.rmtree(d, ignore_errors=True)
            else:
                d.unlink(missing_ok=True)

        img = load_image_rgb(p)
        ih, iw = img.shape[:2]
        x, ratio, pad = preprocess(img, args.imgsz)
        x = np.ascontiguousarray(x, dtype=np.float32)
        _status(acl.rt.memcpy, in_ptr, x.nbytes, x.ctypes.data, x.nbytes, 1)
        r = acl.mdl.execute(model_id, in_ds, out_ds)
        if r != 0:
            print(f"[dump][{stem}] execute 失败 r={r}，跳过", flush=True)
            continue
        sync = getattr(acl.rt, "synchronize_context", None)
        if sync is not None:
            sync(ctx)

        files = [f for f in DUMP_DIR.rglob("*") if f.is_file()]
        n_files_log.append(len(files))
        vs, qs = [], []
        ok = True
        for h in range(3):
            cand = [f for f in files if f.name.startswith(FILE_PREFIX[h])]
            if len(cand) != 1:
                print(f"[dump][{stem}] head{h} 匹配 {len(cand)} 个文件，跳过；"
                      f"样例: {[f.name[:60] for f in files[:5]]}", flush=True)
                ok = False
                break
            v, lane_nz = parse_head(cand[0], HEAD_HW[h])
            stats[h]["lane_nz"] += lane_nz
            stats[h]["uniq_v"].append(int(np.unique(v).size) / v.size)
            q = np.rint(v.astype(np.float64) / HEAD_DELTAS[h]).astype(np.int32)
            stats[h]["uniq_q"].append(int(np.unique(q).size) / q.size)
            vs.append(v)
            qs.append(q)
        if not ok:
            continue
        v_all = np.concatenate(vs)          # fp16 (8400,)
        q_all = np.concatenate(qs)          # int32 (8400,)
        logit = v_all.astype(np.float32)
        p64 = np.clip(v_all.astype(np.float64), -60, 60)
        prob = (1.0 / (1.0 + np.exp(-p64))).astype(np.float32)
        payload = {
            "prob": prob,
            "logit": logit,
            "head_sizes": np.asarray([[80, 80], [40, 40], [20, 20]], dtype=np.int32),
            "s_q": q_all,
            "s_hw": v_all,                  # fp16 原值（硬件真值）
            "delta": np.array(HEAD_DELTAS[0], dtype=np.float32),
            "zero_point": np.array(0, dtype=np.int32),
            "meta": np.array(json.dumps({
                "iw": int(iw), "ih": int(ih), "imgsz": args.imgsz,
                "ratio": float(ratio), "pad": [float(pad[0]), float(pad[1])],
                "impl": "board_fp16_cv3_acldump", "head_deltas": HEAD_DELTAS,
                "model": Path(MODEL).name,
                "note": "acl.dump 抓融合卷积 dequant 输出（fp16 NC1HWC0 lane0）；"
                        "s_q=rint(v/Δ)，|v|>Δ·1024 区域为 fp16 网格代表元"},
                ensure_ascii=False)),
        }
        np.savez_compressed(out_dir / f"{stem}.npz", **payload)
        if n % 20 == 0 or n == len(imgs):
            el = time.time() - t0
            print(f"  [{n}/{len(imgs)}] {el:.0f}s  files/img={np.mean(n_files_log):.0f}",
                  flush=True)

    acl.mdl.unload(model_id)
    acl.rt.destroy_context(ctx)
    acl.rt.reset_device(0)
    acl.finalize()

    report = {"model": Path(MODEL).name, "runtime": "acl.dump fp16 NC1HWC0",
              "n_images": len(list(out_dir.glob('*.npz'))),
              "layer_filter": not args.no_layer,
              "files_per_img_mean": float(np.mean(n_files_log)) if n_files_log else 0,
              "head_deltas": HEAD_DELTAS, "heads": {}}
    for h in range(3):
        report["heads"][f"head{h}"] = {
            "delta": HEAD_DELTAS[h],
            "uniq_frac_v_fp16": float(np.mean(stats[h]["uniq_v"])) if stats[h]["uniq_v"] else 0,
            "uniq_frac_q_hat": float(np.mean(stats[h]["uniq_q"])) if stats[h]["uniq_q"] else 0,
            "lane_nonzero_total": stats[h]["lane_nz"]}
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print("[dump] report ->", args.report, flush=True)
    for h in range(3):
        r = report["heads"][f"head{h}"]
        print(f"[head{h}] Δ={r['delta']:.6e} uniq_v={r['uniq_frac_v_fp16']:.4f} "
              f"uniq_q={r['uniq_frac_q_hat']:.4f} lane_nz={r['lane_nonzero_total']}",
              flush=True)


if __name__ == "__main__":
    main()
