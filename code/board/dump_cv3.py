"""板端：用 cv3 输出 OM dump 整数域网格分数场（饱和主线 E2/E5 的数据源）。

背景：ATC 编译 INT8 OM 时图内自带 float sigmoid，最终输出不在整数网格上
（grid_probe 否定性结果）。``extract_cv3.py`` 把三个检测头 cls 终卷积的
dequant 输出（sigmoid 之前、最后一个整数域张量）挂成图输出，编译为
``*_cv3.om``（4 个输出：output0 + 3 个 ``[1,1,H,W]`` 网格张量）。

本脚本对每张图：

1. 读 3 个网格张量 ``v``（float32，理论值 ``q * Δ_h``）；
2. **网格验证**：``|v/Δ - round(v/Δ)|`` 应在 fp32 舍入量级（经验性 A3 正面证据）；
3. 恢复整数码 ``s_q = round(v/Δ)``——**int32 累加器网格，不截断**（数值取证：
   sigmoid 前无 int8 重量化，q∈±万级，clip 到 ±127 会毁掉格点）；
4. 写 qcert ScoreField npz（``prob=sigmoid(v)``、``logit=v``、``s_q``、``delta``、
   ``meta.head_deltas``），与 experiments.run_e2 / run_e5 直接兼容；
5. 汇总网格验证 + 格点占用统计（uniq 码数；该网格不触顶，饱和发生在 neck
   的 int8 特征层，由 ``--dump-layers`` 链路审计）到 report json。

用法（板上）::

    python3 /root/code/board/dump_cv3.py \
        --model /root/asc_sar/models/yolo11n_ssdd_int8_cv3.om \
        --images /root/asc_sar/data/images \
        --scale-json /root/asc_sar/models/yolo11n_ssdd_int8_scale.json \
        --out /root/qcert_dumps/cv3_int8 \
        --report /root/qcert_results/m0/cv3_grid.json
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np

try:                                   # python -m board.dump_cv3
    from .runner import ensure_dir, json_dump, list_images, load_model, preprocess_image
except ImportError:                    # 直接 python board/dump_cv3.py
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from runner import ensure_dir, json_dump, list_images, load_model, preprocess_image

HEAD_HW = ((80, 80), (40, 40), (20, 20))  # stride 8/16/32 @640，与 decode_dense 顺序一致


def head_deltas_from_scale(scale_json: str) -> list[float]:
    """从 scale 表取三个 cv3 终卷积**激活** Δ（按头 0/1/2 排序）。

    只接受无 ``[k]`` 后缀的激活条目 ``/model.23/cv3.{h}/cv3.{h}.2/Conv_quant_dequant``；
    带后缀的是逐通道权重 scale，误用会让网格验证假阴性。
    """
    from qcert.scale_table import ScaleTable

    t = ScaleTable.from_json(scale_json)
    found = {}
    for h in (0, 1, 2):
        name = f"/model.23/cv3.{h}/cv3.{h}.2/Conv_quant_dequant"
        e = t.get(name)
        if e is not None:
            found[h] = float(e.delta)
    if sorted(found) != [0, 1, 2]:
        cand = [n for n in t.names()
                if re.search(r"/model\.23/cv3\.\d/cv3\.\d\.2/", n)][:20]
        raise RuntimeError(
            f"scale 表缺 cv3 终卷积激活条目（找到 {sorted(found)}）；"
            f"候选名示例: {cand}")
    return [found[h] for h in (0, 1, 2)]


def grid_output_indices(model) -> list[int]:
    """按空间尺寸识别三个网格输出（80/40/20 降序）的输出索引。"""
    grid = []
    for i, d in enumerate(model.out_dims):
        dd = [int(x) for x in d]
        if len(dd) == 4 and dd[1] == 1:
            grid.append((dd[2], i))       # (H, output_index)
    grid.sort(reverse=True)
    assert [g[0] for g in grid] == [h for h, _ in HEAD_HW], \
        f"网格输出尺寸不符（期望 80/40/20）: {grid}；全部 out_dims={model.out_dims}"
    return [i for _, i in grid]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--scale-json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--contrast", type=float, default=1.0,
                    help="压测集对比度标签（写进 meta，E2 按它分组）")
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help=">0 时只跑前 N 张（冒烟）")
    ap.add_argument("--deploy", default="/root/asc_sar/deploy",
                    help="板上 deploy 目录（runner 的相对路径推断失效时兜底）")
    args = ap.parse_args(argv)

    if args.deploy not in sys.path and Path(args.deploy).is_dir():
        sys.path.insert(0, args.deploy)
    from qcert.scores import ScoreField, save_score_npz

    deltas = head_deltas_from_scale(args.scale_json)
    print(f"[cv3] 逐头激活 Δ = {deltas}")

    imgs = list_images(args.images, args.limit)
    out_dir = ensure_dir(args.out)

    model = load_model(args.model, args.device_id)
    print(f"[cv3] 模型输出数 {len(model.out_dims)}: {model.out_dims}")
    grid_idx = grid_output_indices(model)
    print(f"[cv3] 网格输出索引 {grid_idx}（80/40/20 顺序）")

    stats = {h: {"on_grid": [], "uniq": [], "qmin": [], "qmax": [], "dev": []}
             for h in (0, 1, 2)}
    t0 = time.time()
    try:
        for n, p in enumerate(imgs, 1):
            x, ratio, pad, (ih, iw) = preprocess_image(p, args.imgsz)
            arrs = [a for a, _ in model.infer_raw(x)]
            logits, s_q_all = [], []
            for h, gi in enumerate(grid_idx):
                v = arrs[gi].astype(np.float32).ravel()
                r = v.astype(np.float64) / deltas[h]
                q = np.rint(r)
                stats[h]["on_grid"].append(float((np.abs(r - q) < 1e-2).mean()))
                stats[h]["dev"].append(float(np.abs(r - q).max()) if r.size else 0.0)
                stats[h]["uniq"].append(int(np.unique(q).size))
                stats[h]["qmin"].append(int(q.min()))
                stats[h]["qmax"].append(int(q.max()))
                logits.append(v)
                s_q_all.append(q.astype(np.int32))  # 累加器网格码，不截断
            logit = np.concatenate(logits)
            q_cat = np.concatenate(s_q_all)
            p64 = np.clip(logit.astype(np.float64), -60, 60)
            prob = (1.0 / (1.0 + np.exp(-p64))).astype(np.float32)
            sf = ScoreField(
                prob=prob, logit=logit.astype(np.float32), centers=None,
                head_sizes=np.asarray(HEAD_HW, dtype=np.int32).reshape(-1),
                s_q=q_cat, delta=deltas[0], zero_point=0,
                meta={"iw": iw, "ih": ih, "imgsz": args.imgsz,
                      "ratio": float(ratio), "pad": list(map(float, pad)),
                      "impl": "int32_cv3", "head_deltas": deltas,
                      "contrast": float(args.contrast),
                      "model": Path(args.model).name,
                      "note": "cv3 终卷积 dequant 输出（sigmoid 前的整数域 logit）"},
                stem=p.stem)
            save_score_npz(out_dir / f"{p.stem}.npz", sf)
            if n % 50 == 0 or n == len(imgs):
                print(f"  [{n}/{len(imgs)}] {time.time() - t0:.0f}s")
    finally:
        model.destroy()

    report = {"model": Path(args.model).name, "n_images": len(imgs),
              "deltas": deltas, "heads": {}}
    ok = True
    for h in (0, 1, 2):
        s = stats[h]
        head_ok = float(np.mean(s["on_grid"])) >= 0.999
        ok = ok and head_ok
        report["heads"][f"head{h}"] = {
            "delta": deltas[h],
            "on_grid_frac": float(np.mean(s["on_grid"])),
            "max_dev": float(np.max(s["dev"])),
            "uniq_frac": float(np.mean([u / (HEAD_HW[h][0] * HEAD_HW[h][1])
                                        for u in s["uniq"]])),
            "q_min": int(np.min(s["qmin"])), "q_max": int(np.max(s["qmax"]))}
    report["verdict"] = {
        "grid_found": bool(ok),
        "note": "on_grid_frac>=0.999 且 Δ 与 scale 表一致 → 整数网格可读出"
                "（经验性 A3 正面证据）；网格为 int32 累加器码（不触顶），"
                "uniq_frac 为格点占用（E2 输入），饱和审计看 neck int8 层"}
    json_dump(report, args.report)
    print(f"[verdict] {report['verdict']}")
    for h in (0, 1, 2):
        r = report["heads"][f"head{h}"]
        print(f"[head{h}] Δ={r['delta']:.6e} on_grid={r['on_grid_frac']:.4f} "
              f"dev_max={r['max_dev']:.2e} uniq={r['uniq_frac']:.4f} "
              f"q∈[{r['q_min']},{r['q_max']}]")
    print(f"[cv3] dumps -> {out_dir}  report -> {args.report}")
    return report


if __name__ == "__main__":
    main()
