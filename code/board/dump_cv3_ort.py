"""本机（x86）：用 onnxruntime 跑 cv3 输出 ONNX，dump 整数域网格分数场。

E3 的第三实现（ORT QDQ 参考）数据源。与 ``dump_cv3.py``（板上 ACL）输出
格式完全一致，按 stem 配对后可喂 ``experiments.run_e3 --ort``。

输入模型是 ``amct_to_sim.py`` 产出的 fake-quant 仿真 ONNX（AMCT deploy 图含
Ascend 私有算子，ORT 跑不了；仿真图把 AscendQuant 改写为 Mul/Round/Add/Clip
链、Conv 权重按 ``W_f=W_q·deq·s_in`` 就地反量化、AscendDequant 改写为
Identity，浮点语义与板上整数链路等价，差仅在浮点累加舍入与 Round 模式）。

仿真图输出的是 requant 之前的连续值 ``v_real``；cv3 三个网格张量的 requant
在 numpy 侧完成：``q = round(v/Δ)``，``logit = q·Δ``，``prob = sigmoid(logit)``。
数值取证（probe_grid）：cv3 输出是 **int32 累加器网格**（q∈±万级，sigmoid
前无 int8 重量化），所以 q **不截断**、按 int32 落盘——与板上
``dump_cv3.py``（ACL 读出的张量即 ``q·Δ``）逐位对齐。Δ 的唯一正确来源是
板上 scale json 的无后缀激活条目
（``/model.23/cv3.{0,1,2}/cv3.{0,1,2}.2/Conv_quant_dequant``），经
``--head-deltas`` 传入（按空间尺寸降序 80/40/20）。

用法::

    python code/board/dump_cv3_ort.py \
        --model asc-sar/models/yolo11n_ssdd_int8_cv3_sim.onnx \
        --images asc-sar/datasets/ssdd/images/test \
        --out asc-sar/results/dumps/ort \
        --report asc-sar/results/m0/ort_grid.json \
        --head-deltas 6.938855e-4,1.323877e-3,5.517007e-4
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

try:
    from .runner import ensure_dir, json_dump, list_images, preprocess_image
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from runner import ensure_dir, json_dump, list_images, preprocess_image

HEAD_HW = ((80, 80), (40, 40), (20, 20))


def grid_outputs(sess, head_deltas: str) -> tuple[list[int], list[float]]:
    """按空间尺寸 80/40/20 识别网格输出索引；Δ 来自 --head-deltas（降序对应）。"""
    outs = sess.get_outputs()
    grid = []
    for i, o in enumerate(outs):
        shape = [d if isinstance(d, int) else -1 for d in o.shape]
        if len(shape) == 4 and shape[1] == 1 and shape[2] == shape[3]:
            grid.append((shape[2], i, o.name))
    grid.sort(reverse=True)
    assert [g[0] for g in grid] == [h for h, _ in HEAD_HW], \
        f"网格输出尺寸不符（期望 80/40/20）: {grid}"
    deltas = [float(s) for s in head_deltas.split(",")]
    assert len(deltas) == len(grid), \
        f"--head-deltas 需 {len(grid)} 个值（按 hw 降序），实得 {deltas}"
    return [i for _, i, _ in grid], deltas


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--contrast", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--head-deltas", required=True,
                    help="三个分类头的输出激活 Δ（scale json 无后缀激活条目），"
                         "逗号分隔、按空间尺寸降序 80,40,20，"
                         "如 6.938855e-4,1.323877e-3,5.517007e-4")
    args = ap.parse_args(argv)

    from runner import CODE
    sys.path.insert(0, str(CODE))
    from qcert.scores import ScoreField, save_score_npz

    import onnxruntime as ort

    sess = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    grid_idx, deltas = grid_outputs(sess, args.head_deltas)
    print(f"[ort] 网格输出索引 {grid_idx}，逐头 Δ = {deltas}")

    imgs = list_images(args.images, args.limit)
    out_dir = ensure_dir(args.out)

    stats = {h: {"res": [], "uniq": [], "qmin": [], "qmax": []} for h in (0, 1, 2)}
    t0 = time.time()
    for n, p in enumerate(imgs, 1):
        x, ratio, pad, (ih, iw) = preprocess_image(p, args.imgsz)
        arrs = sess.run(None, {sess.get_inputs()[0].name: x})
        logits, s_q_all = [], []
        for h, gi in enumerate(grid_idx):
            v = arrs[gi].astype(np.float64).ravel()
            r = v / deltas[h]
            # numpy 侧 requant：int32 累加器网格，不截断（sigmoid 前无 int8 重量化）
            q = np.rint(r).astype(np.int32)
            stats[h]["res"].append(float(np.abs(r - q).mean()))
            stats[h]["uniq"].append(int(np.unique(q).size))
            stats[h]["qmin"].append(int(q.min()))
            stats[h]["qmax"].append(int(q.max()))
            logits.append((q * deltas[h]).astype(np.float32))
            s_q_all.append(q)
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
                  "impl": "ort_int32_cv3", "head_deltas": deltas,
                  "contrast": float(args.contrast),
                  "model": Path(args.model).name,
                  "note": "ORT CPU 跑 fake-quant 仿真图（amct_to_sim 产物），"
                          "cv3 requant 在 numpy 侧完成（x86 参考实现）"},
            stem=p.stem)
        save_score_npz(out_dir / f"{p.stem}.npz", sf)
        if n % 50 == 0 or n == len(imgs):
            print(f"  [{n}/{len(imgs)}] {time.time() - t0:.0f}s")

    report = {"model": Path(args.model).name, "runtime": "onnxruntime-CPU",
              "n_images": len(imgs), "deltas": deltas, "heads": {},
              "note": "requant 在 numpy 侧完成，网格由构造保证；"
                      "requant_resid 是连续值到网格的平均/最大距离（期望 ~0.25/0.5）"}
    for h in (0, 1, 2):
        s = stats[h]
        n_el = HEAD_HW[h][0] * HEAD_HW[h][1]
        report["heads"][f"head{h}"] = {
            "delta": deltas[h],
            "requant_resid_mean": float(np.mean(s["res"])),
            "requant_resid_max": float(np.max(s["res"])),
            "uniq_frac": float(np.mean([u / n_el for u in s["uniq"]])),
            "q_min": int(np.min(s["qmin"])), "q_max": int(np.max(s["qmax"]))}
    report["verdict"] = {"grid_found": True, "quantized_in": "numpy",
                         "code_dtype": "int32",
                         "note": "cv3 是 int32 累加器网格（不触顶），"
                                 "uniq_frac 为格点占用（E2 输入）"}
    json_dump(report, args.report)
    print(f"[verdict] {report['verdict']}")
    for h in (0, 1, 2):
        r = report["heads"][f"head{h}"]
        print(f"[head{h}] Δ={r['delta']:.6e} resid={r['requant_resid_mean']:.3f} "
              f"uniq={r['uniq_frac']:.4f} q∈[{r['q_min']},{r['q_max']}]")
    print(f"[ort] dumps -> {out_dir}  report -> {args.report}")
    return report


if __name__ == "__main__":
    main()
