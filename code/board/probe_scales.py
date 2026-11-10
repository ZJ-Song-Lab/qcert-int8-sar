"""模块 2 / M0 硬门：把逐层 scale 读成规范 json，验证假设 **A3**。

A3 = "量化配置对使用者可读"。判据不是"能打开文件"，而是**输出层**
的 ``delta / zero_point`` 能被读出来，且单位与 ``qcert.qint`` 的仿射
约定一致（``a = (a_q - zp) * delta``）。读不出就没有整数域契约，
M0 不过，后面的 M4 训练成本一分都不要投（paper/05 §8）。

三条读取路径（可靠性递减，见 ``qcert.scale_table``）：

1. ``--input quantized.onnx`` —— QDQ 节点，**ATC 编译 INT8 OM 的实际输入**，
   论文里 "scale 可读" 应以这条为准；
2. ``--input amct_config.json`` —— AMCT 配置，常常只存策略不存 scale；
3. ``--input *.npz`` —— 板上 dump 时随分数场落盘的 delta。

示例::

    python -m board.probe_scales --input quantized.onnx --out results/m0/scale.json
    python -m board.probe_scales --om yolov8n_int8.om \\
        --report results/m0/om_probe.json
    python -m board.probe_scales --dry-run            # x86 自检
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:                                   # python -m board.probe_scales
    from .runner import RESULTS, ensure_dir, json_dump
except ImportError:                    # 直接 python board/probe_scales.py
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from runner import RESULTS, ensure_dir, json_dump  # noqa: F401


DEFAULT_CANDIDATES = ("output", "scores", "conv2d", "cls")


def pick_output(table, candidates=DEFAULT_CANDIDATES):
    """按名字猜输出层那一条 scale（找不到返回 None，由调用方判 FAIL）。"""
    for cand in candidates:
        for name in table.names():
            if cand in name.lower():
                return table.get(name)
    return None


def _dry_run(out_dir: Path) -> dict:
    """造一条 npz scale 再读回来，验证 from_npz + to_json 往返。"""
    import numpy as np

    from qcert.scale_table import ScaleTable

    tmp = ensure_dir(out_dir / "_dry")
    p = tmp / "dry_scale.npz"
    np.savez(p, delta=np.array(0.0078125, dtype=np.float32),
             zero_point=np.array(0, dtype=np.int32))
    table = ScaleTable.from_npz(p, name="output0")
    out = tmp / "dry_scale.json"
    table.to_json(out)
    back = ScaleTable.from_json(out)
    e = back.require("output0")
    return {"source": "dry-run", "n_entries": len(back.names()),
            "output_entry": e.to_dict(),
            "roundtrip_ok": bool(abs(e.delta - 0.0078125) < 1e-9)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=None, help="量化 onnx / amct json / npz")
    ap.add_argument("--kind", default="auto", choices=["auto", "onnx", "json", "npz"])
    ap.add_argument("--om", default=None, help="只做 .om 浅层探测（不解析私有 schema）")
    ap.add_argument("--out", default=None, help="规范 ScaleTable json 的输出路径")
    ap.add_argument("--report", default=None, help="A3 判定报告的 json 路径")
    ap.add_argument("--candidates", default=",".join(DEFAULT_CANDIDATES),
                    help="输出层名字的候选子串，逗号分隔")
    ap.add_argument("--dry-run", action="store_true", help="x86 自检，不读真实模型")
    args = ap.parse_args(argv)

    report = {"a3_readable": False}
    if args.dry_run:
        report = _dry_run(ensure_dir(RESULTS / "m0"))
        report["a3_readable"] = bool(report.pop("roundtrip_ok", False))
    elif args.om and not args.input:
        from qcert.scale_table import probe_om
        probe = probe_om(args.om)
        report = {"a3_readable": False, "om_probe": probe,
                  "note": "只做浅层探测；正式口径请用 --input 指向量化 ONNX"}
        print(f"[probe] {probe['path']} {probe['size']}B "
              f"keyword_hits={len(probe['keyword_hits'])}")
    else:
        if not args.input:
            ap.error("需要 --input（量化 onnx / amct json / npz）或 --om 或 --dry-run")
        from qcert.scale_table import load_any
        table = load_any(args.input, args.kind)
        cands = tuple(c.strip() for c in args.candidates.split(",") if c.strip())
        e = pick_output(table, cands)
        if args.out:
            table.to_json(args.out)
            print(f"[scale] 规范表 -> {args.out}")
        print(table.summary())
        report = {"a3_readable": bool(e is not None),
                  "source": table.meta.get("source"),
                  "input": str(args.input),
                  "n_entries": len(table.names()),
                  "output_entry": e.to_dict() if e is not None else None,
                  "candidates": list(cands),
                  "note": ("输出层 scale 可读，A3 成立" if e is not None
                           else "表里有 scale 但没匹配到输出层名，请用 --candidates 指定")}

    dst = args.report or (args.out.replace(".json", "_report.json")
                          if args.out else None)
    if dst:
        json_dump(report, dst)
        print(f"[probe] 报告 -> {dst}")
    print(f"[A3] {'PASS' if report.get('a3_readable') else 'FAIL'} "
          f"({report.get('source', report.get('note', ''))})")
    return report


if __name__ == "__main__":
    main()
