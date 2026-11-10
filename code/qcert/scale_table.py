"""逐层量化步长（scale）表读取器。

这是假设 A3（"量化配置对使用者可读"）的落地。AMCT / ATC 的产物形态随 CANN
版本变化，因此这里提供三条读取路径，按可靠性排序：

1. ``from_onnx_quant(path)``：读 *量化后的 ONNX*。支持两类节点：
   - ``AscendQuant`` / ``AscendDequant``：modelslim(CANN 自带) PTQ 的产物，
     **这是 ATC 编译 INT8 OM 的实际输入，因此最可靠**，也是论文中 "scale 可读"
     的正式依据。注意 AscendQuant 的 ``scale`` 属性存的是 *倒数* (1/scale)，
     AscendDequant 的 ``*_dequant_scale`` 参数是 UINT64，其低 32 位是
     ``w_scale * x_scale`` 的 float32 位模式；
   - ``QuantizeLinear`` / ``DequantizeLinear``：标准 QDQ(onnxruntime 量化产物)。
2. ``from_amct_json(path)``：读 AMCT ``*_config.json``（容错递归搜索）。
3. ``from_npz(path)``：读板上 dump 时随分数场一起落盘的 ``delta / zero_point``。

``probe_om(path)`` 只做 .om 的浅层探测（大小、可见字符串），**不假装能解析
私有 protobuf schema**；论文里应写成"scale 来自 ATC 输入侧的 QDQ 初始值"。
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

SCALE_KEYS = ("scale", "delta", "quant_scale", "step", "scale_value")
ZP_KEYS = ("zero_point", "zeropoint", "zp", "offset", "zero_point_value")


@dataclass
class ScaleEntry:
    name: str
    delta: float
    zero_point: int = 0
    bits: int = 8
    rmin: float | None = None
    rmax: float | None = None
    source: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ScaleTable:
    entries: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------ 基本操作
    def add(self, entry: ScaleEntry):
        self.entries[entry.name] = entry
        return self

    def get(self, name: str) -> ScaleEntry | None:
        return self.entries.get(name)

    def require(self, name: str) -> ScaleEntry:
        e = self.get(name)
        if e is None:
            raise KeyError(f"scale 表中没有 {name!r}；现有 {len(self.entries)} 项："
                           f"{list(self.entries)[:8]}{'...' if len(self.entries) > 8 else ''}")
        return e

    def names(self):
        return list(self.entries)

    def select(self, pattern: str) -> dict:
        rx = re.compile(pattern)
        return {k: v for k, v in self.entries.items() if rx.search(k)}

    def merge(self, other: "ScaleTable"):
        self.entries.update(other.entries)
        self.meta.update(other.meta)
        return self

    def sorted_entries(self):
        return sorted(self.entries.values(), key=lambda e: e.name)

    def summary(self) -> str:
        lines = [f"[scale] {len(self.entries)} 项  source={self.meta.get('source', '?')}"]
        for e in self.sorted_entries()[:10]:
            lines.append(f"  {e.name:<48s} delta={e.delta:.6g} zp={e.zero_point} "
                         f"bits={e.bits}")
        if len(self.entries) > 10:
            lines.append(f"  ... 其余 {len(self.entries) - 10} 项")
        return "\n".join(lines)

    # ------------------------------------------------------------ 序列化
    def to_json(self, path, indent: int = 2):
        payload = {
            "meta": self.meta,
            "entries": [e.to_dict() for e in self.sorted_entries()],
        }
        Path(path).write_text(json.dumps(payload, indent=indent, ensure_ascii=False),
                              encoding="utf-8")
        return path

    @classmethod
    def from_json(cls, path) -> "ScaleTable":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        rows = payload.get("entries", payload if isinstance(payload, list) else [])
        t = cls(meta=payload.get("meta", {}))
        for r in rows:
            t.add(ScaleEntry(**{k: r[k] for k in r if k in ScaleEntry.__annotations__}))
        return t

    # ------------------------------------------------------------ 数据源 1: 量化 ONNX
    @classmethod
    def from_onnx_quant(cls, path) -> "ScaleTable":
        """从量化 ONNX 读逐层 scale（AscendQuant 块或标准 QDQ 节点）。"""
        try:
            import onnx
            from onnx import numpy_helper
        except ImportError as exc:  # pragma: no cover - 取决于环境
            raise RuntimeError("读取量化 ONNX 需要 pip install onnx") from exc

        model = onnx.load(str(path))
        inits = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
        t = cls(meta={"source": "onnx_quant", "path": str(path)})
        seen = set()

        def _add(name, delta, zp=0, bits=8, source=""):
            delta = float(delta)
            if not np.isfinite(delta) or delta <= 0:
                return
            if name in seen:
                name = f"{name}#{len(seen)}"
            seen.add(name)
            t.add(ScaleEntry(name=name, delta=delta, zero_point=int(zp),
                             bits=int(bits), source=source))

        for node in model.graph.node:
            if node.op_type == "AscendQuant":
                # modelslim 产物 attrs: offset=INT(x_zp, 非对称时非 0),
                # quant_bit=INT, scale=FLOAT(1/x_scale 倒数)。
                # 注意 offset 必须按属性实际类型取值(.i): 读 .f 会得 0.0
                # 的静默假象, 导致 dump 报表里非对称 zp 全部丢成 0。
                attrs = {a.name: a for a in node.attribute}
                inv = attrs.get("scale")
                if inv is None:
                    continue

                def _val(a):
                    return onnx.helper.get_attribute_value(a)

                off = attrs.get("offset")
                qb = attrs.get("quant_bit")
                _add(node.name or f"AscendQuant[{len(seen)}]", 1.0 / _val(inv),
                     zp=round(float(_val(off))) if off is not None else 0,
                     bits=int(_val(qb)) if qb is not None else 8,
                     source="AscendQuant")
            elif node.op_type == "AscendDequant":
                # input[1] 是 UINT64 initializer，低 32 位是 w_scale*x_scale 的
                # float32 位模式(modelslim dag/graph.gen_ascend_dequant_onnx_node)
                if len(node.input) < 2 or node.input[1] not in inits:
                    continue
                arr = np.asarray(inits[node.input[1]]).ravel()
                base = node.name or node.input[1]
                for i in range(arr.size):
                    f32 = (np.array([arr[i]], dtype=np.uint64)
                           .astype(np.uint32).view(np.float32)[0])
                    _add(base if arr.size == 1 else f"{base}[{i}]",
                         f32, source="AscendDequant")
            elif node.op_type in ("QuantizeLinear", "DequantizeLinear"):
                if len(node.input) < 2:
                    continue
                sc = inits.get(node.input[1])
                if sc is None:
                    continue
                zp = inits.get(node.input[2]) if len(node.input) > 2 else None
                sarr = np.asarray(sc).ravel()
                zarr = np.asarray(zp).ravel() if zp is not None else np.zeros_like(sarr)
                for i in range(sarr.size):
                    base = node.input[1]
                    name = base[:-5] if base.endswith("scale") else f"{node.name or base}_{i}"
                    _add(name, sarr[i],
                         zp=zarr[i] if i < zarr.size else 0,
                         source=node.op_type)
        if not t.entries:
            raise ValueError(f"{path} 中未找到 AscendQuant/QDQ scale；确认这是量化后的 ONNX")
        return t

    # ------------------------------------------------------------ 数据源 2: AMCT json
    @classmethod
    def from_amct_json(cls, path) -> "ScaleTable":
        cfg = json.loads(Path(path).read_text(encoding="utf-8"))
        t = cls(meta={"source": "amct_config", "path": str(path)})

        def walk(node, prefix):
            if isinstance(node, dict):
                lower = {k.lower(): k for k in node}
                scale_key = next((lower[k] for k in SCALE_KEYS if k in lower), None)
                if scale_key is not None:
                    val = node[scale_key]
                    if isinstance(val, (int, float)) and float(val) > 0:
                        zk = next((lower[k] for k in ZP_KEYS if k in lower), None)
                        # rmin/rmax 可能显式为 null（导出的 scale 表常见）
                        rmin = node.get("rmin")
                        rmax = node.get("rmax")
                        # 节点自带 name 字段时优先用它（否则名字会是 entries[123]
                        # 这种结构路径，输出层候选名永远匹配不上）
                        node_name = node.get("name")
                        t.add(ScaleEntry(
                            name=(str(node_name) if isinstance(node_name, str)
                                  and node_name else (prefix or "<root>")),
                            delta=float(val),
                            zero_point=int(node[zk]) if zk is not None else 0,
                            rmin=float(rmin) if rmin is not None else None,
                            rmax=float(rmax) if rmax is not None else None,
                            source="amct_config",
                        ))
                for k, v in node.items():
                    walk(v, f"{prefix}.{k}" if prefix else str(k))
            elif isinstance(node, list):
                for i, v in enumerate(node):
                    walk(v, f"{prefix}[{i}]")

        walk(cfg, "")
        if not t.entries:
            raise ValueError(
                f"{path} 中没有找到 scale 字段。AMCT 的 *_config.json 往往只存"
                "量化策略而不存标定后的 scale；scale 实际落在量化 ONNX 里。"
                "请改用 --onnx 读取量化后的 onnx。")
        return t

    # ------------------------------------------------------------ 数据源 3: npz
    @classmethod
    def from_npz(cls, path, name: str = "output") -> "ScaleTable":
        z = np.load(str(path), allow_pickle=True)
        if "delta" not in z:
            raise KeyError(f"{path} 缺少 delta 字段；现有键 {list(z.keys())}")
        t = cls(meta={"source": "npz", "path": str(path)})
        t.add(ScaleEntry(name=name, delta=float(np.asarray(z["delta"]).ravel()[0]),
                         zero_point=int(np.asarray(z["zero_point"]).ravel()[0])
                         if "zero_point" in z else 0,
                         source="npz"))
        return t


def probe_om(path) -> dict:
    """.om 浅层探测：只报告可验证的事实，不臆造解析结果。"""
    p = Path(path)
    data = p.read_bytes()
    head = data[:64]
    ascii_runs = re.findall(rb"[ -~]{6,}", data[:4 * 1024 * 1024])
    hits = [s.decode("ascii", "ignore") for s in ascii_runs
            if any(k in s.lower() for k in (b"ascend", b"ge.", b"quant", b"scale"))]
    return {
        "path": str(p),
        "size": p.stat().st_size,
        "head_hex": head.hex(),
        "looks_like_onnx": head[:4] == b"\x08\x0c\x12\x08",
        "keyword_hits": hits[:20],
        "note": ".om 是私有 protobuf 容器；生产口径请从 ATC 输入侧的量化 ONNX "
                "读取逐层 scale（见 ScaleTable.from_onnx_quant）。",
    }


def load_any(path, kind: str = "auto") -> ScaleTable:
    """按扩展名/内容自动选择读取器。"""
    p = Path(path)
    kind = kind.lower()
    if kind == "auto":
        suffix = p.suffix.lower()
        kind = {".onnx": "onnx", ".json": "json", ".npz": "npz", ".om": "om"}.get(suffix, "json")
    if kind == "onnx":
        return ScaleTable.from_onnx_quant(p)
    if kind == "json":
        return ScaleTable.from_amct_json(p)
    if kind == "npz":
        return ScaleTable.from_npz(p)
    if kind == "om":
        info = probe_om(p)
        raise ValueError(f"无法直接从 .om 读 scale（{info['note']}）；"
                         f"请提供对应的量化 ONNX 或 AMCT 配置。探测结果：{info}")
    raise ValueError(f"未知 kind={kind}")


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description="QCert scale 表读取 / 打印")
    ap.add_argument("--input", required=True, help="量化 onnx / amct json / npz / om")
    ap.add_argument("--kind", default="auto", choices=["auto", "onnx", "json", "npz", "om"])
    ap.add_argument("--out", default=None, help="导出规范 json")
    ap.add_argument("--dry-run", action="store_true", help="只打印，不写文件")
    args = ap.parse_args(argv)

    t = load_any(args.input, args.kind)
    print(t.summary())
    if args.out and not args.dry_run:
        t.to_json(args.out)
        print(f"[scale] 已写出 {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
