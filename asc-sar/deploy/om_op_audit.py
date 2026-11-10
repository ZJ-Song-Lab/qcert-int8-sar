# -*- coding: utf-8 -*-
"""B 问题审计: 解析 atc --mode=1 dump 的 OM 结构 json, 检查量化算子链路。

背景:
  - modelslim 输出 ONNX: 88 Conv 全挂 AscendQuant(80 个 + 8 处复用),
    88 AscendDequant, scale 与 CPU 仿真逐位一致(已验证)。
  - OM dump 算子统计: Conv2D=88, AscendQuant=33, AscendDequant=0,
    Cast=97 —— ATC 做了图优化(DQ 融合进 Conv / Q-DQ 对折叠)。

判定:
  - Conv2D 输入(第一个 input_desc) dtype 是 int8 → 量化路径生效;
  - Conv2D 输入是 fp16 → 该层量化退化(回退 fp16 计算)。
  - Cast 的 src/dst 指示 ATC 为对齐 cube 算子做的类型转换。
"""
import json
import sys
from collections import Counter
from pathlib import Path

import onnx  # 仅用于读 ONNX 对照量化名单

DT = {"DT_FLOAT": "f32", "DT_FLOAT16": "f16", "DT_INT8": "i8",
      "DT_INT16": "i16", "DT_INT32": "i32", "DT_INT64": "i64",
      "DT_UINT8": "u8", "DT_UINT16": "u16", "DT_BOOL": "bool"}
# ge::DataType 实际编码(format_for_int 高 23 位)
GE_DT = {0: "f32", 1: "f16", 2: "i8", 3: "i16", 4: "u16", 5: "u8",
         6: "i32", 7: "i64", 8: "u32", 9: "u64", 10: "f64", 11: "bool"}


def get_dt_from_fi(desc):
    fi = get_format_int(desc)
    return GE_DT.get(fi, f"fi={fi}") if fi is not None else None


def get_dtype(desc):
    """从 input_desc/output_desc 提取实际 dtype(优先 format_for_int 编码)。"""
    if not desc:
        return "?"
    fi = get_dt_from_fi(desc)
    if fi:
        return fi
    for a in desc.get("attr", []):
        if a.get("key") in ("origin_data_type", "data_type"):
            v = a.get("value", {}).get("s", "")
            if v:
                return DT.get(v, v)
    return "?"


def get_format_int(desc):
    """format_for_int 高 23 位是 dtype(CANN ge::DataType), 低 23 位是 format。"""
    if not desc:
        return None
    for a in desc.get("attr", []):
        if a.get("key") == "format_for_int":
            v = a.get("value", {}).get("i")
            if isinstance(v, int):
                return v >> 23
    return None


def main(om_json, onnx_path):
    with open(om_json, encoding="utf-8") as f:
        d = json.load(f)
    ops = d["graph"][0]["op"]

    convs, casts, aqs = [], [], []
    for o in ops:
        t = o["type"]
        if t == "Conv2D":
            convs.append(o)
        elif t == "Cast":
            casts.append(o)
        elif t == "AscendQuant":
            aqs.append(o)

    # Conv2D: 输入 dtype / 输出 dtype
    conv_stat = Counter()
    conv_rows = []
    for o in convs:
        idesc = (o.get("input_desc") or [])
        odesc = (o.get("output_desc") or [None])[0]
        # 找 x 输入(第一个非 weight/bias): input_desc 顺序与算子 input 一致
        dts = []
        for k, desc in enumerate(idesc[:2]):  # 0=x, 1=weight
            dts.append((get_dtype(desc), get_format_int(desc)))
        out_dt = get_dtype(odesc)
        x_dt = dts[0][0] if dts else "?"
        # format_for_int 更可靠: desc attr 里字符串缺失时用编码
        x_fi = dts[0][1] if dts else None
        conv_stat[(x_dt, out_dt)] += 1
        conv_rows.append((o["name"], x_dt, dts[1][0] if len(dts) > 1 else "?",
                          out_dt, x_fi))

    print(f"=== Conv2D 共 {len(convs)} ===")
    for (x, y), c in conv_stat.most_common():
        print(f"  输入{x} -> 输出{y}: {c} 个")

    # AscendQuant: 输入/输出 dtype
    aq_stat = Counter()
    for o in aqs:
        i = (o.get("input_desc") or [None])[0]
        out = (o.get("output_desc") or [None])[0]
        aq_stat[(get_dtype(i), get_dtype(out))] += 1
    print(f"\n=== AscendQuant 共 {len(aqs)} ===")
    for (a, b), c in aq_stat.most_common():
        print(f"  输入{a} -> 输出{b}: {c} 个")

    # Cast: src -> dst
    cast_stat = Counter()
    cast_by_src = {}
    for o in casts:
        i = (o.get("input_desc") or [None])[0]
        out = (o.get("output_desc") or [None])[0]
        key = (get_dtype(i), get_dtype(out))
        cast_stat[key] += 1
        cast_by_src.setdefault(key, []).append(o["name"])
    print(f"\n=== Cast 共 {len(casts)} ===")
    for (a, b), c in cast_stat.most_common():
        print(f"  {a} -> {b}: {c} 个   例: {cast_by_src[(a, b)][:2]}")

    # 列出非 i8 输入的 Conv 名单(量化退化层)
    print("\n=== 非 i8 输入的 Conv 名单 ===")
    for name, x_dt, w_dt, out_dt, fi in conv_rows:
        if "i8" not in x_dt and fi != 2:
            print(f"    {name}: x={x_dt} w={w_dt} -> {out_dt}")
    print(f"\n判定: i8 输入 Conv {sum(1 for r in conv_rows if r[1]=='i8' or r[4]==2)}"
          f"/{len(conv_rows)}; 退化层如上 -> 定位 ATC 图优化破坏点")

    # ---- 与 ONNX 对照(可选): 哪些量化 Conv 在 OM 里还是 int8 输入 ----
    if not onnx_path or not Path(onnx_path).exists():
        print(f"[skip] ONNX 对照(文件不存在): {onnx_path}")
        return
    m = onnx.load(onnx_path)
    q_input_conv = set()  # Conv 名 -> 是否挂 AscendQuant
    aq_out = set()
    for n in m.graph.node:
        if n.op_type == "AscendQuant":
            aq_out.add(n.output[0])
    for n in m.graph.node:
        if n.op_type == "Conv":
            q_input_conv[n.name] = n.input[0] in aq_out

    # OM Conv 名字是 ONNX Conv 名(quant 前缀一致), 模糊匹配
    om_names = {r[0]: r for r in conv_rows}
    n_match = n_i8 = n_f16 = 0
    mismatch = []
    for name, quantized in q_input_conv.items():
        # OM 名可能带后缀, 用前缀匹配
        om = om_names.get(name) or next(
            (v for k, v in om_names.items() if k.startswith(name)), None)
        if om is None:
            continue
        n_match += 1
        x_dt = om[1]
        if "i8" in x_dt or om[4] == 2:  # ge::DT_INT8 == 2
            n_i8 += 1
        else:
            n_f16 += 1
            mismatch.append((name, x_dt, om[4], om[3]))
    print(f"\n=== ONNX 对照: 名字匹配 {n_match}/{len(q_input_conv)} ===")
    print(f"  int8 输入 Conv: {n_i8}")
    print(f"  非 int8 输入 Conv: {n_f16}")
    for name, x_dt, fi, out in mismatch[:20]:
        print(f"    {name}: x={x_dt}(fi_dtype={fi}) -> {out}")
    if n_f16 == 0:
        print("\n判定: 全部 Conv 保持 int8 输入 → 量化路径完整, B 问题在数值实现;")
    else:
        print(f"\n判定: {n_f16} 个 Conv 退化非 int8 → ATC 图优化破坏量化路径")


if __name__ == "__main__":
    om = sys.argv[1] if len(sys.argv) > 1 else "results/om_info/int8.json"
    ox = sys.argv[2] if len(sys.argv) > 2 else "models/yolo11n_int8.onnx"
    main(om, ox)
