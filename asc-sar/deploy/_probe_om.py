# -*- coding: utf-8 -*-
"""一次性探查: OM dump 里 model.10(C2PSA 注意力块)的算子序列与 dtype 边界。"""
import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "results/om_info/int8.json"
pat = sys.argv[2] if len(sys.argv) > 2 else "/model.10/"

d = json.load(open(path, encoding="utf-8"))
ops = d["graph"][0]["op"]
GE_DT = {0: "f32", 1: "f16", 2: "i8", 3: "i16", 4: "u16", 5: "u8",
         6: "i32", 7: "i64", 8: "u32", 9: "u64", 10: "f64", 11: "bool"}


def dt(desc):
    if not desc:
        return "?"
    for a in desc.get("attr", []):
        if a.get("key") == "format_for_int":
            v = a.get("value", {}).get("i")
            if isinstance(v, int):
                return GE_DT.get(v >> 23, str(v >> 23))
    for a in desc.get("attr", []):
        if a.get("key") == "origin_data_type":
            return a.get("value", {}).get("s", "?")
    return "?"


sel = [o for o in ops if pat in o["name"]]
print(f"[{pat}] 命中 {len(sel)} 个算子")
for o in sel:
    i = o.get("input_desc") or []
    od = (o.get("output_desc") or [None])[0]
    n_in = len(i)
    ins = ",".join(dt(x) for x in i[:4])
    print(f"{o['type']:<14} {o['name']:<64} n_in={n_in:<3} in[{ins}] out[{dt(od)}]")
