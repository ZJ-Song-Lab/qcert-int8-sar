# -*- coding: utf-8 -*-
"""一次性探查: dump OM JSON 关键算子的完整 desc attrs, 反推 format_for_int 编码。"""
import json
import sys
from collections import Counter

path = sys.argv[1] if len(sys.argv) > 1 else "results/om_info/fp16.json"
pats = sys.argv[2:] or ["/model.10/m/m.0/attn/qkv/conv/Conv"]

d = json.load(open(path, encoding="utf-8"))
ops = d["graph"][0]["op"]
print(f"[{path}] total ops: {len(ops)}")


def attrs(desc):
    out = {}
    if not desc:
        return out
    for a in desc.get("attr", []):
        v = a.get("value") or {}
        val = "?"
        for k in ("i", "s", "f", "b"):
            if k in v:
                val = v[k]
                break
        if val == "?" and "list" in v:
            val = v["list"]
        out[a["key"]] = val
    return out


def fi(desc):
    return attrs(desc).get("format_for_int")


# 1) 全图 fi 直方图(input_desc[0] / output_desc[0])
c_in, c_out = Counter(), Counter()
for o in ops:
    i = o.get("input_desc") or []
    if i:
        c_in[fi(i[0])] += 1
    od = (o.get("output_desc") or [None])[0]
    if od:
        c_out[fi(od)] += 1
print("input_desc[0]  fi hist:", dict(sorted(c_in.items(), key=lambda kv: (kv[0] is None, kv[0]))))
print("output_desc[0] fi hist:", dict(sorted(c_out.items(), key=lambda kv: (kv[0] is None, kv[0]))))

# 2) 关键算子完整 desc(每个 pattern 最多展示 3 个命中)
for pat in pats:
    hits = [o for o in ops if pat in o["name"]][:3]
    print(f"\n### pattern={pat}  hits={len(hits)}")
    for o in hits:
        print(f"=== {o['type']}  {o['name']}  n_in={len(o.get('input_desc') or [])}")
        for tag, descs in (("in", o.get("input_desc") or []), ("out", o.get("output_desc") or [])):
            for j, dd in enumerate(descs):
                extra = {k: dd[k] for k in dd if k != "attr"}
                print(f"  {tag}[{j}] attrs={attrs(dd)}" + (f" other={extra}" if extra else ""))
