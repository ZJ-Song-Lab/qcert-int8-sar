# -*- coding: utf-8 -*-
"""提取 OM json 中指定 dtype 的算子名(如 fp16 Conv2D / MaxPool / BatchMatMul)。

op 条目: {"type": "Conv2D", "src_name": [...输入张量名...], "dst_name": [...]}
fp16 Conv 的 src_name 含其输入 conv 名, 用于定位未量化的 plain Conv。
"""
import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "results/om_info/int8.json"
want_dtype = sys.argv[2] if len(sys.argv) > 2 else "DT_FLOAT16"
want_type = sys.argv[3] if len(sys.argv) > 3 else "Conv2D"

with open(path, encoding="utf-8") as f:
    data = json.load(f)


def walk(o, hits):
    if isinstance(o, dict):
        t = o.get("type")
        if t == want_type and isinstance(o.get("input_desc"), list):
            if any(isinstance(d, dict) and d.get("dtype") == want_dtype
                   for d in o["input_desc"]):
                hits.append(o)
        for v in o.values():
            walk(v, hits)
    elif isinstance(o, list):
        for v in o:
            walk(v, hits)


hits = []
walk(data, hits)
print(f"{want_type} with {want_dtype}: {len(hits)}")
for h in hits:
    print(f"  id={h.get('id')} src={h.get('src_name')}")
