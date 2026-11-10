# -*- coding: utf-8 -*-
"""OM json 算子 dtype 统计: 找出哪些算子在 OM 里跑 fp16。

results/om_info/int8.json 是 aclmdl dumped 的 OM proto json。
op 条目形如 {"type": "Cast", "input": [{dtype: ..., shape: ...}], ...}。
统计 op type -> 输入 dtype Counter, 确认非量化算子的实际执行精度。
"""
import collections
import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "results/om_info/int8.json"
with open(path, encoding="utf-8") as f:
    data = json.load(f)

stats = collections.defaultdict(collections.Counter)  # type -> Counter(dtype)
seen = collections.Counter()


def walk(o):
    if isinstance(o, dict):
        t = o.get("type")
        if isinstance(t, str) and "input_desc" in o and isinstance(o["input_desc"], list):
            for item in o["input_desc"]:
                if isinstance(item, dict) and "dtype" in item:
                    stats[t][item["dtype"]] += 1
            seen[t] += 1
        for v in o.values():
            walk(v)
    elif isinstance(o, list):
        for v in o:
            walk(v)


walk(data)
print(f"{'op_type':<20} {'count':>5}  input dtype 分布")
for t, c in sorted(seen.items()):
    print(f"{t:<20} {c:>5}  {dict(stats[t])}")
