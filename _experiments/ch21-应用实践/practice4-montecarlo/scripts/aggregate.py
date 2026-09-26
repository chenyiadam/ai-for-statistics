# aggregate.py：把 runs/ 下的网格结果拼成一张表（含蒙特卡洛区间）
# 复现自第 21 章 21.5.5 节汇总块；在 practice4-montecarlo 目录下运行
import glob
import json

import pandas as pd

rows = []
for f in glob.glob("runs/n*_*.json"):
    d = json.loads(open(f, encoding="utf-8").read())
    c = d["config"]
    for m, v in d["reject"].items():
        rows.append({"n": c["n"], "design": c["design"], "lam": c["lam"],
                     "method": m, "rate": v["rate"], "mcse": v["mcse"],
                     "lo": v["lo"], "hi": v["hi"], "R": d["R"]})
df = pd.DataFrame(rows)
print(df.pivot_table(index=["design", "n", "lam"], columns="method", values="rate")
        .round(3).to_string())
df.to_csv("runs/grid_summary.csv", index=False)
print("已写出 runs/grid_summary.csv，共", len(df), "行")
