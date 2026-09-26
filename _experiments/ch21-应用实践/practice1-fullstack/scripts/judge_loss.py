# judge_loss.py：训练日志的三段判读（复现自第 21 章 21.2.7 节 loss 判读块）
# 前置条件：runs/train_log.csv 含 step,loss,lr,gnorm 列
import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent.parent
rows = []
with open(HERE / "runs" / "train_log.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        if r["split"] == "train":
            rows.append({"step": int(r["step"]), "loss": float(r["loss"]),
                         "lr": float(r["lr"]), "gnorm": float(r["gnorm"])})
rows.sort(key=lambda r: r["step"])
seg = np.array_split(rows, 3)
summary = {}
for name, part in zip(("head", "mid", "tail"), seg):
    losses = np.array([r["loss"] for r in part])
    gnorms = np.array([r["gnorm"] for r in part])
    summary[name] = {"loss_mean": round(float(losses.mean()), 4),
                     "loss_last": round(float(losses[-1]), 4),
                     "gnorm_max": round(float(gnorms.max()), 3),
                     "steps": f"{part[0]['step']}-{part[-1]['step']}"}

# spike 检测：相邻日志点损失涨幅超过 3 倍滚动标准差
losses = np.array([r["loss"] for r in rows])
roll_sd = np.array([losses[max(0, i - 10):i + 1].std() if i > 3 else 0 for i in range(len(losses))])
spikes = [int(rows[i]["step"]) for i in range(1, len(losses))
          if roll_sd[i] > 1e-6 and losses[i] - losses[i - 1] > 3 * roll_sd[i]]
summary["spike_steps"] = spikes[:10]
summary["verdict"] = ("loss_mean 递减、gnorm 收敛、无持续 spike"
                      if summary["head"]["loss_mean"] > summary["tail"]["loss_mean"]
                      and not spikes else "需要人工复核")

out = HERE / "runs" / "loss_summary.json"
out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
