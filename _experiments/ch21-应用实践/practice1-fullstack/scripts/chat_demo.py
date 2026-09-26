#!/usr/bin/env python3
"""本地人机对话流程验证（第 21 章 21.2.9/21.2.13 节的落地）。

用 Qwen2.5-0.5B-Instruct（H:/aifs_ch21_data/models/qwen2.5-0.5b-instruct，RTX 3060 bf16）
跑一轮固定问题的脚本化对话，逐轮写入 runs/dialog_log.jsonl，验证：
加载 -> chat template -> GPU 推理 -> 响应记录 的完整链路。

两种模式：
    python chat_demo.py --demo          # 脚本化三问（无人值守，输出留档）
    python chat_demo.py                 # 交互式（stdin 逐行提问，exit 退出）
"""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "H:/aifs_ch21_data/models/qwen2.5-0.5b-instruct"
HERE = Path(__file__).parent.parent
LOG = HERE / "runs" / "dialog_log.jsonl"

DEMO_QUESTIONS = [
    "用一句话说明什么是异方差，以及它对普通最小二乘估计的标准误有什么影响。",
    "给定样本 3, 1, 4, 1, 5，求样本方差（分母用 n-1）。",
    "什么是 wild bootstrap，它在什么场合比 HC 稳健标准误更可取？",
]


def load():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    return tok, model


@torch.no_grad()
def ask(tok, model, history: list[dict], max_new_tokens: int = 256) -> tuple[str, float]:
    text = tok.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
    inputs = tok([text], return_tensors="pt").to("cuda")
    t0 = time.time()
    out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                         do_sample=True, temperature=0.7, top_p=0.8)
    reply = tok.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
    return reply.strip(), round(time.time() - t0, 2)


def log_round(role_q: str, reply: str, sec: float, tokens: int):
    rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "q": role_q, "a": reply, "gen_seconds": sec,
           "gen_tokens": tokens, "tok_per_s": round(tokens / max(sec, 1e-9), 1),
           "device": "cuda", "model": "Qwen2.5-0.5B-Instruct bf16"}
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="脚本化三问，无人值守")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()

    tok, model = load()
    print(f"模型已加载到 {next(model.parameters()).device}，"
          f"参数量 {sum(p.numel() for p in model.parameters()):,}")
    history = []

    questions = DEMO_QUESTIONS if args.demo else iter(lambda: input("用户> ").strip()), None
    if args.demo:
        for q in DEMO_QUESTIONS:
            history.append({"role": "user", "content": q})
            reply, sec = ask(tok, model, history, args.max_new_tokens)
            n_tok = len(tok.encode(reply))
            rec = log_round(q, reply, sec, n_tok)
            print(f"\n用户> {q}\n模型> {rec['a']}\n（{sec}s，{n_tok} tokens，"
                  f"{rec['tok_per_s']} tok/s）")
            history.append({"role": "assistant", "content": rec["a"]})
    else:
        print("交互模式：逐行输入问题，exit 退出。")
        while True:
            try:
                q = input("用户> ").strip()
            except EOFError:
                break
            if not q or q.lower() == "exit":
                break
            history.append({"role": "user", "content": q})
            reply, sec = ask(tok, model, history, args.max_new_tokens)
            n_tok = len(tok.encode(reply))
            rec = log_round(q, reply, sec, n_tok)
            print(f"模型> {rec['a']}（{sec}s）")
            history.append({"role": "assistant", "content": rec["a"]})
    print("对话日志已追加到", LOG)


if __name__ == "__main__":
    main()
