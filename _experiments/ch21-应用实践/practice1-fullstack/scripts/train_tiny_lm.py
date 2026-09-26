#!/usr/bin/env python3
"""从零预训练一个小型 GPT（约 7M 参数）在本章语料上，产出训练日志与检查点。

适配本地硬件：RTX 3060 12GB（bf16 自动混合精度）、i7-10700、32GB 内存。
语料：H:/aifs_ch21_data/corpus/book_full.txt（本书 21 章全文，2.34MB）
分词器：H:/aifs_ch21_data/tokenizer/tok8k.model（SentencePiece BPE，8k 词表）

用法：python train_tiny_lm.py
输出：runs/train_log.csv、runs/tiny_lm.pt、runs/tiny_lm_config.json、runs/gen_sample.txt
"""
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(20260101)

SPM = "H:/aifs_ch21_data/tokenizer/tok8k.model"
CORPUS = "H:/aifs_ch21_data/corpus/book_full.txt"
OUT = Path(__file__).parent.parent / "runs"

BLOCK = 256
BATCH = 48
STEPS = 2500
LR0, LR_MIN, WARMUP = 3e-4, 3e-5, 100
N_LAYER, N_EMBD, N_HEAD = 6, 256, 8
LOG_EVERY = 25


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(N_EMBD)
        self.attn = nn.MultiheadAttention(N_EMBD, N_HEAD, batch_first=True)
        self.ln2 = nn.LayerNorm(N_EMBD)
        self.mlp = nn.Sequential(nn.Linear(N_EMBD, 4 * N_EMBD), nn.GELU(),
                                 nn.Linear(4 * N_EMBD, N_EMBD))

    def forward(self, x, causal_mask):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=causal_mask, need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.wte = nn.Embedding(vocab, N_EMBD)
        self.wpe = nn.Embedding(BLOCK, N_EMBD)
        self.blocks = nn.ModuleList(Block() for _ in range(N_LAYER))
        self.lnf = nn.LayerNorm(N_EMBD)
        self.head = nn.Linear(N_EMBD, vocab, bias=False)
        self.head.weight = self.wte.weight                      # 权重绑定
        mask = torch.triu(torch.ones(BLOCK, BLOCK, dtype=torch.bool), 1)
        self.register_buffer("mask", mask)
        self.register_buffer("pos", torch.arange(BLOCK))

    def forward(self, idx):
        B, T = idx.shape
        x = self.wte(idx) + self.wpe(self.pos[:T])
        for blk in self.blocks:
            x = blk(x, self.mask[:T, :T])
        return self.head(self.lnf(x))


def get_batch(data, rng):
    ix = rng.integers(0, len(data) - BLOCK - 1, size=BATCH)
    x = torch.stack([torch.from_numpy(data[i:i + BLOCK].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + BLOCK].astype(np.int64)) for i in ix])
    return x.pin_memory().to("cuda", non_blocking=True), y.pin_memory().to("cuda", non_blocking=True)


@torch.no_grad()
def estimate_loss(model, data, iters=20):
    model.eval()
    rng = np.random.default_rng(7)
    losses = []
    for _ in range(iters):
        x, y = get_batch(data, rng)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = F.cross_entropy(model(x).view(-1, model.head.weight.shape[0]), y.view(-1))
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def main():
    sp = spm.SentencePieceProcessor(model_file=SPM)
    text = open(CORPUS, encoding="utf-8").read()
    ids = np.array(sp.encode(text), dtype=np.int64)
    n_val = int(0.05 * len(ids))
    train, val = ids[:-n_val], ids[-n_val:]
    vocab = sp.get_piece_size()
    n_params = sum(p.numel() for p in TinyGPT(vocab).parameters())
    print(f"tokens: train {len(train):,} / val {len(val):,}；vocab {vocab}；参数量 {n_params:,}")

    model = TinyGPT(vocab).to("cuda")
    opt = torch.optim.AdamW(model.parameters(), lr=LR0, weight_decay=0.1, betas=(0.9, 0.95))
    rng = np.random.default_rng(20260101)
    OUT.mkdir(exist_ok=True)
    log_path = OUT / "train_log.csv"
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["step", "loss", "lr", "gnorm", "tok_s", "elapsed_s", "split"])

    t0 = time.time()
    for step in range(1, STEPS + 1):
        lr = (LR0 * step / WARMUP) if step <= WARMUP else \
            LR_MIN + 0.5 * (LR0 - LR_MIN) * (1 + math.cos(math.pi * (step - WARMUP) / (STEPS - WARMUP)))
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = get_batch(train, rng)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = F.cross_entropy(model(x).view(-1, vocab), y.view(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        opt.step()
        if step % LOG_EVERY == 0 or step == 1:
            tok_s = BATCH * BLOCK * LOG_EVERY / max(time.time() - t0, 1e-9)
            row = [step, round(loss.item(), 4), f"{lr:.2e}", round(gnorm, 3),
                   int(tok_s), round(time.time() - t0, 1), "train"]
            csv.writer(open(log_path, "a", newline="", encoding="utf-8")).writerow(row)
            print("step", step, "loss", row[1], "lr", row[2], "gnorm", row[3], "tok/s", row[4])
        if step == STEPS // 2:
            v = estimate_loss(model, val)
            csv.writer(open(log_path, "a", newline="", encoding="utf-8")).writerow(
                [step, round(v, 4), "", "", "", round(time.time() - t0, 1), "val"])

    v_final = estimate_loss(model, val, 50)
    torch.save(model.state_dict(), OUT / "tiny_lm.pt")
    (OUT / "tiny_lm_config.json").write_text(json.dumps({
        "vocab": vocab, "block": BLOCK, "n_layer": N_LAYER, "n_embd": N_EMBD,
        "n_head": N_HEAD, "params": n_params, "steps": STEPS,
        "val_loss": round(v_final, 4), "train_tokens": len(train),
        "hardware": "RTX 3060 12GB / i7-10700 / bf16 autocast",
        "seed": 20260101}, ensure_ascii=False, indent=2), encoding="utf-8")

    # 生成样例：给定章节风格的开头续写 120 token
    model.eval()
    ctx = torch.tensor([sp.encode("回归诊断的第一步是检查误差的方差是否为常数")], device="cuda")
    for _ in range(120):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(ctx[:, -BLOCK:])
        nxt = torch.multinomial(F.softmax(logits[:, -1].float(), -1), 1)
        ctx = torch.cat([ctx, nxt], 1)
    sample = sp.decode(ctx[0].tolist())
    (OUT / "gen_sample.txt").write_text(sample, encoding="utf-8")
    print("val_loss:", round(v_final, 4))
    print("生成样例已写入 runs/gen_sample.txt")


if __name__ == "__main__":
    main()
