#!/usr/bin/env python3
"""下载 Qwen2.5-0.5B-Instruct 到 H 盘（走 hf-mirror.com 镜像），用于本地人机对话流程。

用法：HF_ENDPOINT=https://hf-mirror.com python download_qwen.py
"""
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from huggingface_hub import snapshot_download

p = snapshot_download(
    "Qwen/Qwen2.5-0.5B-Instruct",
    local_dir="H:/aifs_ch21_data/models/qwen2.5-0.5b-instruct",
    allow_patterns=["*.json", "*.safetensors", "merges.txt", "vocab.json"],
    max_workers=4,
)
print("下载完成：", p)
