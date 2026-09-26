#!/usr/bin/env python3
"""把本地 Qwen 端点包装成带审计日志的科研服务（复现自第 21 章 21.2.13 节 FastAPI 块）。

用法：
    python serve_local.py            # 默认 127.0.0.1:8000
另开终端测试：
    curl -s http://127.0.0.1:8000/ask -H "Content-Type: application/json" \
         -d '{"question": "样本方差为什么除以 n-1"}'
每次 POST /ask 在 runs/api_log.jsonl 追加一条审计记录。
"""
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "H:/aifs_ch21_data/models/qwen2.5-0.5b-instruct"
HERE = Path(__file__).parent.parent
API_LOG = HERE / "runs" / "api_log.jsonl"

app = FastAPI(title="local-stats-llm")
_state = {}


class Ask(BaseModel):
    question: str
    max_new_tokens: int = 256


@app.on_event("startup")
def _load():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    _state["tok"], _state["model"] = tok, model


@app.post("/ask")
def ask(body: Ask):
    tok, model = _state["tok"], _state["model"]
    msgs = [{"role": "user", "content": body.question}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok([text], return_tensors="pt").to("cuda")
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=body.max_new_tokens,
                             do_sample=True, temperature=0.7, top_p=0.8)
    reply = tok.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
    sec = round(time.time() - t0, 2)
    rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "q": body.question, "a": reply, "gen_seconds": sec,
           "gen_tokens": int((out.shape[1] - inputs.input_ids.shape[1]))}
    with API_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return {"answer": reply, "gen_seconds": sec}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
