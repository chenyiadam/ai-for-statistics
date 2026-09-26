# 实践一复现说明（全栈训练 + 本地人机对话流程，重点验证项）

## 硬件与环境（指纹见 runs/env_fingerprint.txt）
- GPU：NVIDIA GeForce RTX 3060 12GB（驱动 576.88，CC 8.6）；CPU i7-10700；内存 32GB
- torch 2.3.0+cu121（CUDA 可用）+ transformers 4.53.2 + sentencepiece 0.2.2
- 语料：本书 21 章全文（2.34MB）→ `H:/aifs_ch21_data/corpus/book_full.txt`
- 分词器：`H:/aifs_ch21_data/tokenizer/tok8k.model`（SentencePiece BPE，8k 词表）
- 对话模型：Qwen2.5-0.5B-Instruct bf16 → `H:/aifs_ch21_data/models/qwen2.5-0.5b-instruct`
  （经 `HF_ENDPOINT=https://hf-mirror.com` 下载，约 22 秒）

## 两条训练线
1. **从零预训练**（scripts/train_tiny_lm.py）：6.85M 参数 GPT（6 层、256 维、8 头、权重绑定），
   bf16 自动混合精度，2500 步，约 207 秒跑完（约 6 TFLOPS 有效算力）。
2. **预训练模型对话**（scripts/chat_demo.py、serve_local.py）：下载 Qwen2.5-0.5B-Instruct，
   验证"加载 → chat template → GPU 推理 → 记录"的完整本地人机对话流程。

## 运行命令

```bash
PY=D:/Anaconda/envs/fed_clean_v2/python.exe
cd "I:/mydesk/ai-for-statistics/_experiments/ch21-应用实践/practice1-fullstack"

# 0) 环境指纹
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv

# 1) 训练分词器并体检（输出在 runs/tokenizer_train.log）
# 2) 从零预训练 + 生成样例（约 3.5 分钟）
$PY scripts/train_tiny_lm.py
# 3) loss 三段判读（章内 21.2.7 要求）
$PY scripts/judge_loss.py
# 4) 本地人机对话：脚本化三问（无人值守）或交互模式
$PY scripts/chat_demo.py --demo
$PY scripts/chat_demo.py
# 5) 服务化 + 审计日志（另开终端 POST /ask）
$PY scripts/serve_local.py
curl -s http://127.0.0.1:8000/ask -H "Content-Type: application/json" \
     -d "{\"question\": \"样本方差为什么除以 n-1\"}"
```

## 实测

| 环节 | 结果 | 执行文件 |
|---|---|---|
| 环境指纹 | RTX 3060 12GB / torch 2.3.0+cu121 / CUDA True | runs/env_fingerprint.txt |
| 分词器 | 8k 词表，3.816 bytes/token，unk 率 0，中文切分合理 | runs/tokenizer_train.log |
| 从零训练 | 6.85M 参数，2500 步，207 秒，loss 约 9.0→6.5，val 6.99 | runs/train_log.csv、tiny_lm_config.json、gen_sample.txt |
| loss 判读 | head 段损失均值 19.60（首步初始化异常）→ tail 6.54；检出 step 2225 单点 spike，脚本判定"需人工复核" | runs/loss_summary.json |
| 本地对话 | Qwen 0.5B bf16 在 3060 上 26.7–29.7 tok/s，中文问答正常，逐轮写入 JSONL | runs/dialog_log.jsonl、chat_demo_console.log |
| 服务化 | POST /ask 返回 HTTP 200，每次调用追加审计记录到 runs/api_log.jsonl | runs/api_log.jsonl、api_test_console.log |

## 模型输出质量评估（重要）

- 7M 从零模型在 0.57M token 上只能学到字符级统计规律，生成文本不成句——这符合预期：
  本实践验证的是"训练循环与日志判读"，不是模型质量。
- Qwen2.5-0.5B 的回答**不可直接采信**，实测出现两类典型错误：
  1. 把 wild bootstrap 附会为"Hutchinson's test（野鸡检验）"——无中生有的引用（幻觉）；
  2. 把 Breusch-Pagan 检验的原假设说成"无自相关"——与 Durbin-Watson 张冠李戴。
  这正呼应第 1 章"幻觉"与本章"评估"小节的告诫：本地模型回答必须经外部验证后才可使用。
- 数值题表现尚可：样本方差一题正确写出均值 2.8 与 n-1 公式（因 256 token 截断未算完）。

## 踩坑记录（本轮实际遇到）

1. transformers 5.x 要求 torch≥2.5，本机 torch 2.3 会被禁用（"PyTorch was not found"），
   需固定 `transformers==4.53.2`；
2. 4.53 的 `from_pretrained` 用 `torch_dtype` 而非 5.x 的 `dtype`；
3. 把位置编码写成普通属性而非 `register_buffer`，导致 CPU/GPU 设备不一致报错；
4. mcp 包与 transformers 无关但同理有 API 版本问题，见 practice3 README。
