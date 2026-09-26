# 第 21 章实证复现包（本地证据存档）

本文件夹是《AI 赋能统计研究》第 21 章四个实践的本地复现与证据存档，建立于 2026-09-25。
所有脚本基于 conda 环境 `fed_clean_v2`（Python 3.10，`D:/Anaconda/envs/fed_clean_v2/python.exe`）与本机硬件（NVIDIA RTX 3060 12GB、i7-10700、32GB 内存）适配。

## 目录结构

- `practice1-fullstack/` — 实践一：全栈训练与本地人机对话流程（重点验证项）
  - `scripts/` 训练与推理脚本；`data/` 语料与分词器；`runs/` 训练日志与产出；`logs/` 对话与服务日志
- `practice2-skill/` — 实践二：科研 Skill（回归诊断）开发与测试
- `practice3-mcp/` — 实践三：Agent 与 MCP 工具集成本地冒烟
- `practice4-montecarlo/` — 实践四：异方差检验水平失真的蒙特卡洛闭环
- `ai-derivations/` — 本章数学/统计证明的 AI 辅助推导过程存档（提示词与完整回复）

## 大文件存放位置

模型权重与语料等大文件统一放在 `H:/aifs_ch21_data/`（三块候选盘中 H 盘剩余空间最大，约 69G；F 盘约 30G、G 盘约 56G，均不足以容纳解压后的模型与中间产物冗余）。项目盘 I 仅保留脚本与文本日志。

## 环境

- 解释器：`D:/Anaconda/envs/fed_clean_v2/python.exe`
- 关键依赖（2026-09-25 装机实测）：torch 2.3.0+cu121（CUDA 可用，RTX 3060）、numpy 1.26.4、scipy 1.15.3、statsmodels 0.15.0、pandas 2.3.3、transformers（本轮经代理安装）、mcp、fastapi、uvicorn、sentencepiece 0.2.2
- 网络出口：本机代理 `http://127.0.0.1:62287`；PyPI 走清华镜像；模型下载走 `HF_ENDPOINT=https://hf-mirror.com`
- 环境指纹存档：`practice1-fullstack/runs/env_fingerprint.txt`

## 运行方式

每个实践子目录内的 README 给出该实践的逐条运行命令与预期输出；全部命令均可直接复制执行。日志、指标与对话记录保存在各自的 `runs/` 或 `logs/` 下，作为可复现证据。

## 与章节的对应

第 21 章正文中的代码块在本复现包中以可独立运行的脚本形式组织；正文新增的"复现存档"小节（21.9 前）给出每个实践的证据文件清单。
