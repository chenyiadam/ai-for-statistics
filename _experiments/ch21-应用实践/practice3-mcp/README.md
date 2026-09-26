# 实践三复现说明（Agent 与 MCP 工具集成本地冒烟）

## 目录
- `server/lit_server.py` — 文献检索 MCP 服务器（21.4.3 节；含 curl 回退补丁）
- `server/library/notes.jsonl` — save_note 产出的文献笔记
- `tests/smoke_client.py` — 协议层自动化冒烟（21.4.5 节方式二，扩展了 get_paper/to_bibtex/资源读取）
- `.mcp.json.example` — 客户端注册配置示例（21.4.4 节）
- `runs/` — 证据存档

## 实测（mcp 1.30.0，本机代理环境）

- 协议层：stdio 启动、initialize、list_tools 全通；工具名四件套齐全。
- 真实数据：`search_arxiv("heteroskedasticity robust standard errors", 3)` 返回 3 条真实 arXiv 条目；
  `get_paper` 精确取回首条；`to_bibtex` 生成 citekey `halkiewicz2025testing`；
  不存在编号 `0000.0000` 触发显式错误分支而非空结果；
  `save_note` 写入 notes.jsonl，资源 `notes://recent` 可回读。
- 证据：`runs/smoke_result.json`、`runs/smoke_console.log`。

## 复现中的两个关键坑（已修复并记录）

1. **mcp 2.x 破坏性变更**：`mcp` 包 2.x 将 `FastMCP` 更名为 `MCPServer`，第 21 章代码基于 v1 API，
   需固定 `pip install "mcp<2"`（本机实测 1.30.0 可用）。
2. **arXiv 按 TLS 指纹拦截 Python 客户端**：urllib/http.client 无论加什么请求头（含完整模拟 curl 头）
   一律收到 406；Windows 自带 curl（Schannel TLS 指纹）可正常访问。已在 `_http_get` 中加入
   urllib→curl 的自动回退。逐项排查过程：改 UA、改 Accept、https、绕过代理直连，均 406，
   排除请求头与代理因素后确认是 TLS 指纹层过滤。

## 与正文的差异

- 冒烟客户端的 `command` 用 `sys.executable` 替代裸 `python`，保证客户端与服务器用同一解释器
  （正文 21.4.4 的"常见错误"表格恰好覆盖这一点）。
- 21.4.6 的交互式 orchestrator（需人工确认输入）未纳入自动化冒烟，其结构已在正文给出；
  非交互环境下可把确认步骤换成 `--yes` 参数，风险自担并留下审计日志。
