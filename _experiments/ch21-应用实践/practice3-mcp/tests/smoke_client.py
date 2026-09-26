# tests/smoke_client.py：不经由模型，直接验证协议层是否通
# 复现自第 21 章 21.4.5 节；command 改用 sys.executable 以保证用同一解释器
# 前置条件：pip install "mcp<2"，且网络可访问 arXiv API（本机经代理可达）
import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER = Path(__file__).parent.parent / "server" / "lit_server.py"
RUNS = Path(__file__).parent.parent / "runs"
RUNS.mkdir(exist_ok=True)


async def main() -> None:
    params = StdioServerParameters(command=sys.executable, args=[str(SERVER)])
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            tools = await s.list_tools()
            names = [t.name for t in tools.tools]
            print("tools:", names)
            assert {"search_arxiv", "get_paper", "to_bibtex", "save_note"} <= set(names)

            res = await s.call_tool("search_arxiv",
                                    {"query": "heteroskedasticity robust standard errors",
                                     "max_results": 3})
            payload = res.content[0].text
            print("search ok, first 200 chars:", payload[:200])

            # 单篇精确获取：取一个真实存在的编号（上一条检索的第一条结果）
            first_id = json.loads(payload)["papers"][0]["arxiv_id"]
            res_g = await s.call_tool("get_paper", {"arxiv_id": first_id})
            g = json.loads(res_g.content[0].text)
            print("get_paper:", g["found"], g["title"][:80])

            res_b = await s.call_tool("to_bibtex", {"arxiv_id": first_id})
            b = json.loads(res_b.content[0].text)
            print("bibtex citekey:", b["citekey"])

            res2 = await s.call_tool("to_bibtex", {"arxiv_id": "0000.0000"})
            print("nonexistent id ->", res2.content[0].text[:120])

            # 写操作：save_note + 资源读取回读
            res_n = await s.call_tool("save_note", {"title": "冒烟测试笔记",
                                                    "summary": "practice3 smoke test",
                                                    "arxiv_id": first_id})
            print("save_note ->", res_n.content[0].text[:120])
            notes = await s.read_resource("notes://recent")
            print("resource notes://recent len:", len(notes.contents[0].text))

            # 证据存档
            log = {"tools": names,
                   "search_first200": payload[:200],
                   "get_paper": {"found": g["found"], "title": g["title"]},
                   "bibtex_citekey": b["citekey"],
                   "nonexistent_branch": res2.content[0].text[:120]}
            (RUNS / "smoke_result.json").write_text(
                json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
            print("smoke_result.json 已保存")


if __name__ == "__main__":
    asyncio.run(main())
