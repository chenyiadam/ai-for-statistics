#!/usr/bin/env python3
"""文献检索 MCP 服务器（stdio 传输）。

复现自《AI 赋能统计研究》第 21 章 21.4.3 节。
依赖：mcp<2（v1 API；mcp 2.x 已把 FastMCP 更名为 MCPServer）
运行：python lit_server.py            # 由客户端以子进程方式启动
"""
import json
import subprocess
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

from mcp.server.fastmcp import FastMCP

ROOT = Path(__file__).parent / "library"
ROOT.mkdir(exist_ok=True)
NOTES = ROOT / "notes.jsonl"
mcp = FastMCP("lit-server")

ATOM = "{http://www.w3.org/2005/Atom}"


def _http_get(url: str, timeout: int = 25) -> bytes:
    """先走 urllib；若被 arXiv 按 TLS 指纹拦截（406），回退到系统 curl。

    本机实测（2026-09-25）：arXiv 对 Python/OpenSSL 握手返回 406，与请求头无关；
    Windows 自带 curl（Schannel TLS）可正常访问，故以 curl 作为回退通道。
    """
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.read()
    except Exception as exc:
        if "406" not in str(exc) and "403" not in str(exc):
            raise
        cp = subprocess.run(["curl", "-sL", "--max-time", str(timeout), url],
                            capture_output=True, timeout=timeout + 10)
        if cp.returncode != 0 or not cp.stdout:
            raise RuntimeError(f"urllib 与 curl 均失败：{exc} / {cp.stderr[:200]}") from exc
        return cp.stdout


def _query_arxiv(query: str, max_results: int = 10) -> list[dict]:
    url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode({
        "search_query": f'all:"{query}"',
        "start": 0, "max_results": max_results,
        "sortBy": "submittedDate", "sortOrder": "descending"})
    xml = _http_get(url)
    root = ET.fromstring(xml)
    out = []
    for e in root.findall(f"{ATOM}entry"):
        aid = e.findtext(f"{ATOM}id", "").strip()
        out.append({
            "arxiv_id": aid.rsplit("/abs/", 1)[-1],
            "title": " ".join(e.findtext(f"{ATOM}title", "").split()),
            "authors": [a.findtext(f"{ATOM}name", "") for a in e.findall(f"{ATOM}author")],
            "published": e.findtext(f"{ATOM}published", "")[:10],
            "summary": " ".join(e.findtext(f"{ATOM}summary", "").split()),
            "url": aid,
        })
    return out


@mcp.tool()
def search_arxiv(query: str, max_results: int = 10) -> str:
    """检索 arXiv 最新论文，返回结构化元数据列表。"""
    if max_results > 50:
        max_results = 50
    papers = _query_arxiv(query, max_results)
    return json.dumps({"count": len(papers), "papers": papers}, ensure_ascii=False)


@mcp.tool()
def get_paper(arxiv_id: str) -> str:
    """按 arXiv 编号精确获取单篇论文的元数据；编号不存在时返回空结果。"""
    url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode(
        {"id_list": arxiv_id, "max_results": 1})
    try:
        root = ET.fromstring(_http_get(url))
    except Exception as exc:                       # 网络失败要显式报告，不许静默
        return json.dumps({"error": f"arXiv 查询失败: {exc}"}, ensure_ascii=False)
    entries = root.findall(f"{ATOM}entry")
    if not entries:
        return json.dumps({"found": False, "arxiv_id": arxiv_id}, ensure_ascii=False)
    e = entries[0]
    return json.dumps({
        "found": True,
        "arxiv_id": arxiv_id,
        "title": " ".join(e.findtext(f"{ATOM}title", "").split()),
        "authors": [a.findtext(f"{ATOM}name", "") for a in e.findall(f"{ATOM}author")],
        "published": e.findtext(f"{ATOM}published", "")[:10],
        "summary": " ".join(e.findtext(f"{ATOM}summary", "").split()),
        "url": "https://arxiv.org/abs/" + arxiv_id,
    }, ensure_ascii=False)


@mcp.tool()
def to_bibtex(arxiv_id: str, citekey: str = "") -> str:
    """把 arXiv 元数据转成 BibTeX 条目；元数据取自 arXiv，不凭记忆生成。"""
    meta = json.loads(get_paper(arxiv_id))
    if not meta.get("found"):
        return json.dumps({"error": f"未找到 {arxiv_id}，无法生成 BibTeX"}, ensure_ascii=False)
    key = citekey or meta["authors"][0].split()[-1].lower() + meta["published"][:4] + \
          meta["title"].split()[0].lower()
    authors = " and ".join(meta["authors"])
    return json.dumps({"citekey": key, "bibtex": (
        f"@misc{{{key},\n"
        f"  title  = {{{meta['title']}}},\n"
        f"  author = {{{authors}}},\n"
        f"  year   = {{{meta['published'][:4]}}},\n"
        f"  eprint = {{{arxiv_id}}},\n"
        f"  url    = {{{meta['url']}}}\n"
        "}")}, ensure_ascii=False)


@mcp.tool()
def save_note(title: str, summary: str, arxiv_id: str = "") -> str:
    """把一条文献笔记追加到本地 library/notes.jsonl（写操作，需在客户端侧确认）。"""
    rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "title": title, "summary": summary, "arxiv_id": arxiv_id}
    with NOTES.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return json.dumps({"saved": str(NOTES), "title": title}, ensure_ascii=False)


@mcp.resource("notes://recent")
def recent_notes() -> str:
    """返回最近 20 条文献笔记，供 Agent 作为上下文读取。"""
    if not NOTES.exists():
        return "[]"
    lines = NOTES.read_text(encoding="utf-8").strip().split("\n")[-20:]
    return "[" + ",".join(lines) + "]"


if __name__ == "__main__":
    mcp.run(transport="stdio")
