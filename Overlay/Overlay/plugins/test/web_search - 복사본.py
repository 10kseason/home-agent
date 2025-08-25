#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Web Search Plugin (Process or Microservice)
-------------------------------------------
A versatile web search plugin that can run as a one-shot CLI, a long-running microservice,
or a helper that posts results back to your agent event bus.

Features
- Providers: auto (bing -> serpapi -> ddg), or force via --provider
- Summaries: optional lightweight summarization by fetching top result pages (HTML -> text)
- Posts result to agent via Event schema: {"type": "web.search.result", "payload": {...}}
- FastAPI server mode (--serve) with /work endpoint: {"q": "...", "topk": 5, "summarize": true, "post": true}

Environment variables
- BING_API_KEY          (for provider=bing or auto)
- SERPAPI_KEY           (for provider=serpapi or auto)
- AGENT_EVENT_URL       default http://127.0.0.1:8765/event

Examples
- One-shot CLI:   python web_search.py --query "삼성 주가 전망" --topk 5 --summarize --post
- Server mode:    python web_search.py --serve --port 8877
                  curl -X POST http://127.0.0.1:8877/work -d '{"q":"chatgpt news","topk":5,"post":true}' -H "Content-Type: application/json"

Notes
- This script only depends on the standard library plus 'requests' (for HTTP). FastAPI/uvicorn are optional (only needed for --serve).
- If 'duckduckgo_search' library is available, it will be used for provider=ddg; otherwise a lightweight HTML fallback is used.
"""

import argparse
import asyncio
import contextlib
import dataclasses
import datetime as _dt
import html
import json
import logging
import os
import random
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

try:
    import requests
except Exception as e:
    print("ERROR: 'requests' package is required. pip install requests", file=sys.stderr)
    raise

AGENT_EVENT_URL = os.environ.get("AGENT_EVENT_URL", "http://127.0.0.1:8765/event")

# ---------------- Logging ----------------
LOG_FORMAT = "%(asctime)s | %(levelname)-7s | web.search | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("web.search")

# ---------------- Models ----------------

@dataclasses.dataclass
class SearchItem:
    title: str
    url: str
    snippet: str = ""
    source: str = ""
    position: int = 0

@dataclasses.dataclass
class SearchResult:
    query: str
    provider: str
    items: List[SearchItem]
    ts: str = dataclasses.field(default_factory=lambda: _dt.datetime.now().isoformat(timespec="seconds"))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "provider": self.provider,
            "ts": self.ts,
            "items": [dataclasses.asdict(x) for x in self.items],
        }

# ---------------- Utils ----------------

def _norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()

def _extract_text(html_str: str, max_chars: int = 2400) -> str:
    # SUPER lightweight text extraction; avoids heavy dependencies.
    # Remove scripts/styles, tags, condense whitespace.
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", html_str or "")
    text = re.sub(r"(?is)<noscript.*?>.*?</noscript>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = html.unescape(text or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]

def _summarize(text: str, max_sentences: int = 3) -> str:
    # A naive summarizer: pick the first N "sentences" (split by .!? plus Korean punctuation)
    # This is surprisingly robust for headlines/lede paragraphs.
    if not text:
        return ""
    # Split by Korean/English sentence boundaries
    parts = re.split(r"(?<=[\.!?])\s+|(?<=[。！？])\s+|(?<=[\u3002\uFF01\uFF1F])\s+", text)
    parts = [p.strip() for p in parts if p and len(p.strip()) > 0]
    if not parts:
        return text[:320]
    return " ".join(parts[:max_sentences])[:800]

def _http_get(url: str, timeout: float = 6.0, headers: Optional[Dict[str, str]] = None) -> Tuple[int, str]:
    try:
        r = requests.get(url, timeout=timeout, headers=headers or {})
        return r.status_code, r.text
    except Exception as e:
        return 0, f"ERR: {e}"

def post_event(type_: str, payload: Dict[str, Any], url: Optional[str] = None) -> Tuple[int, str]:
    url = url or AGENT_EVENT_URL
    ev = {
        "type": type_,
        "payload": payload or {},
        "timestamp": _dt.datetime.utcnow().isoformat() + "Z",
    }
    try:
        r = requests.post(url, json=ev, timeout=5.0)
        logger.info(f"[post] {type_} -> {url} [{r.status_code}]")
        return r.status_code, r.text[:400]
    except Exception as e:
        logger.warning(f"[post] failed: {e}")
        return 0, str(e)

# ---------------- Providers ----------------

def search_bing(query: str, topk: int = 5) -> SearchResult:
    api_key = os.environ.get("BING_API_KEY") or os.environ.get("AZURE_BING_KEY")
    if not api_key:
        raise RuntimeError("BING_API_KEY not set")
    url = "https://api.bing.microsoft.com/v7.0/search"
    headers = {"Ocp-Apim-Subscription-Key": api_key}
    params = {"q": query, "count": int(topk), "safeSearch": "Moderate", "textDecorations": False}
    r = requests.get(url, headers=headers, params=params, timeout=6.0)
    r.raise_for_status()
    data = r.json()
    items: List[SearchItem] = []
    for i, ent in enumerate((data.get("webPages", {}) or {}).get("value", [])[:topk], start=1):
        items.append(SearchItem(title=_norm_space(ent.get("name", "")),
                                url=ent.get("url", ""),
                                snippet=_norm_space(ent.get("snippet", "")),
                                source="bing",
                                position=i))
    return SearchResult(query=query, provider="bing", items=items)

def search_serpapi(query: str, topk: int = 5) -> SearchResult:
    api_key = os.environ.get("SERPAPI_KEY")
    if not api_key:
        raise RuntimeError("SERPAPI_KEY not set")
    url = "https://serpapi.com/search.json"
    params = {"q": query, "engine": "google", "num": int(topk), "api_key": api_key, "hl": "ko"}
    r = requests.get(url, params=params, timeout=6.0)
    r.raise_for_status()
    data = r.json()
    items: List[SearchItem] = []
    for i, ent in enumerate((data.get("organic_results") or [])[:topk], start=1):
        items.append(SearchItem(title=_norm_space(ent.get("title", "")),
                                url=ent.get("link", ""),
                                snippet=_norm_space(ent.get("snippet", "")),
                                source="google",
                                position=i))
    return SearchResult(query=query, provider="serpapi", items=items)

def search_ddg(query: str, topk: int = 5) -> SearchResult:
    # Prefer duckduckgo_search library if available
    items: List[SearchItem] = []
    try:
        from duckduckgo_search import DDGS  # type: ignore
        with DDGS(timeout=6.0) as ddgs:
            for i, r in enumerate(ddgs.text(query, max_results=int(topk)), start=1):
                items.append(SearchItem(title=_norm_space(r.get("title", "")),
                                        url=r.get("href", ""),
                                        snippet=_norm_space(r.get("body", "")),
                                        source="ddg",
                                        position=i))
        return SearchResult(query=query, provider="ddg", items=items)
    except Exception as e:
        logger.warning(f"[ddg] lib missing or failed ({e}); falling back to HTML")
        # VERY lightweight public HTML fallback; not guaranteed stable.
        q = requests.utils.quote(query)
        url = f"https://duckduckgo.com/html/?q={q}"
        status, html_str = _http_get(url, timeout=6.0, headers={"User-Agent": "Mozilla/5.0"})
        if status >= 400 or not html_str or html_str.startswith("ERR:"):
            raise RuntimeError(f"duckduckgo fallback failed: {status}")
        # naive scraping
        # each result snippet is in <a class="result__a"> and <a ...> parent with url in href; snippet in div result__snippet
        matches = re.findall(r'<a[^>]+class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', html_str, flags=re.I|re.S)
        for i, (u, t) in enumerate(matches[:topk], start=1):
            title = _norm_space(re.sub("<.*?>", " ", t))
            items.append(SearchItem(title=title, url=u, snippet="", source="ddg", position=i))
        return SearchResult(query=query, provider="ddg", items=items)

def run_search(query: str, topk: int = 5, provider: str = "auto") -> SearchResult:
    provider = (provider or "auto").lower()
    if provider == "bing":
        return search_bing(query, topk)
    if provider == "serpapi":
        return search_serpapi(query, topk)
    if provider == "ddg":
        return search_ddg(query, topk)

    # auto: prefer Bing -> SerpAPI -> DDG
    errs = []
    for p in ("bing", "serpapi", "ddg"):
        try:
            return run_search(query, topk, provider=p)
        except Exception as e:
            errs.append(f"{p}: {e}")
            continue
    raise RuntimeError("All providers failed: " + "; ".join(errs))

# ---------------- Fetch & Summarize ----------------

def enrich_with_summaries(res: SearchResult, max_to_fetch: int = 3, timeout: float = 5.0) -> None:
    n = min(max_to_fetch, len(res.items))
    for i in range(n):
        u = res.items[i].url
        try:
            r = requests.get(u, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code >= 400:
                continue
            text = _extract_text(r.text, max_chars=4000)
            if not text:
                continue
            summary = _summarize(text, max_sentences=3)
            if summary:
                res.items[i].snippet = (res.items[i].snippet or "") + ("  " if res.items[i].snippet else "") + summary
        except Exception as e:
            continue

# ---------------- CLI / Server ----------------

def run_cli(args: argparse.Namespace) -> int:
    q = _norm_space(args.query)
    if not q:
        logger.error("empty query")
        return 2
    logger.info(f"search: {q} (topk={args.topk}, provider={args.provider})")
    res = run_search(q, topk=args.topk, provider=args.provider)
    if args.summarize:
        enrich_with_summaries(res, max_to_fetch=min(3, args.topk))
    out = res.to_dict()
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if args.post:
        post_event("web.search.result", {"query": res.query, "provider": res.provider, "items": out["items"]})
    return 0

def run_server(args: argparse.Namespace) -> int:
    try:
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse
        import uvicorn
    except Exception:
        logger.error("Server mode requires fastapi and uvicorn: pip install fastapi uvicorn")
        return 2

    app = FastAPI(title="Web Search Plugin")

    @app.post("/work")
    async def work(payload: Dict[str, Any]):
        q = _norm_space(payload.get("q") or payload.get("query") or "")
        topk = int(payload.get("topk", args.topk))
        provider = (payload.get("provider") or args.provider or "auto")
        do_summary = bool(payload.get("summarize", args.summarize))
        do_post = bool(payload.get("post", args.post))

        if not q:
            return JSONResponse({"error": "empty query"}, status_code=400)
        try:
            res = run_search(q, topk=topk, provider=provider)
            if do_summary:
                enrich_with_summaries(res, max_to_fetch=min(3, topk))
            out = res.to_dict()
            if do_post:
                post_event("web.search.result", {"query": res.query, "provider": res.provider, "items": out["items"]})
            return out
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    logger.info(f"serving on http://127.0.0.1:{args.port}/work")
    uvicorn.run(app, host="127.0.0.1", port=int(args.port), log_level="info")
    return 0

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Web Search Plugin")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--query", "-q", type=str, help="search query (one-shot CLI mode)")
    g.add_argument("--serve", action="store_true", help="run FastAPI server mode")
    p.add_argument("--topk", type=int, default=5, help="number of results")
    p.add_argument("--provider", type=str, default="auto", choices=["auto","bing","serpapi","ddg"], help="search provider")
    p.add_argument("--summarize", action="store_true", help="fetch and inline brief summaries of top results")
    p.add_argument("--post", action="store_true", help="post results to agent event bus")
    p.add_argument("--port", type=int, default=8877, help="port for server mode")
    return p

def main(argv: Optional[List[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    if args.serve:
        return run_server(args)
    return run_cli(args)

if __name__ == "__main__":
    sys.exit(main())
