#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Web Search Plugin v2 - Time-aware & High-volume
---------------------------------------------------------
A versatile web search plugin with local time awareness and enhanced token usage.

New Features v2:
- Local time integration for weather/news queries
- Increased search volume (default 10-15 results)
- Enhanced summarization using more tokens
- Smart query enhancement based on content type
- Regional awareness for weather searches

Features
- Providers: auto (bing -> serpapi -> ddg), or force via --provider
- Time-aware queries: automatically adds current date/time for relevant searches
- Enhanced summaries: fetches and summarizes more content with higher token usage
- Posts result to agent via Event schema: {"type": "web.search.result", "payload": {...}}
- FastAPI server mode (--serve) with /work endpoint

Environment variables
- BING_API_KEY          (for provider=bing or auto)
- SERPAPI_KEY           (for provider=serpapi or auto)
- AGENT_EVENT_URL       default http://127.0.0.1:8350/event
- WEB_SEARCH_REGION     default "ko-KR" for Korean region
- WEB_SEARCH_LOCATION   default "Busan, South Korea"

Examples
- One-shot CLI:   python web_search.py --query "오늘 날씨" --topk 10 --summarize --post
- Server mode:    python web_search.py --serve --port 8877
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
import locale

try:
    import requests
except Exception as e:
    print("ERROR: 'requests' package is required. pip install requests", file=sys.stderr)
    raise

# Enhanced configuration
AGENT_EVENT_URL = os.environ.get("AGENT_EVENT_URL", "http://127.0.0.1:8350/event")
AGENT_EVENT_KEY = os.environ.get("AGENT_EVENT_KEY")
WEB_SEARCH_REGION = os.environ.get("WEB_SEARCH_REGION", "ko-KR")
WEB_SEARCH_LOCATION = os.environ.get("WEB_SEARCH_LOCATION", "Busan, South Korea")
DEFAULT_TOPK = int(os.environ.get("WEB_SEARCH_DEFAULT_TOPK", "12"))  # 증가된 기본값

# Enhanced token usage for LM Studio integration
MAX_SUMMARY_CHARS = int(os.environ.get("WEB_SEARCH_MAX_CHARS", "8000"))  # 증가
MAX_SUMMARY_SENTENCES = int(os.environ.get("WEB_SEARCH_MAX_SENTENCES", "8"))  # 증가
MAX_CONTENT_FETCH = int(os.environ.get("WEB_SEARCH_MAX_FETCH", "5"))  # 더 많은 페이지 fetch

# ---------------- Logging ----------------
LOG_FORMAT = "%(asctime)s | %(levelname)-7s | web.search | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("web.search")

# ---------------- Time & Location Utils ----------------

def get_local_time_info() -> Dict[str, str]:
    """Get comprehensive local time information for search enhancement"""
    now = _dt.datetime.now()
    
    # Set Korean locale if available
    try:
        locale.setlocale(locale.LC_TIME, 'ko_KR.UTF-8')
    except:
        try:
            locale.setlocale(locale.LC_TIME, 'Korean_Korea.1252')
        except:
            pass  # Use system default
    
    return {
        "date": now.strftime("%Y-%m-%d"),
        "date_kr": now.strftime("%Y년 %m월 %d일"),
        "time": now.strftime("%H:%M"),
        "datetime": now.strftime("%Y-%m-%d %H:%M"),
        "weekday": now.strftime("%A"),
        "weekday_kr": ["월", "화", "수", "목", "금", "토", "일"][now.weekday()] + "요일",
        "month": now.strftime("%B"),
        "year": str(now.year),
        "hour": now.hour,
        "is_morning": 6 <= now.hour < 12,
        "is_afternoon": 12 <= now.hour < 18,
        "is_evening": 18 <= now.hour < 22,
        "is_night": now.hour >= 22 or now.hour < 6,
    }

def enhance_query_with_context(query: str) -> Tuple[str, Dict[str, Any]]:
    """Enhance search query with time and location context"""
    time_info = get_local_time_info()
    enhanced_query = query.strip()
    context = {"original_query": query, "enhancements": []}
    
    query_lower = query.lower()
    
    # Weather-related queries
    weather_keywords = ["날씨", "weather", "온도", "temperature", "비", "rain", "눈", "snow", "바람", "wind"]
    if any(kw in query_lower for kw in weather_keywords):
        if "오늘" not in query_lower and "today" not in query_lower:
            enhanced_query = f"{query} 오늘 {time_info['date_kr']}"
            context["enhancements"].append(f"Added today's date: {time_info['date_kr']}")
        
        if WEB_SEARCH_LOCATION not in query and "부산" not in query_lower and "busan" not in query_lower:
            enhanced_query = f"{enhanced_query} {WEB_SEARCH_LOCATION}"
            context["enhancements"].append(f"Added location: {WEB_SEARCH_LOCATION}")
    
    # News-related queries
    news_keywords = ["뉴스", "news", "최신", "latest", "소식", "today", "오늘"]
    if any(kw in query_lower for kw in news_keywords):
        if not any(date_kw in query_lower for date_kw in ["오늘", "today", "최신", "latest", time_info['date']]):
            enhanced_query = f"{query} {time_info['date_kr']} 최신"
            context["enhancements"].append(f"Added current date for news: {time_info['date_kr']}")
    
    # Stock/finance queries
    finance_keywords = ["주가", "stock", "주식", "증시", "kospi", "nasdaq", "환율", "exchange rate"]
    if any(kw in query_lower for kw in finance_keywords):
        if "실시간" not in query_lower and "real-time" not in query_lower:
            enhanced_query = f"{query} {time_info['date_kr']} 실시간"
            context["enhancements"].append(f"Added real-time context for finance: {time_info['date_kr']}")
    
    # Event/schedule queries  
    event_keywords = ["일정", "schedule", "이벤트", "event", "공연", "concert", "영화", "movie"]
    if any(kw in query_lower for kw in event_keywords):
        if not any(time_kw in query_lower for time_kw in ["오늘", "이번주", "today", "this week"]):
            enhanced_query = f"{query} {time_info['date_kr']} 이번주"
            context["enhancements"].append(f"Added time context for events: this week")
    
    context["enhanced_query"] = enhanced_query
    context["time_info"] = time_info
    
    return enhanced_query, context

# ---------------- Models ----------------

@dataclasses.dataclass
class SearchItem:
    title: str
    url: str
    snippet: str = ""
    source: str = ""
    position: int = 0
    enhanced_summary: str = ""  # 새로운 필드: 향상된 요약

@dataclasses.dataclass
class SearchResult:
    query: str
    provider: str
    items: List[SearchItem]
    enhanced_query: str = ""  # 향상된 검색어
    query_context: Dict[str, Any] = dataclasses.field(default_factory=dict)
    ts: str = dataclasses.field(default_factory=lambda: _dt.datetime.now().isoformat(timespec="seconds"))
    total_results: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "enhanced_query": self.enhanced_query,
            "provider": self.provider,
            "ts": self.ts,
            "total_results": self.total_results,
            "query_context": self.query_context,
            "items": [dataclasses.asdict(x) for x in self.items],
        }

# ---------------- Enhanced Utils ----------------

def _norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()

def _extract_text_enhanced(html_str: str, max_chars: int = MAX_SUMMARY_CHARS) -> str:
    """Enhanced text extraction with better content filtering"""
    if not html_str:
        return ""
    
    # Remove scripts, styles, and navigation elements
    text = re.sub(r"(?is)<(script|style|nav|header|footer|aside).*?>.*?</\1>", " ", html_str)
    text = re.sub(r"(?is)<(noscript|iframe|object|embed).*?>.*?</\1>", " ", text)
    
    # Remove comments and CDATA
    text = re.sub(r"(?s)<!--.*?-->", " ", text)
    text = re.sub(r"(?s)<!\[CDATA\[.*?\]\]>", " ", text)
    
    # Remove all HTML tags
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    
    # Decode HTML entities
    text = html.unescape(text or "")
    
    # Clean up whitespace and normalize
    text = re.sub(r"\s+", " ", text).strip()
    
    # Remove common footer/header text patterns
    text = re.sub(r"(?i)(copyright|©|\(c\)|all rights reserved|privacy policy|terms of service|subscribe|newsletter|follow us).*", "", text)
    
    return text[:max_chars]

def _enhanced_summarize(text: str, max_sentences: int = MAX_SUMMARY_SENTENCES) -> str:
    """Enhanced summarization with better sentence boundary detection"""
    if not text:
        return ""
    
    # Split by various sentence boundaries (Korean and English)
    sentence_boundaries = [
        r"(?<=[\.!?])\s+",  # English sentences
        r"(?<=[。！？])\s+",  # Korean punctuation (full-width)
        r"(?<=[ㄱ-ㅎㅏ-ㅣ가-힣])\.\s+",  # Korean sentences ending with period
        r"(?<=다)\.\s+",  # Common Korean sentence ending
        r"(?<=요)\.\s+",  # Polite Korean ending
    ]
    
    parts = [text]  # Start with full text
    for pattern in sentence_boundaries:
        new_parts = []
        for part in parts:
            new_parts.extend(re.split(pattern, part))
        parts = new_parts
    
    # Clean and filter sentences
    sentences = []
    for part in parts:
        cleaned = part.strip()
        if (cleaned and 
            len(cleaned) >= 10 and  # Minimum length
            len(cleaned) <= 500 and  # Maximum length per sentence
            not re.match(r'^[\s\W]*$', cleaned)):  # Not just punctuation/whitespace
            sentences.append(cleaned)
    
    if not sentences:
        return text[:800] if len(text) > 800 else text
    
    # Select best sentences (prefer longer, more informative ones)
    sentences.sort(key=len, reverse=True)
    selected = sentences[:max_sentences]
    
    # Sort back to original order if possible
    result = " ".join(selected)
    return result[:2000]  # Increased max length

def _http_get_enhanced(url: str, timeout: float = 8.0) -> Tuple[int, str]:
    """Enhanced HTTP get with better headers and error handling"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }
    
    try:
        # Add random delay to avoid rate limiting
        time.sleep(random.uniform(0.1, 0.3))
        
        r = requests.get(url, timeout=timeout, headers=headers, allow_redirects=True)
        return r.status_code, r.text
    except Exception as e:
        logger.debug(f"HTTP GET failed for {url}: {e}")
        return 0, f"ERR: {e}"

# ---------------- Search Providers ----------------

def search_bing_enhanced(query: str, topk: int = DEFAULT_TOPK) -> SearchResult:
    """Enhanced Bing search with more results and better parameters"""
    api_key = os.environ.get("BING_API_KEY") or os.environ.get("AZURE_BING_KEY")
    if not api_key:
        raise RuntimeError("BING_API_KEY not set")
    
    url = "https://api.bing.microsoft.com/v7.0/search"
    headers = {"Ocp-Apim-Subscription-Key": api_key}
    
    # Enhanced parameters
    params = {
        "q": query,
        "count": min(topk, 50),  # Bing max is 50
        "offset": 0,
        "mkt": WEB_SEARCH_REGION,  # Market region
        "safeSearch": "Moderate",
        "textDecorations": True,  # Enable for better formatting
        "textFormat": "HTML",
        "freshness": "Day" if any(kw in query.lower() for kw in ["뉴스", "news", "latest", "오늘"]) else None
    }
    
    # Remove None values
    params = {k: v for k, v in params.items() if v is not None}
    
    r = requests.get(url, headers=headers, params=params, timeout=8.0)
    r.raise_for_status()
    data = r.json()
    
    items: List[SearchItem] = []
    web_pages = (data.get("webPages", {}) or {}).get("value", [])
    
    for i, ent in enumerate(web_pages[:topk], start=1):
        items.append(SearchItem(
            title=_norm_space(ent.get("name", "")),
            url=ent.get("url", ""),
            snippet=_norm_space(ent.get("snippet", "")),
            source="bing",
            position=i
        ))
    
    total_results = (data.get("webPages", {}) or {}).get("totalEstimatedMatches", len(items))
    
    return SearchResult(
        query=query, 
        provider="bing", 
        items=items, 
        total_results=total_results
    )

def search_serpapi_enhanced(query: str, topk: int = DEFAULT_TOPK) -> SearchResult:
    """Enhanced SerpAPI search with more results"""
    api_key = os.environ.get("SERPAPI_KEY")
    if not api_key:
        raise RuntimeError("SERPAPI_KEY not set")
    
    url = "https://serpapi.com/search.json"
    params = {
        "q": query,
        "engine": "google",
        "num": min(topk, 100),  # Google allows up to 100
        "api_key": api_key,
        "hl": "ko",  # Korean language
        "gl": "kr",  # South Korea location
        "google_domain": "google.co.kr"
    }
    
    r = requests.get(url, params=params, timeout=8.0)
    r.raise_for_status()
    data = r.json()
    
    items: List[SearchItem] = []
    organic_results = data.get("organic_results", [])
    
    for i, ent in enumerate(organic_results[:topk], start=1):
        items.append(SearchItem(
            title=_norm_space(ent.get("title", "")),
            url=ent.get("link", ""),
            snippet=_norm_space(ent.get("snippet", "")),
            source="google",
            position=i
        ))
    
    # Try to get total results count
    search_info = data.get("search_information", {})
    total_results = search_info.get("total_results", len(items))
    if isinstance(total_results, str):
        # Parse "About 1,234,567 results"
        total_results = int(re.sub(r'[^\d]', '', total_results)) if total_results else len(items)
    
    return SearchResult(
        query=query, 
        provider="serpapi", 
        items=items,
        total_results=total_results
    )

def search_ddg_enhanced(query: str, topk: int = DEFAULT_TOPK) -> SearchResult:
    """Enhanced DuckDuckGo search"""
    items: List[SearchItem] = []
    
    try:
        try:
            from ddgs import DDGS
        except Exception:
            from duckduckgo_search import DDGS
        
        with DDGS(timeout=8.0) as ddgs:
            results = ddgs.text(
                query,
                region=WEB_SEARCH_REGION.lower().replace('-', '_'),
                safesearch="moderate",
                timelimit=None,  # No time limit unless news
                max_results=min(topk, 200)  # DDG can handle more
            )
            
            for i, r in enumerate(results, start=1):
                items.append(SearchItem(
                    title=_norm_space(r.get("title", "")),
                    url=r.get("href", ""),
                    snippet=_norm_space(r.get("body", "")),
                    source="ddg",
                    position=i
                ))
        
        return SearchResult(
            query=query, 
            provider="ddg", 
            items=items,
            total_results=len(items)
        )
        
    except Exception as e:
        logger.warning(f"[ddg] lib missing or failed ({e}); falling back to HTML")
        
        # HTML fallback with better parsing
        q = requests.utils.quote(query)
        url = f"https://duckduckgo.com/html/?q={q}&kl={WEB_SEARCH_REGION.lower()}"
        status, html_str = _http_get_enhanced(url, timeout=8.0)
        
        if status >= 400 or not html_str or html_str.startswith("ERR:"):
            raise RuntimeError(f"DuckDuckGo fallback failed: {status}")
        
        # Enhanced HTML parsing
        matches = re.findall(
            r'<a[^>]+class="result__a"[^>]*href="([^"]+)"[^>]*>\s*(.*?)\s*</a>',
            html_str, 
            flags=re.I | re.S
        )
        
        for i, (u, t) in enumerate(matches[:topk], start=1):
            title = _norm_space(re.sub(r"<.*?>", " ", t))
            if title and u:  # Only add if we have both title and URL
                items.append(SearchItem(
                    title=title, 
                    url=u, 
                    snippet="", 
                    source="ddg", 
                    position=i
                ))
        
        return SearchResult(
            query=query, 
            provider="ddg", 
            items=items,
            total_results=len(items)
        )

def run_search_enhanced(query: str, topk: int = DEFAULT_TOPK, provider: str = "auto") -> SearchResult:
    """Enhanced search with query context and better provider selection"""
    enhanced_query, context = enhance_query_with_context(query)
    
    provider = (provider or "auto").lower()
    result = None
    
    if provider == "bing":
        result = search_bing_enhanced(enhanced_query, topk)
    elif provider == "serpapi":
        result = search_serpapi_enhanced(enhanced_query, topk)
    elif provider == "ddg":
        result = search_ddg_enhanced(enhanced_query, topk)
    else:
        # Auto mode with smart provider selection
        errs = []
        
        # Prefer different providers based on query type
        query_lower = query.lower()
        if any(kw in query_lower for kw in ["뉴스", "news", "latest"]):
            providers = ["serpapi", "bing", "ddg"]  # Google/Bing better for news
        elif any(kw in query_lower for kw in ["날씨", "weather"]):
            providers = ["bing", "serpapi", "ddg"]  # Bing good for weather
        else:
            providers = ["serpapi", "bing", "ddg"]  # Default order
        
        for p in providers:
            try:
                result = run_search_enhanced(enhanced_query, topk, provider=p)
                break
            except Exception as e:
                errs.append(f"{p}: {e}")
                continue
        
        if not result:
            raise RuntimeError("All providers failed: " + "; ".join(errs))
    
    # Add context information
    result.enhanced_query = enhanced_query
    result.query_context = context
    
    return result

# ---------------- Enhanced Summarization ----------------

def enrich_with_enhanced_summaries(res: SearchResult, max_to_fetch: int = MAX_CONTENT_FETCH, timeout: float = 6.0) -> None:
    """Enhanced summarization with more content and better processing"""
    n = min(max_to_fetch, len(res.items))
    
    for i in range(n):
        item = res.items[i]
        url = item.url
        
        try:
            status, content = _http_get_enhanced(url, timeout=timeout)
            if status < 200 or status >= 400 or not content:
                continue
            
            # Extract and enhance text
            text = _extract_text_enhanced(content, max_chars=MAX_SUMMARY_CHARS)
            if not text or len(text) < 50:  # Skip if too short
                continue
            
            # Generate enhanced summary
            summary = _enhanced_summarize(text, max_sentences=MAX_SUMMARY_SENTENCES)
            
            if summary and len(summary) > len(item.snippet):
                # Replace or enhance existing snippet
                item.enhanced_summary = summary
                if not item.snippet or len(summary) > len(item.snippet) * 2:
                    item.snippet = summary[:500] + ("..." if len(summary) > 500 else "")
            
            logger.debug(f"Enhanced summary for {url}: {len(summary)} chars")
            
        except Exception as e:
            logger.debug(f"Failed to enhance {url}: {e}")
            continue

# ---------------- Event Posting ----------------

def post_event_enhanced(type_: str, payload: Dict[str, Any], url: Optional[str] = None) -> Tuple[int, str]:
    """Enhanced event posting with better payload structure"""
    url = url or AGENT_EVENT_URL
    _priority = 5
    
    try:
        if isinstance(payload, dict) and 'priority' in payload:
            _priority = int(payload.get('priority', 5))
    except Exception:
        _priority = 5
    
    # Enhanced event structure
    ev = {
        'type': type_,
        'payload': payload or {},
        'priority': _priority,
        'timestamp': _dt.datetime.utcnow().isoformat() + 'Z',
        'source': 'web_search_enhanced',
        'version': '2.0'
    }
    
    try:
        headers = {"Content-Type": "application/json"}
        if AGENT_EVENT_KEY:
            headers["X-Agent-Key"] = AGENT_EVENT_KEY
            
        r = requests.post(url, json=ev, headers=headers, timeout=5.0)
        logger.info(f"[post] {type_} -> {url} [{r.status_code}] ({len(payload.get('items', []))} items)")
        return r.status_code, r.text[:400]
    except Exception as e:
        logger.error(f"[post] Failed to post event: {e}")
        return 0, f"ERR: {e}"

# ---------------- CLI / Server ----------------

def run_cli_enhanced(args: argparse.Namespace) -> int:
    q = _norm_space(args.query)
    if not q:
        logger.error("Empty query provided")
        return 2
    
    logger.info(f"Enhanced search: '{q}' (topk={args.topk}, provider={args.provider})")
    
    try:
        res = run_search_enhanced(q, topk=max(1, args.topk), provider=args.provider)
        
        if args.summarize:
            logger.info(f"Enriching with enhanced summaries...")
            enrich_with_enhanced_summaries(res, max_to_fetch=min(MAX_CONTENT_FETCH, args.topk))
        
        out = res.to_dict()
        
        # Pretty print results
        print(json.dumps(out, ensure_ascii=False, indent=2))
        
        if args.post:
            status, resp = post_event_enhanced(
                "web.search.result", 
                {
                    "query": res.query,
                    "enhanced_query": res.enhanced_query,
                    "provider": res.provider,
                    "total_results": res.total_results,
                    "items": [dataclasses.asdict(item) for item in res.items],
                    "query_context": res.query_context
                }
            )
            logger.info(f"Posted to agent: {status} - {resp[:100]}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return 1

def run_server_enhanced(args: argparse.Namespace) -> int:
    try:
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse
        import uvicorn
    except Exception:
        logger.error("Server mode requires fastapi and uvicorn: pip install fastapi uvicorn")
        return 2

    app = FastAPI(title="Enhanced Web Search Plugin v2")

    @app.get("/health")
    async def health():
        return {
            "status": "healthy",
            "version": "2.0",
            "features": ["time_aware", "enhanced_summaries", "high_volume"],
            "default_topk": DEFAULT_TOPK,
            "max_summary_chars": MAX_SUMMARY_CHARS
        }

    @app.post("/work")
    async def work(payload: Dict[str, Any]):
        q = _norm_space(payload.get("q") or payload.get("query") or "")
        topk = max(1, int(payload.get("topk", args.topk)))
        provider = (payload.get("provider") or args.provider or "auto")
        do_summary = bool(payload.get("summarize", args.summarize))
        do_post = bool(payload.get("post", args.post))

        if not q:
            return JSONResponse({"error": "empty query"}, status_code=400)
        
        try:
            logger.info(f"Server search: '{q}' (topk={topk}, provider={provider}, summarize={do_summary})")
            
            res = run_search_enhanced(q, topk=topk, provider=provider)
            
            if do_summary:
                enrich_with_enhanced_summaries(res, max_to_fetch=min(MAX_CONTENT_FETCH, topk))
            
            out = res.to_dict()
            
            if do_post:
                post_event_enhanced(
                    "web.search.result", 
                    {
                        "query": res.query,
                        "enhanced_query": res.enhanced_query,
                        "provider": res.provider,
                        "total_results": res.total_results,
                        "items": [dataclasses.asdict(item) for item in res.items],
                        "query_context": res.query_context
                    }
                )
            
            return out
            
        except Exception as e:
            logger.error(f"Server search failed: {e}")
            return JSONResponse({"error": str(e)}, status_code=500)

    logger.info(f"Enhanced Web Search server running on http://127.0.0.1:{args.port}/work")
    uvicorn.run(app, host="127.0.0.1", port=int(args.port), log_level="info")
    return 0

def build_enhanced_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Enhanced Web Search Plugin v2 - Time-aware & High-volume")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--query", "-q", type=str, help="search query (one-shot CLI mode)")
    g.add_argument("--serve", action="store_true", help="run FastAPI server mode")
    
    p.add_argument("--topk", type=int, default=DEFAULT_TOPK, 
                   help=f"number of results (default: {DEFAULT_TOPK})")
    p.add_argument("--provider", type=str, default="auto", 
                   choices=["auto","bing","serpapi","ddg"], 
                   help="search provider (auto selects best for query type)")
    p.add_argument("--summarize", action="store_true", 
                   help="fetch and generate enhanced summaries of top results")
    p.add_argument("--post", action="store_true", 
                   help="post results to agent event bus")
    p.add_argument("--port", type=int, default=8877, 
                   help="port for server mode")
    
    # Enhanced options
    p.add_argument("--no-time-enhance", action="store_true",
                   help="disable automatic time-based query enhancement")
    p.add_argument("--max-fetch", type=int, default=MAX_CONTENT_FETCH,
                   help=f"max pages to fetch for summaries (default: {MAX_CONTENT_FETCH})")
    
    return p

def main(argv: Optional[List[str]] = None) -> int:
    args = build_enhanced_argparser().parse_args(argv)
    
    # Show startup info
    logger.info(f"Enhanced Web Search Plugin v2 starting...")
    logger.info(f"Default results: {DEFAULT_TOPK}, Max summary chars: {MAX_SUMMARY_CHARS}")
    logger.info(f"Region: {WEB_SEARCH_REGION}, Location: {WEB_SEARCH_LOCATION}")
    
    if args.serve:
        return run_server_enhanced(args)
    else:
        return run_cli_enhanced(args)

if __name__ == "__main__":
    sys.exit(main())