
# plugins/webtools.py
# Provides two plugin tools:
#  - web.search {q, k?:5, region?: "kr-kr", safesearch?: "moderate"}
#  - web.read {url, max_chars?: 6000}
def register(registry):
    try:
        from duckduckgo_search import DDGS
    except Exception as e:
        registry.emit("error", f"[webtools] duckduckgo_search not installed: {e}")
        return

    import httpx
    from bs4 import BeautifulSoup

    def search(payload):
        q = (payload or {}).get("q") or (payload or {}).get("query") or ""
        k = int((payload or {}).get("k") or (payload or {}).get("max_results") or 5)
        region = (payload or {}).get("region") or "kr-kr"
        safesearch = (payload or {}).get("safesearch") or "moderate"
        if not q:
            return {"ok": False, "error": "missing q"}
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(q, region=region, safesearch=safesearch, max_results=k):
                if not isinstance(r, dict): continue
                results.append({"title": r.get("title"), "href": r.get("href"), "body": r.get("body")})
        if results:
            tops = results[:3]
            lines = [f"SEARCH: {q}"] + [f" - {i+1}. {x.get('title') or ''} :: {x.get('href') or ''}" for i, x in enumerate(tops)]
            registry.emit("web", "\n".join(lines))
        return {"ok": True, "q": q, "results": results}

    def read(payload):
        url = (payload or {}).get("url") or (payload or {}).get("href")
        max_chars = int((payload or {}).get("max_chars") or 6000)
        if not url:
            return {"ok": False, "error": "missing url"}
        with httpx.Client(timeout=15.0, follow_redirects=True) as client:
            r = client.get(url)
        html = r.text or ""
        soup = BeautifulSoup(html, "lxml")
        title = (soup.title.string.strip() if soup.title and soup.title.string else "") or ""
        for tag in soup(["script","style","noscript"]): tag.extract()
        text = soup.get_text("\n", strip=True)
        text = "\n".join([line for line in text.splitlines() if line.strip()])
        if len(text) > max_chars:
            text = text[:max_chars] + " …"
        head = f"READ: {title or url}\n{url}"
        preview = (text[:500] + " …") if len(text) > 520 else text
        registry.emit("web", head + "\n" + preview)
        return {"ok": True, "url": url, "title": title, "text": text}

    registry.register_tool("web.search", search)
    registry.register_tool("web.read", read)
