# tools/tool/web_fetch_lite.py
from __future__ import annotations
import urllib.parse, urllib.request, html, re

def fetch_lite(args):
    url = args["url"]
    allowed = set(args.get("allowed_domains") or [])
    max_bytes = int(args.get("max_bytes", 400_000))

    host = urllib.parse.urlparse(url).hostname or ""
    if allowed and not any(host.endswith(d) for d in allowed):
        raise RuntimeError(f"Domain not allowed: {host}")

    with urllib.request.urlopen(url, timeout=10) as resp:
        data = resp.read(max_bytes)
    text = data.decode("utf-8", errors="ignore")
    # 매우 단순 HTML→텍스트 정제
    text = re.sub(r"(?is)<script.*?</script>|<style.*?</style>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = html.unescape(re.sub(r"\s+", " ", text)).strip()
    return {"url": url, "text": text[:200000], "bytes": len(data)}

TOOL_HANDLERS = {"web.fetch_lite": fetch_lite}
