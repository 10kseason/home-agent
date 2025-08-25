"""Plugin to open URLs in the default web browser."""
import webbrowser


def register(registry):
    def open_url(payload):
        url = (payload or {}).get("url") or (payload or {}).get("href")
        if not url:
            return {"ok": False, "error": "missing url"}
        try:
            webbrowser.open(url)
            registry.emit("web", f"OPEN: {url}")
            return {"ok": True, "url": url}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    registry.register_tool("web.open", open_url)
