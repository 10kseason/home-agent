"""
Overlay sink plugin: forwards selected events (stt./ocr.) to the local Overlay app.
Place this file at: agent/plugins/overlay_sink.py
"""
from typing import List, Dict, Any, Optional
import os, asyncio

try:
    import httpx  # async http client
except Exception:  # optional
    httpx = None
import requests
from loguru import logger

# Import the BasePlugin from the same package so the loader finds this subclass
from . import BasePlugin

OVERLAY_EVENT_URL = os.environ.get("OVERLAY_EVENT_URL", "http://127.0.0.1:8350/overlay/event")

class OverlaySink(BasePlugin):
    name = "overlay_sink"
    handles: List[str] = ["stt.", "ocr."]  # subscribe prefixes

    async def _post(self, payload: Dict[str, Any]) -> None:
        try:
            if httpx is not None:
                async with httpx.AsyncClient(timeout=4.0) as client:
                    r = await client.post(OVERLAY_EVENT_URL, json=payload)
                    if r.status_code >= 300:
                        logger.debug(f"[overlay_sink] POST {r.status_code}: {r.text[:200]}")
            else:
                def _do():
                    try:
                        requests.post(OVERLAY_EVENT_URL, json=payload, timeout=4.0)
                    except Exception:
                        pass
                await asyncio.to_thread(_do)
        except Exception as e:
            logger.debug(f"[overlay_sink] post failed: {e}")

    async def handle(self, event) -> None:
        # Forward as-is; Overlay routes by 'type'
        payload = {
            "type": getattr(event, "type", "event"),
            "payload": getattr(event, "payload", {}) or {},
            "priority": int(getattr(event, "priority", 5)),
            "timestamp": getattr(event, "timestamp", None),
        }
        await self._post(payload)
