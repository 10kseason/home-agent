from __future__ import annotations

"""
Overlay send plugin v2: relay to BOTH toast and overlay message feed.

- Listens for `overlay.send`.
- Re-emits:
  1) `overlay.toast` with `_relay=True` for Windows toast
  2) `overlay.message` with `_relay=True` for Luna Overlay message feed
"""

import time
from loguru import logger
from agent.schemas import Event
from . import BasePlugin


class OverlaySendPlugin(BasePlugin):
    name = "overlay_send"
    handles = ["overlay.send"]

    def _normalize(self, payload: dict) -> dict:
        """Normalize fields so both toast and overlay message have what they need."""
        p = dict(payload or {})
        p.setdefault("_relay", True)
        p.setdefault("title", p.get("title", "") or p.get("heading", "") or "")
        p.setdefault("text", p.get("text", "") or p.get("message", "") or "")
        p.setdefault("level", p.get("level", "info"))
        p.setdefault("channel", p.get("channel", "system"))
        p.setdefault("ts_ms", int(time.time() * 1000))
        return p

    async def handle(self, event) -> None:
        payload = getattr(event, "payload", {}) or {}
        if not payload:
            logger.debug("[overlay_send] empty payload")
            return

        norm = self._normalize(payload)

        # 1) 윈도우 토스트용 (sink가 /overlay/event 로 보냄)
        await self.ctx.bus.publish(
            Event(
                type="overlay.toast",
                payload=norm,
                priority=getattr(event, "priority", 5),
                source=getattr(event, "source", "agent"),
            )
        )

        # 2) 오버레이 메시지 피드용 (sink가 /event 로 보냄)
        await self.ctx.bus.publish(
            Event(
                type="overlay.message",
                payload=norm,
                priority=getattr(event, "priority", 5),
                source=getattr(event, "source", "agent"),
            )
        )

        logger.debug("[overlay_send] relayed to overlay.toast and overlay.message")
