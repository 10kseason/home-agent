from __future__ import annotations

"""Overlay send plugin: republish overlay messages for relay.

This plugin listens for :code:`overlay.send` events and re-emits them as
:code:`overlay.toast` events tagged with ``_relay``.  The overlay sink will then
forward the message to the Luna Overlay application while the overlay toast
plugin shows a local Windows notification.
"""

from loguru import logger

from agent.schemas import Event
from . import BasePlugin


class OverlaySendPlugin(BasePlugin):
    """Republish overlay messages so sinks and toasts handle them."""

    name = "overlay_send"
    handles = ["overlay.send"]

    async def handle(self, event) -> None:
        payload = getattr(event, "payload", {}) or {}
        if not payload:
            logger.debug("[overlay_send] empty payload")
            return

        # Ensure relay flag so the overlay sink forwards the event and the
        # overlay toast plugin displays a notification.
        new_payload = dict(payload)
        new_payload.setdefault("_relay", True)

        await self.ctx.bus.publish(
            Event(
                type="overlay.toast",
                payload=new_payload,
                priority=getattr(event, "priority", 5),
                source=getattr(event, "source", "agent"),
            )
        )
