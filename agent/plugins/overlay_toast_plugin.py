from __future__ import annotations

"""Overlay toast plugin: show Windows toast notifications for relayed messages.

The agent's server republishs :code:`overlay.toast` events with a ``_relay`` flag
via :func:`agent.server._toast_handler`.  This plugin listens for those relayed
messages and displays a local Windows toast so the user is notified even when
the message originated from elsewhere.

OverlaySink handles forwarding the same events to the overlay application.
"""

import asyncio
from loguru import logger

from agent.sinks import _toast_win10, _toast_winotify
from . import BasePlugin


class OverlayToastPlugin(BasePlugin):
    """Display Windows toasts for relayed overlay messages."""

    name = "overlay_toast"
    handles = ["overlay.toast"]

    async def handle(self, event) -> None:
        payload = getattr(event, "payload", {}) or {}
        # Only show toasts for events that were relayed by the server to avoid
        # duplicating local toast_notify() calls.
        if not payload.get("_relay"):
            return

        title = payload.get("title", "")
        text = payload.get("text", "")

        def _show() -> None:
            if _toast_win10(title, text):
                return
            if _toast_winotify(title, text):
                return
            logger.info(f"[TOAST] {title}: {text}")

        # Run the potentially blocking toast APIs in a thread.
        await asyncio.to_thread(_show)
