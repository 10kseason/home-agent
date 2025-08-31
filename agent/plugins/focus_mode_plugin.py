from . import BasePlugin
from loguru import logger

class FocusModePlugin(BasePlugin):
    """Minimal toggle handler for Focus Mode state."""

    name = "focus_mode"
    handles = ["FocusMode.toggle"]

    async def handle(self, event):
        on = bool(event.payload.get("on"))
        self.ctx.focus_mode = on
        logger.info(f"[focus_mode] {'enabled' if on else 'disabled'}")
