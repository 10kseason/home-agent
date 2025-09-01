from . import BasePlugin
from loguru import logger

class CaptureAssistPlugin(BasePlugin):
    """Cleanup for Capture Assist text."""
    name = "capture_assist"
    handles = ["capture_assist.text"]

    async def handle(self, event):
        if not event.payload.get("assist"):
            return
        text = event.payload.get("text", "")
        text = text.replace("\u200b", "").strip()
        event.payload["text"] = text
        await self.ctx.bus.publish(event)
        logger.debug(f"[{self.name}] cleaned and republished capture_assist.text")
