from . import BasePlugin
from loguru import logger

class CaptureAssistPlugin(BasePlugin):
    """Normalize capture-assist text and republish for downstream handlers."""
    name = "capture_assist_cleanup"
    handles = ["capture_assist.text"]

    async def handle(self, event):
        text = event.payload.get("text", "").strip()
        event.payload["text"] = text
        # Republish so translator and others can consume
        await self.ctx.bus.publish(event)
        logger.debug(f"[{self.name}] cleaned and republished capture_assist.text")
