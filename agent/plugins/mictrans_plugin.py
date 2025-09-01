from . import BasePlugin
from loguru import logger

class MicTransPlugin(BasePlugin):
    """Cleanup for MicTrans transcripts."""
    name = "mictrans"
    handles = ["mictrans.text"]

    async def handle(self, event):
        if not event.payload.get("assist"):
            return
        text = event.payload.get("text", "")
        parts = [p.strip() for p in text.split()]
        dedup = []
        for p in parts:
            if not dedup or dedup[-1] != p:
                dedup.append(p)
        event.payload["text"] = " ".join(dedup)
        await self.ctx.bus.publish(event)
        logger.debug(f"[{self.name}] cleaned and republished mictrans.text")
