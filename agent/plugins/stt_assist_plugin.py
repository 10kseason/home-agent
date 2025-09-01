from . import BasePlugin
from loguru import logger

class STTAssistPlugin(BasePlugin):
    """Cleanup for STT Assist transcripts."""
    name = "stt_assist"
    handles = ["stt.text"]

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
        logger.debug(f"[{self.name}] cleaned stt.text")
