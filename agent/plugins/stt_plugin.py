from . import BasePlugin
from loguru import logger

class STTPlugin(BasePlugin):
    name = "stt_cleanup"
    handles = ["stt.text"]

    async def handle(self, event):
        text = event.payload.get("text", "")
        # 반복 제거 (예: "I'm going to do it" 반복 등)
        parts = [p.strip() for p in text.split()]
        dedup = []
        for p in parts:
            if not dedup or dedup[-1] != p:
                dedup.append(p)
        cleaned = " ".join(dedup)
        event.payload["text"] = cleaned
        # The event bus delivers this event to all subscribers in order,
        # so republishing would cause duplicate processing. Simply update
        # the payload and allow downstream handlers (e.g. translator) to
        # see the cleaned text.
        logger.debug(f"[{self.name}] cleaned stt.text")
