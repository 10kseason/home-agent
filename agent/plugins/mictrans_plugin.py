from . import BasePlugin
from loguru import logger

class MicTransPlugin(BasePlugin):
    """Clean microphone transcription text."""
    name = "mictrans_cleanup"
    handles = ["mictrans.text"]

    async def handle(self, event):
        text = event.payload.get("text", "")
        event.payload["text"] = text.strip()
        logger.debug(f"[{self.name}] cleaned mictrans.text")
