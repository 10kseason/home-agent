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
        # 후속 플러그인에 동일 이벤트가 전달되므로 재전송은 필요 없다.
        # 재전송 시 Overlay에 중복 표기가 발생하므로 제거한다.
        logger.debug(f"[{self.name}] cleaned capture_assist.text")
