from . import BasePlugin
from loguru import logger

class OCRPlugin(BasePlugin):
    name = "ocr_cleanup"
    handles = ["ocr.text"]

    async def handle(self, event):
        # Skip assist-mode OCR; handled by capture_assist plugin
        if event.payload.get("assist"):
            return

        text = event.payload.get("text", "")
        # 간단한 노이즈 제거
        text = text.replace("\u200b", "").strip()
        event.payload["text"] = text
        # 이벤트는 원본 객체가 전달되므로 재전송 없이도 후속 플러그인이
        # 정제된 텍스트를 수신한다. 재발행하면 Overlay에 중복 표시되므로 생략.
        logger.debug(f"[{self.name}] cleaned ocr.text")
