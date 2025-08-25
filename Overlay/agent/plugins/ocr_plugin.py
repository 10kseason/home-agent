from . import BasePlugin
from loguru import logger

class OCRPlugin(BasePlugin):
    name = "ocr_cleanup"
    handles = ["ocr.text"]

    async def handle(self, event):
        text = event.payload.get("text", "")
        # 간단한 노이즈 제거
        text = text.replace("\u200b", "").strip()
        # 다음 단계로 번역을 유도하기 위해 translator가 듣는 동일 타입을 다시 publish
        event.payload["text"] = text
        await self.ctx.bus.publish(event)  # 재전송 → translator가 수신
        logger.debug(f"[{self.name}] cleaned and republished ocr.text")
