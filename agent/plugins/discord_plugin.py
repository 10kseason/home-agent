from . import BasePlugin
from loguru import logger

class DiscordPlugin(BasePlugin):
    name = "discord_batch"
    handles = ["discord.batch"]  # payload: {"messages": [str, ...]}

    async def handle(self, event):
        msgs = event.payload.get("messages", [])
        if not msgs:
            return
        # 단순 병합 후 translator에 넘김
        joined = "\n".join(msgs)
        new_event = event.model_copy(update={
            "type": "discord.text",
            "payload": {"text": joined}
        })
        await self.ctx.bus.publish(new_event)  # translator가 처리
        logger.debug(f"[{self.name}] merged {len(msgs)} messages → discord.text")
