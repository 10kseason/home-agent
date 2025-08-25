from typing import Literal
from loguru import logger

AutonomyLevel = Literal["auto", "confirm", "hybrid"]

class Policy:
    def __init__(self, level: AutonomyLevel = "auto"):
        self.level = level

    async def should_execute(self, intent: str, importance: int = 5) -> bool:
        """
        intent: e.g., "translate", "summarize", "overlay"
        importance: 1..10 (1 critical)
        """
        if self.level == "auto":
            return True
        if self.level == "confirm":
            logger.info(f"[CONFIRM] Would execute {intent} (importance={importance})")
            return False
        if self.level == "hybrid":
            return importance <= 4  # critical auto, others confirm
        return True
