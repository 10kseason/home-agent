import yaml
from loguru import logger
from typing import Any, Dict
from .event_bus import EventBus
from .policy import Policy
from . import sinks as sinks_mod

class Ctx:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.bus = EventBus(
            max_queue_size=config["limits"]["max_queue_size"],
            dedup_window=config["limits"]["dedup_window_seconds"],
        )
        self.policy = Policy(config["policy"]["autonomy_level"])
        self.sinks = sinks_mod

def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
