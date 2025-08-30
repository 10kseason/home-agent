import re
import time
from typing import Optional

_LAST_CMD_MS = -1e9

def detect_command(text: str, cfg) -> Optional[str]:
    """Return canonical command name if detected respecting cooldown."""
    global _LAST_CMD_MS
    now_ms = time.time() * 1000.0
    cooldown = (cfg.detection or {}).get("cooldown_ms", 800)
    if now_ms - _LAST_CMD_MS < cooldown:
        return None
    commands = cfg.commands or {}
    require_boundary = (cfg.detection or {}).get("require_boundary", False)
    for cmd, syns in commands.items():
        for syn in syns:
            if require_boundary:
                if re.search(rf"(^|\s){re.escape(syn)}(\s|$)", text):
                    _LAST_CMD_MS = now_ms
                    return cmd
            else:
                if syn in text:
                    _LAST_CMD_MS = now_ms
                    return cmd
    return None

def _reset_state() -> None:
    global _LAST_CMD_MS
    _LAST_CMD_MS = -1e9
