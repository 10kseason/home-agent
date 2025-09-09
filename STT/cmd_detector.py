import re
import time
import os
from typing import Optional, Callable

import requests

_EVENT_URL = (
    os.environ.get("EVENT_URL")
    or os.environ.get("AGENT_EVENT_URL")
    or "http://127.0.0.1:8350/event"
)
_EVENT_KEY = os.environ.get("EVENT_KEY") or os.environ.get("AGENT_EVENT_KEY")

_LAST_CMD_MS = -1e9
# Per-command last emit timestamp (ms) to avoid duplicates
_CMD_LAST_TS = {}

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

def process_text(text: str, cfg, event_func: Callable[[str, dict, int], None] | None = None) -> Optional[str]:
    """Detect command and post ``cmd.detected`` event if found.

    Parameters
    ----------
    text:
        Recognized speech text.
    cfg:
        Assist configuration containing command mappings.
    event_func:
        Optional custom event poster. Defaults to posting to the agent server.
    """
    cmd = detect_command(text, cfg)
    if cmd:
        # Allow disabling command emission via env (ASSIST_ENABLE_DETECTED=0)
        enabled_env = os.environ.get("ASSIST_ENABLE_DETECTED")
        enabled = True if enabled_env is None else str(enabled_env).strip() not in ("0", "false", "no", "off", "")
        if not enabled:
            return cmd

        # De-duplicate same command within a TTL window
        now_ms = time.time() * 1000.0
        try:
            dup_ttl_ms = int((getattr(cfg, 'detection', None) or {}).get('duplicate_ttl_ms', 2000))
        except Exception:
            dup_ttl_ms = 2000
        last = _CMD_LAST_TS.get(cmd, -1e12)
        if now_ms - last < dup_ttl_ms:
            return cmd  # suppress duplicate emit
        _CMD_LAST_TS[cmd] = now_ms

        _post_event = event_func or _default_post
        _post_event("cmd.detected", {"cmd": cmd, "ts": time.time()}, 1)
    return cmd

def _default_post(_type: str, payload: dict, _prio: int = 5) -> None:
    try:
        headers = {"Content-Type": "application/json"}
        if _EVENT_KEY:
            headers["X-Agent-Key"] = _EVENT_KEY
        requests.post(_EVENT_URL, json={"type": _type, "payload": payload, "priority": _prio}, headers=headers, timeout=3)
    except Exception:
        pass

def _reset_state() -> None:
    global _LAST_CMD_MS
    _LAST_CMD_MS = -1e9
    try:
        _CMD_LAST_TS.clear()
    except Exception:
        pass
