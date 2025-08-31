from __future__ import annotations
import os
import requests
from typing import Dict, Any

_EVENT_URL = os.environ.get("AGENT_EVENT_URL") or os.environ.get("EVENT_URL") or "http://127.0.0.1:8350/event"
_EVENT_KEY = os.environ.get("AGENT_EVENT_KEY") or os.environ.get("EVENT_KEY")


def _post(name: str) -> Dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if _EVENT_KEY:
        headers["X-Agent-Key"] = _EVENT_KEY
    try:
        requests.post(_EVENT_URL, json={"type": name, "payload": {}}, headers=headers, timeout=3)
    except Exception:
        pass
    return {"sent": name}


def stt_assist_start(_: Dict[str, Any]) -> Dict[str, Any]:
    return _post("stt_assist.start")


def ocr_assist_start(_: Dict[str, Any]) -> Dict[str, Any]:
    return _post("ocr_assist.start")


def stt_assist_stop(_: Dict[str, Any]) -> Dict[str, Any]:
    return _post("stt_assist.stop")


def ocr_assist_stop(_: Dict[str, Any]) -> Dict[str, Any]:
    return _post("ocr_assist.stop")


TOOL_HANDLERS = {
    "stt_assist.start": stt_assist_start,
    "ocr_assist.start": ocr_assist_start,
    "stt_assist.stop": stt_assist_stop,
    "ocr_assist.stop": ocr_assist_stop,
}
