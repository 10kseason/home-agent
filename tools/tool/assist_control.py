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


def stt_start(_: Dict[str, Any]) -> Dict[str, Any]:
    return _post("stt.start")


def stt_stop(_: Dict[str, Any]) -> Dict[str, Any]:
    return _post("stt.stop")


def ocr_start(_: Dict[str, Any]) -> Dict[str, Any]:
    return _post("ocr.start")


def ocr_stop(_: Dict[str, Any]) -> Dict[str, Any]:
    return _post("ocr.stop")


def stt_assist_start(_: Dict[str, Any]) -> Dict[str, Any]:
    return _post("stt_assist.start")


def stt_assist_stop(_: Dict[str, Any]) -> Dict[str, Any]:
    return _post("stt_assist.stop")


def ocr_assist_start(_: Dict[str, Any]) -> Dict[str, Any]:
    return _post("ocr_assist.start")


def ocr_assist_stop(_: Dict[str, Any]) -> Dict[str, Any]:
    return _post("ocr_assist.stop")


def assist_on(_: Dict[str, Any]) -> Dict[str, Any]:
    return _post("assist.on")


def assist_off(_: Dict[str, Any]) -> Dict[str, Any]:
    return _post("assist.off")


TOOL_HANDLERS = {
    "stt_start": stt_start,
    "stt_stop": stt_stop,
    "ocr_start": ocr_start,
    "ocr_stop": ocr_stop,
    "stt_assist_start": stt_assist_start,
    "stt_assist_stop": stt_assist_stop,
    "ocr_assist_start": ocr_assist_start,
    "ocr_assist_stop": ocr_assist_stop,
    "assist_on": assist_on,
    "assist_off": assist_off,
}
