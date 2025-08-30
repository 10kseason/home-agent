"""Assistive OCR script using PaddleOCR.
Captures the full screen and posts recognized text to the agent event bus."""
from __future__ import annotations

import os
import io
from typing import List

import numpy as np
from PIL import Image
from mss import mss
import requests

try:  # PaddleOCR is heavy; import lazily
    from paddleocr import PaddleOCR
except Exception:  # pragma: no cover - runtime dependency
    PaddleOCR = None

_EVENT_URL = (
    os.environ.get("EVENT_URL")
    or os.environ.get("AGENT_EVENT_URL")
    or os.environ.get("OVERLAY_EVENT_URL")
    or "http://127.0.0.1:8350/event"
)
_EVENT_KEY = os.environ.get("EVENT_KEY") or os.environ.get("AGENT_EVENT_KEY")


def _post_event(_type: str, _payload: dict, _prio: int = 5) -> None:
    try:
        headers = {"Content-Type": "application/json"}
        if _EVENT_KEY:
            headers["X-Agent-Key"] = _EVENT_KEY
        requests.post(
            _EVENT_URL,
            json={"type": _type, "payload": _payload, "priority": _prio},
            headers=headers,
            timeout=3,
        )
    except Exception:
        pass


def _capture_screen() -> Image.Image:
    with mss() as sct:
        shot = sct.grab(sct.monitors[1])
        img = Image.frombytes("RGB", shot.size, shot.rgb)
        return img


def _run_ocr(img: Image.Image) -> str:
    if PaddleOCR is None:
        raise RuntimeError("paddleocr is not installed")
    ocr = PaddleOCR(use_angle_cls=True, lang="korean")
    np_img = np.array(img)
    result = ocr.ocr(np_img, cls=True)
    lines: List[str] = []
    for line in result:
        for item in line:
            lines.append(item[1][0])
    return "\n".join(lines).strip()


def main() -> None:
    img = _capture_screen()
    text = _run_ocr(img)
    if text:
        _post_event("ocr.text", {"text": text, "source": "paddle_assist"})
        print(text)


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
