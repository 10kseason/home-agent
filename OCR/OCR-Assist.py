"""Assistive OCR script using PaddleOCR.
Captures the full screen and posts recognized text to the agent event bus."""
from __future__ import annotations

import os

import argparse
from dataclasses import dataclass, field
from typing import List, Optional

import pathlib
import yaml

import numpy as np
from PIL import Image
from mss import mss
import requests
from tools.tts_espeak import speak

try:  # EasyOCR downloads models on first use
    import easyocr
except Exception:  # pragma: no cover - runtime dependency
    easyocr = None

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


@dataclass
class OCRAssistConfig:
    """Configuration for assistive OCR capture."""

    monitor: int = 1  # 1-based monitor index
    region: Optional[List[int]] = None  # [left, top, width, height]
    langs: List[str] = field(default_factory=lambda: ["ko"])
    gpu: bool = False
    announce_text: str = "찍었습니다."
    event_url: Optional[str] = None
    event_key: Optional[str] = None


def load_config(path: str | None = None) -> OCRAssistConfig:
    """Load OCR assist configuration from YAML."""

    if path is None:
        default = pathlib.Path(__file__).with_name("Assist-config.yaml")
        path = str(default) if default.exists() else None

    data: dict = {}
    if path:
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        cap = raw.get("capture") or {}
        data["monitor"] = cap.get("monitor", 1)
        data["region"] = cap.get("region")
        ocr = raw.get("ocr") or {}
        data["langs"] = ocr.get("langs", ["ko"])
        data["gpu"] = ocr.get("gpu", False)
        assist = raw.get("assist") or {}
        if "announce_text" in assist:
            data["announce_text"] = assist["announce_text"]
        event = raw.get("event") or {}
        data["event_url"] = event.get("url")
        data["event_key"] = event.get("key")
    return OCRAssistConfig(**data)


def _capture_screen(cfg: OCRAssistConfig) -> Image.Image:
    with mss() as sct:
        monitor = sct.monitors[cfg.monitor]
        if cfg.region:
            left, top, width, height = cfg.region
            region = {
                "left": monitor["left"] + left,
                "top": monitor["top"] + top,
                "width": width,
                "height": height,
            }
            shot = sct.grab(region)
        else:
            shot = sct.grab(monitor)
        img = Image.frombytes("RGB", shot.size, shot.rgb)
        return img


def _run_ocr(img: Image.Image, cfg: OCRAssistConfig) -> str:
    if easyocr is None:
        raise RuntimeError("easyocr is not installed")
    reader = easyocr.Reader(cfg.langs, gpu=cfg.gpu)
    np_img = np.array(img)
    result = reader.readtext(np_img, detail=0)
    return "\n".join(result).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Assistive screenshot OCR")
    parser.add_argument("--config", help="Path to Assist-config.yaml", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    global _EVENT_URL, _EVENT_KEY
    if cfg.event_url:
        _EVENT_URL = cfg.event_url
    if cfg.event_key:
        _EVENT_KEY = cfg.event_key

    img = _capture_screen(cfg)
    text = _run_ocr(img, cfg)
    if text:
        _post_event("ocr.text", {"text": text, "source": "easyocr_assist"})
        print(text)
        if cfg.announce_text:
            speak(cfg.announce_text, lang="ko")


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
