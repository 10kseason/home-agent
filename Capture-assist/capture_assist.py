"""Assistive OCR script using PaddleOCR.
Captures the full screen and posts recognized text to the agent event bus."""
from __future__ import annotations

import os

import argparse
from dataclasses import dataclass, field
from typing import List, Optional

import pathlib
import sys
import time

# Ensure repository root is on sys.path when executed directly
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import yaml

import numpy as np
from PIL import Image
from mss import mss
import requests as _rq
# eSpeak TTS (optional)
try:
    from tools.tts_espeak import speak
except Exception:  # pragma: no cover - optional dependency
    def speak(*args, **kwargs):
        return None

# Backwards-compatibility alias for tests expecting module-level `requests`
requests = _rq

try:  # EasyOCR downloads models on first use
    import easyocr
except Exception:  # pragma: no cover - runtime dependency
    easyocr = None

# ---- Luna Agent bridge (공통) ----
_EVENT_URL = (
    os.environ.get("AGENT_EVENT_URL")
    or os.environ.get("EVENT_URL")
    or "http://127.0.0.1:8765/event"
)
_EVENT_KEY = os.environ.get("EVENT_KEY") or os.environ.get("AGENT_EVENT_KEY")

LOG_PATH = pathlib.Path(__file__).with_name("capture_assist.log")


def _post_event(_type: str, _payload: dict, _prio: int = 5) -> None:
    headers = {"Content-Type": "application/json"}
    if _EVENT_KEY:
        headers["X-Agent-Key"] = _EVENT_KEY
    try:
        _rq.post(
            _EVENT_URL,
            json={"type": _type, "payload": _payload, "priority": _prio},
            headers=headers,
            timeout=3,
        )
    except Exception:
        pass


def _overlay_toast(message: str, title: str = "Assist-Capture") -> None:
    """Mirror messages to Lunar Bridge overlay as toasts."""
    message = (message or "").strip()
    if not message:
        return
    if len(message) > 240:
        message = message[:239] + "…"
    _post_event("overlay.toast", {"title": title, "text": message})


def _notify(msg: str) -> None:
    """Display a toast notification via the overlay only."""
    _overlay_toast(msg)


def _refine_with_jan(text: str) -> str:
    """Send OCR text to a jan-nano model served by LM Studio or Ollama."""
    endpoint = (
        os.environ.get("JAN_ENDPOINT")
        or os.environ.get("LM_STUDIO_ENDPOINT")
        or os.environ.get("OLLAMA_ENDPOINT")
    )
    if not endpoint or not text:
        return text
    model = (
        os.environ.get("JAN_MODEL")
        or os.environ.get("LM_STUDIO_MODEL")
        or os.environ.get("OLLAMA_MODEL")
        or "jan-nano"
    )
    api_key = (
        os.environ.get("JAN_API_KEY")
        or os.environ.get("LM_STUDIO_API_KEY")
        or os.environ.get("OLLAMA_API_KEY")
        or ""
    )
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Refine OCR output."},
            {"role": "user", "content": text},
        ],
        "temperature": 0.2,
        "reasoning": {"effort": "high"},
    }
    try:
        r = _rq.post(
            f"{endpoint}/chat/completions", headers=headers, json=payload, timeout=10
        )
        r.raise_for_status()
        data = r.json()
        return (data["choices"][0]["message"]["content"] or "").strip() or text
    except Exception:
        return text


def _clear_log(path: pathlib.Path | None = None) -> None:
    """Remove the capture assist log file if it exists."""
    log = path or LOG_PATH
    try:
        if log.exists():
            log.unlink()
    except Exception:
        pass


def _append_log(text: str, path: pathlib.Path | None = None) -> None:
    """Append a timestamped entry to the capture assist log."""
    log = path or LOG_PATH
    try:
        with open(log, "a", encoding="utf-8") as f:
            f.write(f"{time.time()}\t{text}\n")
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

    _notify("5초 뒤 촬영 합니다")
    time.sleep(5)
    img = _capture_screen(cfg)
    _notify("이미지를 OCR 중입니다..")
    text = _run_ocr(img, cfg)
    text = _refine_with_jan(text)
    if text:
        _post_event(
            "capture_assist.text",
            {"text": text, "source": "easyocr_assist", "assist": True},
        )
        
        # 추가 (호환용)
        _post_event("stt.text", {"text": text, "translation": "", "source": "easyocr_assist"})
        _overlay_toast(f"[Capture-assist] {text}")
        _notify("EasyOCR로 OCR했어요. Overlay 확인 해주세요.")
        _append_log(text)
        print(text)
        if cfg.announce_text:
            speak(cfg.announce_text, lang="ko")

if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
