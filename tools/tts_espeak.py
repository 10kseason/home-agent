from __future__ import annotations

"""eSpeak NG TTS helper with segmentation and pronunciation dictionary."""

import re
import subprocess
import time
from pathlib import Path
from typing import List

import yaml

# Load configuration once
_ROOT = Path(__file__).resolve().parents[1]
_CFG_PATH = _ROOT / "config.yaml"
_CFG: dict | None = None
_HAS_ESPEAK: bool | None = None


def _load_cfg() -> dict:
    global _CFG
    if _CFG is None:
        try:
            data = yaml.safe_load(_CFG_PATH.read_text(encoding="utf-8"))
            _CFG = data.get("tts", {}) or {}
        except Exception:
            _CFG = {}
    return _CFG


def _espeak_available() -> bool:
    global _HAS_ESPEAK
    if _HAS_ESPEAK is None:
        try:
            subprocess.run(["espeak-ng", "--version"], check=False, capture_output=True)
            _HAS_ESPEAK = True
        except Exception:
            _HAS_ESPEAK = False
    return bool(_HAS_ESPEAK)


# Pronunciation dictionary rules
_REPLACEMENTS: List[tuple[re.Pattern[str], str | callable]] = [
    (re.compile(r"(\d+)\s*GB"), lambda m: f"{m.group(1)} 기가바이트"),
    (re.compile(r"(\d+)%"), lambda m: f"{m.group(1)} 퍼센트"),
    (re.compile(r"°C"), lambda m: "도씨"),
    (re.compile(r"\.pdf\b", re.IGNORECASE), lambda m: " 피디에프"),
    (re.compile(r"\.zip\b", re.IGNORECASE), lambda m: " 집"),
    (re.compile(r"\.png\b", re.IGNORECASE), lambda m: " 피앤지"),
    (re.compile(r"\bURL\b"), lambda m: "유알엘"),
    (re.compile(r"\bHTTP\b"), lambda m: "에이치티티피"),
    (re.compile(r"\bCPU\b"), lambda m: "씨피유"),
    (re.compile(r"\bGPU\b"), lambda m: "지피유"),
    (re.compile(r"\bAI\b"), lambda m: "에이아이"),
    (re.compile(r"\bLLM\b"), lambda m: "엘엘엠"),
]


def _apply_dict(text: str) -> str:
    out = text
    for pat, repl in _REPLACEMENTS:
        out = pat.sub(repl, out)
    return out


def _segment(text: str) -> List[str]:
    cfg = _load_cfg().get("espeak_ng", {})
    # chunk_ms currently unused but reserved for future timing logic
    # split by sentence-ending punctuation and optionally commas
    parts: List[str] = []
    for sent in re.split(r"(?<=[.!?])\s+", text):
        sent = sent.strip()
        if not sent:
            continue
        if sent.count(",") >= 2:
            parts.extend(p.strip() for p in sent.split(",") if p.strip())
        else:
            parts.append(sent)
    return parts


def speak(text: str, lang: str = "ko") -> None:
    """Speak ``text`` using eSpeak NG with config-driven parameters."""
    if not _espeak_available():
        return
    cfg = _load_cfg()
    ecfg = cfg.get("espeak_ng", {})
    voices = ecfg.get("voices", {})
    voice = voices.get(lang, voices.get("ko", "ko"))
    rate = int(str(ecfg.get("rate", "+10")))
    pitch = int(str(ecfg.get("pitch", 0)))
    volume = int(str(ecfg.get("volume", 100)))
    gap = int(str(ecfg.get("word_gap_ms", 20)))
    buffer_ms = int(str(ecfg.get("buffer_ms", 300)))
    speed = 175 + rate  # espeak default ~175 wpm
    text = _apply_dict(text)
    for seg in _segment(text):
        cmd = [
            "espeak-ng",
            "-v",
            voice,
            "-s",
            str(speed),
            "-p",
            str(pitch),
            "-a",
            str(volume),
            "-g",
            str(gap),
            seg,
        ]
        subprocess.run(cmd, check=False)
        time.sleep(buffer_ms / 1000.0)
