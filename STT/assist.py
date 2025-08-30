"""Microphone assist mode using faster-whisper.

This module captures microphone audio and streams it through a
WhisperModel. Recognized text is posted to the agent event bus so the
accessibility mode can react in real time.

If no input device is specified, the first available system microphone is
chosen automatically for convenience.

Example:
    python assist.py --model tiny.en

The module is designed to be lightweight and testable. The
`AssistTranscriber` accepts a pre-instantiated model and event posting
function, so unit tests can inject stubs without requiring the heavy
Whisper weights.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import argparse
import queue
import os
import threading
import time
from typing import Callable, Dict, List, Optional
import yaml

import numpy as np
try:
    import sounddevice as sd
except Exception:
    sd = None

try:
    import tkinter as tk
    from tkinter import scrolledtext
except Exception:
    tk = None
    scrolledtext = None


try:  # faster-whisper is optional for import-time
    from faster_whisper import WhisperModel
except Exception:  # pragma: no cover - handled gracefully
    WhisperModel = None

import requests
import importlib.util
import pathlib
import sys
try:
    from .cmd_detector import detect_command
except Exception:  # pragma: no cover - script execution
    spec = importlib.util.spec_from_file_location(
        "cmd_detector", pathlib.Path(__file__).with_name("cmd_detector.py")
    )
    cmd_mod = importlib.util.module_from_spec(spec)
    sys.modules["cmd_detector"] = cmd_mod
    assert spec.loader is not None
    spec.loader.exec_module(cmd_mod)
    detect_command = cmd_mod.detect_command

_EVENT_URL = (
    os.environ.get("EVENT_URL")
    or os.environ.get("AGENT_EVENT_URL")
    or os.environ.get("OVERLAY_EVENT_URL")
    or "http://127.0.0.1:8350/event"
)
_EVENT_KEY = os.environ.get("EVENT_KEY") or os.environ.get("AGENT_EVENT_KEY")


def _post_event(_type: str, _payload: dict, _prio: int = 5) -> None:
    """Send event to overlay or agent bus."""
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
class AssistConfig:
    """Configuration for assistive STT."""

    engine: str = "faster-whisper"
    model: str = "small"
    compute_type: str = "int8"
    language: str = "auto"
    vad: str = "silero"
    chunk_sec: float = 7.0
    chunk_overlap_sec: float = 0.5
    beam_size: int = 3
    condition_on_previous_text: bool = False
    temperature_fallback: List[float] = field(
        default_factory=lambda: [0.0, 0.2, 0.4]
    )
    sample_rate: int = 16_000
    block_ms: int = 3_000  # transcribe every N milliseconds
    device_index: Optional[int] = None
    commands: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "capture": ["캡쳐", "캡처", "capture", "スクショ", "截图", "截屏"],
            "summarize": ["요약", "summary", "まとめて", "要約", "总结"],
            "translate": ["번역", "translate", "翻訳", "翻译"],
            "focus": ["집중모드", "집중 모드", "focus mode"],
            "repeat": ["다시", "repeat", "もう一度", "再读", "再来一次"],
            "stop": ["멈춰", "정지", "stop", "停止"],
        }
    )
    detection: Dict[str, int | bool] = field(
        default_factory=lambda: {"require_boundary": False, "cooldown_ms": 800}
    )


class AssistTranscriber:
    """Transcribe PCM16 audio and post results to the event bus."""

    def __init__(
        self,
        model: WhisperModel,
        event_func: Callable[[str, dict, int], None] = _post_event,
        config: AssistConfig | None = None,
        ui: "SubtitleUI" | None = None,
    ) -> None:
        self.model = model
        self.event_func = event_func
        self.cfg = config or AssistConfig()
        self.ui = ui

    def transcribe(self, pcm16: bytes) -> str:
        """Transcribe a chunk of PCM16 mono audio."""
        audio = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        language = None if self.cfg.language == "auto" else self.cfg.language
        kwargs = {
            "beam_size": self.cfg.beam_size,
            "condition_on_previous_text": self.cfg.condition_on_previous_text,
            "temperature": self.cfg.temperature_fallback[0]
            if self.cfg.temperature_fallback
            else 0.0,
            "vad_filter": bool(self.cfg.vad),
            "chunk_length": self.cfg.chunk_sec,
            "chunk_overlap": self.cfg.chunk_overlap_sec,
        }
        try:
            segments, _ = self.model.transcribe(audio, language=language, **kwargs)
        except TypeError:
            segments, _ = self.model.transcribe(audio, language=language)
        text = "".join(seg.text for seg in segments).strip()
        if text:
            payload = {
                "text": text,
                "source": "assist",
                "stt_model": getattr(self.model, "model_size", "unknown"),
            }
            self.event_func("stt.text", payload)
            if self.ui:
                self.ui.push(text)
            cmd = detect_command(text, self.cfg)
            if cmd:
                self.event_func(
                    "cmd.detected",
                    {"cmd": cmd, "ts": time.time()},
                    1,
                )
        return text


class SubtitleUI:
    """Simple tkinter window that shows live transcriptions."""

    def __init__(self):
        self.enabled = tk is not None
        self.queue: "queue.Queue[str]" = queue.Queue()
        self.root = None
        self.chat_box = None
        if self.enabled:
            self.root = tk.Tk()
            self.root.title("Assist STT")
            self.root.geometry("600x240+60+60")
            self.chat_box = scrolledtext.ScrolledText(
                self.root, bg="#101316", fg="#E6E6E6", font=("Arial", 14), wrap="word"
            )
            self.chat_box.pack(fill="both", expand=True, padx=16, pady=16)
            self.chat_box.configure(state="disabled")
            self.root.after(50, self._poll)

    def _poll(self):
        try:
            while True:
                text = self.queue.get_nowait()
                if self.chat_box is not None:
                    self.chat_box.configure(state="normal")
                    self.chat_box.insert(tk.END, text + "\n")
                    self.chat_box.see(tk.END)
                    self.chat_box.configure(state="disabled")
                print(text)
        except queue.Empty:
            pass
        if self.enabled and self.root:
            self.root.after(50, self._poll)

    def push(self, text: str) -> None:
        self.queue.put(text)

    def loop(self) -> None:
        if self.enabled and self.root:
            self.root.mainloop()
        else:  # pragma: no cover - no GUI
            try:
                while True:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                pass


def _select_input_device(device_index: Optional[int]) -> Optional[int]:
    """Return a microphone device index, auto-detecting when unspecified."""
    if device_index is not None:
        return device_index
    if sd is None:
        return None
    try:
        default = sd.default.device  # type: ignore[attr-defined]
        if isinstance(default, (tuple, list)):
            default_in = default[0]
        else:
            default_in = default
        if default_in is not None and default_in >= 0:
            return int(default_in)
    except Exception:
        pass
    try:
        devices = sd.query_devices()  # pragma: no cover - environment dependent
        for i, dev in enumerate(devices):
            if dev.get("max_input_channels", 0) > 0:
                return i
    except Exception:  # pragma: no cover - environment dependent
        pass
    return None


def run(cfg: AssistConfig) -> None:
    """Capture microphone and stream to Whisper, showing subtitles."""
    if WhisperModel is None:
        raise RuntimeError("faster-whisper is not installed")

    model = WhisperModel(cfg.model, device="cpu", compute_type=cfg.compute_type)
    ui = SubtitleUI()
    transcriber = AssistTranscriber(model, _post_event, cfg, ui)

    q: "queue.Queue[bytes]" = queue.Queue()

    def callback(indata, frames, time, status):  # pragma: no cover - realtime
        q.put(bytes(indata))

    blocksize = int(cfg.sample_rate * (cfg.block_ms / 1000))
    device = _select_input_device(cfg.device_index)

    def worker():  # pragma: no cover - realtime loop
        buf = bytearray()
        print("[assist] Listening...")
        while True:
            buf.extend(q.get())
            if len(buf) >= blocksize * 2:
                transcriber.transcribe(bytes(buf))
                buf.clear()

    with sd.RawInputStream(
        samplerate=cfg.sample_rate,
        blocksize=blocksize,
        dtype="int16",
        channels=1,
        callback=callback,
        device=device,
    ):
        threading.Thread(target=worker, daemon=True).start()
        ui.loop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Assistive microphone STT")
    parser.add_argument("--config", help="Path to Assist-config.yaml", default=None)
    parser.add_argument("--model", help="Whisper model size", default="small")
    parser.add_argument("--device-index", type=int, default=None, help="Input device index")
    parser.add_argument(
        "--block-ms", type=int, default=3000, help="Chunk size in milliseconds"
    )
    args = parser.parse_args()

    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        cfg_data = {}
        cfg_data.update(data.get("stt") or {})
        cfg_data.update(data.get("assist") or {})
        cfg = AssistConfig(**cfg_data)
    else:
        cfg = AssistConfig(
            model=args.model,
            block_ms=args.block_ms,
            device_index=args.device_index,
        )
    run(cfg)


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
