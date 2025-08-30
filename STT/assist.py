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

from dataclasses import dataclass
import argparse
import queue
import os
from typing import Callable, Optional

import numpy as np
try:
    import sounddevice as sd
except Exception:
    sd = None


try:  # faster-whisper is optional for import-time
    from faster_whisper import WhisperModel
except Exception:  # pragma: no cover - handled gracefully
    WhisperModel = None

import requests

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
    model_size: str = "tiny.en"
    sample_rate: int = 16_000
    block_ms: int = 3_000  # transcribe every N milliseconds
    device_index: Optional[int] = None


class AssistTranscriber:
    """Transcribe PCM16 audio and post results to the event bus."""

    def __init__(
        self,
        model: WhisperModel,
        event_func: Callable[[str, dict, int], None] = _post_event,
        config: AssistConfig | None = None,
    ) -> None:
        self.model = model
        self.event_func = event_func
        self.cfg = config or AssistConfig()

    def transcribe(self, pcm16: bytes) -> str:
        """Transcribe a chunk of PCM16 mono audio."""
        audio = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        segments, _ = self.model.transcribe(audio, language="en")
        text = "".join(seg.text for seg in segments).strip()
        if text:
            payload = {
                "text": text,
                "source": "assist",
                "stt_model": getattr(self.model, "model_size", "unknown"),
            }
            self.event_func("stt.text", payload)
        return text


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
    """Capture microphone and stream to Whisper."""
    if WhisperModel is None:
        raise RuntimeError("faster-whisper is not installed")

    model = WhisperModel(cfg.model_size, device="auto", compute_type="int8")
    transcriber = AssistTranscriber(model, _post_event, cfg)

    q: queue.Queue[bytes] = queue.Queue()

    def callback(indata, frames, time, status):  # pragma: no cover - realtime
        q.put(bytes(indata))

    blocksize = int(cfg.sample_rate * (cfg.block_ms / 1000))
    device = _select_input_device(cfg.device_index)
    with sd.RawInputStream(
        samplerate=cfg.sample_rate,
        blocksize=blocksize,
        dtype="int16",
        channels=1,
        callback=callback,
        device=device,
    ):
        buf = bytearray()
        print("[assist] Listening...")
        while True:  # pragma: no cover - realtime loop
            buf.extend(q.get())
            if len(buf) >= blocksize * 2:
                transcriber.transcribe(bytes(buf))
                buf.clear()


def main() -> None:
    parser = argparse.ArgumentParser(description="Assistive microphone STT")
    parser.add_argument("--model", default="tiny.en", help="Whisper model size")
    parser.add_argument("--device-index", type=int, default=None, help="Input device index")
    parser.add_argument(
        "--block-ms", type=int, default=3000, help="Chunk size in milliseconds"
    )
    args = parser.parse_args()

    cfg = AssistConfig(
        model_size=args.model,
        block_ms=args.block_ms,
        device_index=args.device_index,
    )
    run(cfg)


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
