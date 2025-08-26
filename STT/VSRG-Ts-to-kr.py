#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VSRG Translator Pro
- System output loopback capture (no microphones) with dual-engine fallback
- VAD-based utterance segmentation + robust RMS-based forced segmentation
- STT via local faster-whisper or remote STT server
- Optional translation via LM Studio / OpenAI-compatible server
- Live English preview while speaking, finalize + translate on pause
- Minimal Tkinter overlay

Run:
  pip install -r requirements.txt
  python app.py --list-devices   # optional
  python app.py
"""

import argparse
import threading
import queue
import time
import sys
import os
import math
import platform
from dataclasses import dataclass
from typing import Optional, List, Deque, Dict, Any
from collections import deque
import base64
from diarizer import SpeakerDiarizer

import yaml
import numpy as np
import sounddevice as sd
import webrtcvad
import requests
import soundfile as sf
# ---- Luna Agent bridge (공통) ----
import requests as _rq
_AGENT_URL = os.environ.get("AGENT_EVENT_URL", "http://127.0.0.1:8765/plugin/event")
_AGENT_KEY = os.environ.get("AGENT_EVENT_KEY")

def _post_event(_type, _payload, _prio=5):
    try:
        headers = {'Content-Type': 'application/json'}
        if _AGENT_KEY:
            headers['X-Agent-Key'] = _AGENT_KEY
        _rq.post(_AGENT_URL, json={
            'type': _type, 'payload': _payload, 'priority': _prio
        }, headers=headers, timeout=3)
    except Exception:
        pass
def _log(s: str):
    print(f"[DEBUG] {s}")
# Optional GUI
try:
    import tkinter as tk
    from tkinter import ttk, scrolledtext
except Exception:
    tk = None
    ttk = None
    scrolledtext = None
    print("[WARN] Tkinter not available; running in console mode.", file=sys.stderr)


# 환경변수 fallback (subprocess에서 전달 안 될 때 대비)
if not os.environ.get("AGENT_EVENT_URL"):
    os.environ["AGENT_EVENT_URL"] = "http://127.0.0.1:8350/plugin/event"
    print("[STT] Set fallback AGENT_EVENT_URL")

print(f"[STT] Using AGENT_EVENT_URL: {os.environ.get('AGENT_EVENT_URL')}")


# STT (local)
try:
    from faster_whisper import WhisperModel
except Exception:
    WhisperModel = None
    print("[INFO] faster-whisper not available (only needed when stt.backend=local or preview.enabled).", file=sys.stderr)

# Fallback loopback
try:
    import soundcard as sc
except Exception:
    sc = None
    print("[INFO] soundcard not available (fallback engine).", file=sys.stderr)

# Per-application capture
try:
    from ctypes import POINTER, cast, string_at
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioClient, IAudioCaptureClient
    from pycaw.constants import AUDCLNT_STREAMFLAGS_LOOPBACK, AUDCLNT_SHAREMODE_SHARED
except Exception:
    AudioUtilities = None
    IAudioClient = None
    IAudioCaptureClient = None
    CLSCTX_ALL = None
    AUDCLNT_STREAMFLAGS_LOOPBACK = 0
    AUDCLNT_SHAREMODE_SHARED = 0
    print("[INFO] pycaw not available (app capture disabled).", file=sys.stderr)


def _log(s: str):
    print(f"[DEBUG] {s}")

def _rms_dbfs(x: np.ndarray) -> float:
    if x.size == 0:
        return -120.0
    # support int16 or float32 mono
    if x.dtype != np.float32:
        x = x.astype(np.float32) / 32768.0 if x.dtype == np.int16 else x.astype(np.float32)
    x = np.clip(x, -1.0, 1.0)
    rms = float(np.sqrt(np.mean(x * x))) if x.size else 0.0
    if rms <= 1e-9:
        return -120.0
    return 20.0 * math.log10(rms)


# =========================
# Config structures
# =========================
@dataclass
class CaptureCfg:
    mode: str
    device_index: int
    device_name: str
    sample_rate: int
    block_ms: int
    channels: int
    dtype: str
    apps: List[str]

@dataclass
class VADCfg:
    aggressiveness: int
    silence_pad_ms: int
    max_segment_ms: int

@dataclass
class ForceCfg:
    enable: bool
    rms_speech_threshold_dbfs: float
    min_forced_segment_ms: int
    sustained_loud_ms: int
    max_buffer_ms: int

@dataclass
class STTCfg:
    backend: str
    model: str
    compute_type: str
    language: Optional[str]
    beam_size: int
    vad_filter: bool
    server_url: str

@dataclass
class TranslateCfg:
    enable: bool
    url: str
    model: str
    temperature: float
    max_tokens: int
    system_prompt: str

@dataclass
class UICfg:
    enable: bool
    font_family: str
    font_size_en: int
    font_size_ko: int
    width: int
    height: int
    topmost: bool
    theme_bg: str
    theme_fg: str
    accent_fg: str

@dataclass
class DebugCfg:
    log_segments: bool
    write_wav_segments: bool
    list_devices_on_start: bool

@dataclass
class PreviewCfg:
    enable: bool
    every_ms: int
    window_ms: int
    model: str
    compute_type: str
    rms_gate_dbfs: float

@dataclass
class Config:
    capture: CaptureCfg
    vad: VADCfg
    force: ForceCfg
    stt: STTCfg
    translate: TranslateCfg
    ui: UICfg
    debug: DebugCfg
    preview: PreviewCfg

def load_config(path: str = "config.yaml") -> Config:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cap = cfg.get("capture", {})
    vad = cfg.get("vad", {})
    frc = cfg.get("force", {})
    stt = cfg.get("stt", {})
    trn = cfg.get("translate", {})
    ui  = cfg.get("ui", {})
    dbg = cfg.get("debug", {})
    prv = cfg.get("preview", {})

    return Config(
        capture=CaptureCfg(
            mode=cap.get("mode", "auto"),
            device_index=int(cap.get("device_index", -1)),
            device_name=cap.get("device_name", ""),
            sample_rate=int(cap.get("sample_rate", 16000)),
            block_ms=int(cap.get("block_ms", 20)),
            channels=int(cap.get("channels", 1)),
            dtype=cap.get("dtype", "float32"),
            apps=[a.lower() for a in cap.get("apps", [])],
        ),
        vad=VADCfg(
            aggressiveness=int(vad.get("aggressiveness", 2)),
            silence_pad_ms=int(vad.get("silence_pad_ms", 300)),
            max_segment_ms=int(vad.get("max_segment_ms", 15000)),
        ),
        force=ForceCfg(
            enable=bool(frc.get("enable", True)),
            rms_speech_threshold_dbfs=float(frc.get("rms_speech_threshold_dbfs", -45.0)),
            min_forced_segment_ms=int(frc.get("min_forced_segment_ms", 1200)),
            sustained_loud_ms=int(frc.get("sustained_loud_ms", 3000)),
            max_buffer_ms=int(frc.get("max_buffer_ms", 20000)),
        ),
        stt=STTCfg(
            backend=stt.get("backend", "local"),
            model=stt.get("model", "small.en"),
            compute_type=stt.get("compute_type", "auto"),
            language=stt.get("language", "en"),
            beam_size=int(stt.get("beam_size", 1)),
            vad_filter=bool(stt.get("vad_filter", False)),
            server_url=stt.get("server_url", "http://127.0.0.1:8008/v1/transcribe-raw"),
        ),
        translate=TranslateCfg(
            enable=bool(trn.get("enable", True)),
            url=trn.get("url", "http://127.0.0.1:1234/v1/chat/completions"),
            model=trn.get("model", "Qwen2.5-7B-Instruct"),
            temperature=float(trn.get("temperature", 0.2)),
            max_tokens=int(trn.get("max_tokens", 128)),
            system_prompt=trn.get("system_prompt", "Translate to Korean."),
        ),
        ui=UICfg(
            enable=bool(ui.get("enable", True)),
            font_family=ui.get("font_family", "Malgun Gothic"),
            font_size_en=int(ui.get("font_size_en", 14)),
            font_size_ko=int(ui.get("font_size_ko", 16)),
            width=int(ui.get("width", 980)),
            height=int(ui.get("height", 260)),
            topmost=bool(ui.get("topmost", True)),
            theme_bg=ui.get("theme_bg", "#101316"),
            theme_fg=ui.get("theme_fg", "#E6E6E6"),
            accent_fg=ui.get("accent_fg", "#9ADCF8"),
        ),
        debug=DebugCfg(
            log_segments=bool(dbg.get("log_segments", True)),
            write_wav_segments=bool(dbg.get("write_wav_segments", False)),
            list_devices_on_start=bool(dbg.get("list_devices_on_start", False)),
        ),
        preview=PreviewCfg(
            enable=bool(prv.get("enable", False)),
            every_ms=int(prv.get("every_ms", 600)),
            window_ms=int(prv.get("window_ms", 4000)),
            model=prv.get("model", "tiny.en"),
            compute_type=prv.get("compute_type", "int8"),
            rms_gate_dbfs=float(prv.get("rms_gate_dbfs", -47.0)),
        ),
    )

# =========================
# Device listing
# =========================
def list_devices():
    print("=== sounddevice audio devices (look for '(loopback)' on Windows/WASAPI) ===")
    devs = sd.query_devices()
    hostapis = sd.query_hostapis()
    for idx, d in enumerate(devs):
        ha = hostapis[d['hostapi']]['name']
        print(f"[{idx:02d}] {d['name']} | {ha} | in={d['max_input_channels']} out={d['max_output_channels']}")
    if sc is not None:
        try:
            spk = sc.default_speaker()
            print(f"\n[soundcard] Default speaker: {spk.name}")
            mics = sc.all_microphones(include_loopback=True)
            loops = [m for m in mics if m.isloopback]
            if loops:
                print("[soundcard] Loopback-capable devices:")
                for m in loops:
                    print(f"  - {m.name}")
        except Exception as e:
            print(f"[soundcard] probe error: {e}", file=sys.stderr)


def is_windows() -> bool:
    return platform.system().lower().startswith("win")


def _find_wasapi_loopback_index(prefer_name: Optional[str] = None) -> Optional[int]:
    try:
        devs = sd.query_devices()
        hostapis = sd.query_hostapis()
    except Exception as e:
        print(f"[WARN] Unable to query devices: {e}", file=sys.stderr)
        return None

    candidates = []
    for idx, d in enumerate(devs):
        ha = hostapis[d['hostapi']]['name']
        if 'wasapi' in ha.lower() and d.get('max_input_channels', 0) > 0:
            name = d.get('name', '')
            if 'loopback' in name.lower():
                candidates.append((idx, name))

    if not candidates:
        return None
    if prefer_name:
        prefer = prefer_name.lower()
        for idx, name in candidates:
            if prefer in name.lower():
                return idx
    return candidates[0][0]


# =========================
# Audio Sources
# =========================
class BaseAudioSource:
    def start(self): raise NotImplementedError
    def read(self, timeout: Optional[float] = None): raise NotImplementedError
    def stop(self): raise NotImplementedError


class SDLoopbackSource(BaseAudioSource):
    """sounddevice-based capture using WASAPI "(loopback)" device"""
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.frame_samples = int(cfg.capture.sample_rate * cfg.capture.block_ms / 1000)
        self.stream = None
        self.queue = queue.Queue(maxsize=50)
        self.running = False

    def _callback(self, indata, frames, time_info, status):
        if status: pass
        data = indata
        # Convert to float32 mono frame of exact length
        if data.dtype != np.float32:
            data = data.astype(np.float32)
        if data.ndim == 2 and data.shape[1] > 1:
            data = data.mean(axis=1, dtype=np.float32)
        elif data.ndim == 2:
            data = data[:, 0]
        data = data.reshape(-1, 1)

        # pad/trim
        if data.shape[0] < self.frame_samples:
            pad = np.zeros((self.frame_samples - data.shape[0], 1), dtype=np.float32)
            data = np.vstack([data, pad])
        elif data.shape[0] > self.frame_samples:
            data = data[:self.frame_samples]

        try:
            self.queue.put_nowait(data.copy())
        except queue.Full:
            pass

    def start(self):
        if self.running: return
        self.running = True

        device = None
        if self.cfg.capture.mode == "device":
            if self.cfg.capture.device_index >= 0:
                device = self.cfg.capture.device_index
            elif self.cfg.capture.device_name:
                for idx, d in enumerate(sd.query_devices()):
                    if self.cfg.capture.device_name.lower() in d['name'].lower():
                        device = idx
                        break
        else:
            # auto: try a WASAPI loopback device
            device = _find_wasapi_loopback_index()

        if is_windows():
            try:
                self.stream = sd.InputStream(
                    samplerate=self.cfg.capture.sample_rate,
                    blocksize=self.frame_samples,
                    dtype=self.cfg.capture.dtype,
                    channels=max(1, self.cfg.capture.channels),
                    device=device,
                    callback=self._callback,
                )
                self.stream.start()
            except Exception as e:
                raise RuntimeError(f"sounddevice loopback failed: {e}")
        else:
            raise RuntimeError("SDLoopbackSource: non-Windows not supported for loopback.")

    def read(self, timeout: Optional[float] = None):
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        self.running = False
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None


class SCLoopbackSource(BaseAudioSource):
    """soundcard-based loopback of default speaker (robust fallback)."""
    def __init__(self, cfg: Config):
        if sc is None:
            raise RuntimeError("soundcard not available")
        self.cfg = cfg
        self.frame_samples = int(cfg.capture.sample_rate * cfg.capture.block_ms / 1000)
        self.queue = queue.Queue(maxsize=50)
        self.running = False
        self.thread = None

    def _runner(self):
        spk = sc.default_speaker()
        mic = sc.get_microphone(spk.name, include_loopback=True)
        bs = self.frame_samples
        with mic.recorder(samplerate=self.cfg.capture.sample_rate, channels=1, blocksize=bs) as rec:
            while self.running:
                try:
                    data = rec.record(numframes=bs)  # float32 [-1,1], (bs,1)
                    if data.dtype != np.float32:
                        data = data.astype(np.float32)
                    if data.shape[1] != 1:
                        data = data.mean(axis=1, keepdims=True, dtype=np.float32)
                    try:
                        self.queue.put_nowait(data.copy())
                    except queue.Full:
                        pass
                except Exception as e:
                    print(f"[soundcard] record error: {e}", file=sys.stderr)
                    time.sleep(0.05)

    def start(self):
        if self.running: return
        self.running = True
        self.thread = threading.Thread(target=self._runner, daemon=True)
        self.thread.start()

    def read(self, timeout: Optional[float] = None):
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
            self.thread = None


class AppLoopbackSource(BaseAudioSource):
    """Capture audio from specific applications via WASAPI loopback"""
    def __init__(self, cfg: Config):
        if AudioUtilities is None:
            raise RuntimeError("pycaw not available")
        self.cfg = cfg
        self.app_names = cfg.capture.apps
        self.frame_samples = int(cfg.capture.sample_rate * cfg.capture.block_ms / 1000)
        self.queue = queue.Queue(maxsize=50)
        self.running = False
        self.thread = None
        self.clients = []

    def _setup_clients(self):
        sessions = AudioUtilities.GetAllSessions()
        for session in sessions:
            proc = session.Process
            if not proc:
                continue
            if proc.name().lower() not in self.app_names:
                continue
            ctl = session._ctl
            client = ctl.QueryInterface(IAudioClient)
            wfx = client.GetMixFormat()
            client.Initialize(
                AUDCLNT_SHAREMODE_SHARED,
                AUDCLNT_STREAMFLAGS_LOOPBACK,
                0,
                0,
                wfx,
                None,
            )
            capture_client = client.GetService(IAudioCaptureClient)
            client.Start()
            self.clients.append((client, capture_client, wfx))

    def _runner(self):
        while self.running:
            for client, cap, wfx in self.clients:
                try:
                    pkt = cap.GetNextPacketSize()
                    while pkt > 0:
                        data_ptr, frames, flags, _, _ = cap.GetBuffer()
                        buff = string_at(data_ptr, frames * wfx.nBlockAlign)
                        arr = np.frombuffer(buff, dtype=np.int16)
                        if wfx.nChannels > 1:
                            arr = arr.reshape(-1, wfx.nChannels).mean(axis=1)
                        arr = arr.astype(np.float32) / 32768.0
                        if wfx.nSamplesPerSec != self.cfg.capture.sample_rate and arr.size > 0:
                            # simple linear resample
                            factor = self.cfg.capture.sample_rate / wfx.nSamplesPerSec
                            idx = np.round(np.arange(0, arr.size * factor) / factor).astype(int)
                            idx = np.clip(idx, 0, arr.size - 1)
                            arr = arr[idx]
                        arr = arr.reshape(-1, 1)
                        if arr.shape[0] < self.frame_samples:
                            pad = np.zeros((self.frame_samples - arr.shape[0], 1), dtype=np.float32)
                            arr = np.vstack([arr, pad])
                        elif arr.shape[0] > self.frame_samples:
                            arr = arr[:self.frame_samples]
                        try:
                            self.queue.put_nowait(arr.copy())
                        except queue.Full:
                            pass
                        cap.ReleaseBuffer(frames)
                        pkt = cap.GetNextPacketSize()
                except Exception:
                    continue
            time.sleep(self.cfg.capture.block_ms / 1000.0)

    def start(self):
        if self.running:
            return
        self._setup_clients()
        if not self.clients:
            raise RuntimeError("No target application sessions found")
        self.running = True
        self.thread = threading.Thread(target=self._runner, daemon=True)
        self.thread.start()

    def read(self, timeout: Optional[float] = None):
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
            self.thread = None
        for client, _, _ in self.clients:
            try:
                client.Stop()
            except Exception:
                pass
        self.clients.clear()


class AudioSourceManager(BaseAudioSource):
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.src: Optional[BaseAudioSource] = None

    def start(self):
        if self.cfg.capture.mode == "app":
            try:
                app_src = AppLoopbackSource(self.cfg)
                app_src.start()
                self.src = app_src
                print("[INFO] Using per-application loopback engine.")
                return
            except Exception as e:
                print(f"[ERROR] app loopback failed: {e}", file=sys.stderr)

        if self.cfg.capture.mode in ("auto", "device"):
            try:
                sd_src = SDLoopbackSource(self.cfg)
                sd_src.start()
                self.src = sd_src
                print("[INFO] Using sounddevice WASAPI loopback engine.")
                return
            except Exception as e:
                print(f"[INFO] sounddevice engine unavailable: {e}", file=sys.stderr)

        if sc is not None:
            try:
                sc_src = SCLoopbackSource(self.cfg)
                sc_src.start()
                self.src = sc_src
                print("[INFO] Using soundcard loopback engine (default speaker).")
                return
            except Exception as e:
                print(f"[ERROR] soundcard loopback failed: {e}", file=sys.stderr)

        raise RuntimeError("No loopback capture method available. Install 'soundcard' or provide a WASAPI '(loopback)' device.")

    def read(self, timeout: Optional[float] = None):
        return self.src.read(timeout) if self.src else None

    def stop(self):
        if self.src:
            self.src.stop()


# =========================
# VAD + Forced segmentation
# =========================
class VADSegmenter:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.vad = webrtcvad.Vad(cfg.vad.aggressiveness)
        self.frame_bytes = int(cfg.capture.sample_rate * cfg.capture.block_ms / 1000) * 2  # int16 mono
        self.silence_frames_needed = math.ceil(cfg.vad.silence_pad_ms / cfg.capture.block_ms)
        self.max_frames = math.ceil(cfg.vad.max_segment_ms / cfg.capture.block_ms)
        self.frames: List[bytes] = []
        self.silence_count = 0
        self.in_speech = False

    def _is_speech(self, frame_int16_mono: np.ndarray) -> bool:
        pcm = frame_int16_mono.tobytes(order='C')
        try:
            return self.vad.is_speech(pcm, sample_rate=self.cfg.capture.sample_rate)
        except Exception:
            # If VAD fails, be permissive
            return True

    def push(self, frame_float32_mono: np.ndarray) -> Optional[bytes]:
        # Convert to int16 mono bytes
        x = np.clip(frame_float32_mono.reshape(-1), -1.0, 1.0)
        int16 = (x * 32767.0).astype(np.int16).reshape(-1, 1)

        speech = self._is_speech(int16)
        self.frames.append(int16.tobytes(order='C'))

        if speech:
            self.in_speech = True
            self.silence_count = 0
        else:
            if self.in_speech:
                self.silence_count += 1

        # End of utterance
        if self.in_speech and self.silence_count >= self.silence_frames_needed:
            segment = b''.join(self.frames)
            self.frames.clear()
            self.silence_count = 0
            self.in_speech = False
            return segment

        # Safety cut (very long)
        if self.in_speech and len(self.frames) >= self.max_frames:
            segment = b''.join(self.frames)
            self.frames.clear()
            self.silence_count = 0
            self.in_speech = False
            return segment

        # Trim buffer outside of speech
        if not self.in_speech and len(self.frames) > self.silence_frames_needed * 3:
            self.frames = self.frames[-self.silence_frames_needed * 3:]
        return None


# =========================
# STT engines
# =========================
class LocalWhisperSTT:
    def __init__(self, cfg: Config):
        if WhisperModel is None:
            raise RuntimeError("faster-whisper not available; set stt.backend=server")
        self.cfg = cfg
        device = "cuda" if self._has_cuda() else "cpu"
        compute = cfg.stt.compute_type
        if compute == "auto":
            compute = "float16" if device == "cuda" else "int8"
        elif device == "cpu" and compute.lower() in ("float16","fp16"):
            compute = "int8"
        print(f"[INFO] STT(local) init: model={cfg.stt.model} device={device} compute_type={compute}")
        self.model = WhisperModel(
            cfg.stt.model,
            device=device,
            compute_type=compute,
        )

    def _has_cuda(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    def transcribe_pcm16_mono(self, pcm_bytes: bytes) -> Dict[str, Any]:
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        segs, info = self.model.transcribe(
            audio=audio,
            beam_size=self.cfg.stt.beam_size,
            vad_filter=self.cfg.stt.vad_filter,
            language=self.cfg.stt.language,
            word_timestamps=False,
            condition_on_previous_text=False,
        )
        text = []
        segments = []
        for s in segs:
            if getattr(s, "text", None):
                t = s.text.strip()
                text.append(t)
                segments.append({
                    "start": float(getattr(s, "start", 0.0) or 0.0),
                    "end": float(getattr(s, "end", 0.0) or 0.0),
                    "text": t,
                })
        return {
            "text": " ".join(text).strip(),
            "segments": segments,
            "language": getattr(info, "language", None),
            "language_probability": float(getattr(info, "language_probability", 0.0) or 0.0),
        }


class RemoteSTT:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.session = requests.Session()
        print(f"[INFO] STT(remote) endpoint: {cfg.stt.server_url}")

    def transcribe_pcm16_mono(self, pcm_bytes: bytes) -> Dict[str, Any]:
        b64 = base64.b64encode(pcm_bytes).decode("ascii")
        payload = {
            "pcm16": b64,
            "sample_rate": self.cfg.capture.sample_rate,
            "language": self.cfg.stt.language,
            "beam_size": self.cfg.stt.beam_size,
            "vad_filter": self.cfg.stt.vad_filter,
        }
        r = self.session.post(self.cfg.stt.server_url, json=payload, timeout=60)
        r.raise_for_status()
        return r.json()


# =========================
# Translator
# =========================
class Translator:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.session = requests.Session()

    def translate(self, english_text: str) -> str:
        if not self.cfg.translate.enable or not english_text.strip():
            return ""
        payload = {
            "model": self.cfg.translate.model,
            "messages": [
                {"role": "system", "content": self.cfg.translate.system_prompt},
                {"role": "user", "content": english_text.strip()},
            ],
            "temperature": self.cfg.translate.temperature,
            "max_tokens": self.cfg.translate.max_tokens,
            "stream": False,
        }
        try:
            resp = self.session.post(self.cfg.translate.url, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"(번역 오류: {e})"


# =========================
# UI
# =========================
class OverlayUI:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.enabled = (tk is not None) and cfg.ui.enable
        self.root = None
        self.chat_box = None
        self.queue = queue.Queue()

        if self.enabled:
            self.root = tk.Tk()
            self.root.title("VSRG Translator Pro")
            self.root.geometry(f"{cfg.ui.width}x{cfg.ui.height}+60+60")
            self.root.configure(bg=cfg.ui.theme_bg)
            self.root.wm_attributes("-topmost", cfg.ui.topmost)

            self.chat_box = scrolledtext.ScrolledText(
                self.root,
                bg=cfg.ui.theme_bg,
                fg=cfg.ui.theme_fg,
                font=(cfg.ui.font_family, cfg.ui.font_size_en),
                wrap="word",
            )
            self.chat_box.pack(fill="both", expand=True, padx=16, pady=(16, 8))
            self.chat_box.configure(state="disabled")

            btn_frame = tk.Frame(self.root, bg=cfg.ui.theme_bg)
            btn_frame.pack(fill="x", padx=12, pady=(8, 12))
            tk.Button(btn_frame, text="Clear", command=self.clear, bg="#2C333A", fg=cfg.ui.theme_fg).pack(side="left")
            tk.Button(btn_frame, text="Quit", command=self.root.destroy, bg="#5A1F1F", fg="#FFFFFF").pack(side="right")

            self.root.after(50, self._poll_queue)

    def _poll_queue(self):
        try:
            while True:
                speaker, en, ko = self.queue.get_nowait()
                if self.chat_box is not None:
                    self.chat_box.configure(state="normal")
                    line = en if speaker is None else f"Speaker {speaker}: {en}"
                    if ko:
                        line += f" / {ko}"
                    self.chat_box.insert(tk.END, line + "\n")
                    self.chat_box.see(tk.END)
                    self.chat_box.configure(state="disabled")
                print(f"[EN] {en}\n[KO] {ko}")
        except queue.Empty:
            pass
        if self.enabled and self.root:
            self.root.after(50, self._poll_queue)

    def push(self, en: str, ko: str, speaker: Optional[int] = None):
        self.queue.put((speaker, en, ko))

    def clear(self):
        if self.chat_box is not None:
            self.chat_box.configure(state="normal")
            self.chat_box.delete("1.0", tk.END)
            self.chat_box.configure(state="disabled")

    def loop(self):
        if self.enabled and self.root:
            self.root.mainloop()
        else:
            try:
                while True:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                pass


# =========================
# Preview STT (live English while speaking)
# =========================
class PreviewSTT:
    def __init__(self, model_name: str, compute_type: str):
        if WhisperModel is None:
            raise RuntimeError("Preview requires faster-whisper installed.")
        device = "cuda"
        try:
            import torch
            if not torch.cuda.is_available():
                device = "cpu"
        except Exception:
            device = "cpu"
        if compute_type == "auto":
            compute_type = "float16" if device == "cuda" else "int8"
        self.model = WhisperModel(model_name, device=device, compute_type=compute_type)

    def transcribe_pcm16_mono(self, pcm_bytes: bytes, language: Optional[str], beam_size: int) -> str:
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        segs, _ = self.model.transcribe(audio=audio, language=language, beam_size=beam_size,
                                        vad_filter=False, word_timestamps=False, condition_on_previous_text=False)
        return " ".join([s.text.strip() for s in segs if getattr(s, "text", None)]).strip()


# =========================
# Orchestration
# =========================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def write_wav(path: str, pcm16: bytes, sr: int):
    audio = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
    sf.write(path, audio, sr)


def run_pipeline(cfg: Config):
    if cfg.debug.list_devices_on_start:
        list_devices()

    audio = AudioSourceManager(cfg)
    audio.start()

    vad = VADSegmenter(cfg)
    if cfg.stt.backend == "local":
        stt = LocalWhisperSTT(cfg)
    else:
        stt = RemoteSTT(cfg)

    trn = Translator(cfg)
    ui  = OverlayUI(cfg)
    diarizer = SpeakerDiarizer()

    # Preview engine (optional)
    preview = None
    if cfg.preview.enable:
        try:
            preview = PreviewSTT(cfg.preview.model, cfg.preview.compute_type)
        except Exception as e:
            print(f"[WARN] preview disabled: {e}", file=sys.stderr)
            preview = None

    # Force segmentation state
    force_enabled = cfg.force.enable
    force_buf: Deque[bytes] = deque()
    force_ms = 0
    frame_ms = cfg.capture.block_ms
    max_buf_frames = max(1, int(cfg.force.max_buffer_ms / frame_ms))

    # Preview rolling window
    preview_buf: Deque[bytes] = deque()
    preview_max_frames = max(1, int(cfg.preview.window_ms / frame_ms))
    last_preview_ts = 0  # ms

    seg_dir = os.path.join(os.getcwd(), "segments")
    if cfg.debug.write_wav_segments:
        ensure_dir(seg_dir)

    def worker():
        nonlocal force_ms, last_preview_ts
        print("[INFO] Running. System output will be captured (mic excluded). Ctrl+C to stop.")
        try:
            while True:
                frame = audio.read(timeout=1.0)
                if frame is None:
                    continue

                # Compute RMS for force/preview logic
                frame_mono = frame.reshape(-1)
                rms_frame = _rms_dbfs(frame_mono)

                # --- Preview feeding (append current frame to rolling window)
                if preview:
                    preview_buf.append(((frame_mono * 32767.0).astype(np.int16)).tobytes())
                    if len(preview_buf) > preview_max_frames:
                        preview_buf.popleft()

                    # while speaking (above gate), run preview periodically
                    now_ms = int(time.time() * 1000)
                    if rms_frame >= cfg.preview.rms_gate_dbfs and (now_ms - last_preview_ts) >= cfg.preview.every_ms:
                        last_preview_ts = now_ms
                        partial = b"".join(preview_buf)
                        try:
                            en_preview = preview.transcribe_pcm16_mono(partial, language=cfg.stt.language, beam_size=1)
                        except Exception:
                            en_preview = ""
                        if en_preview:
                            ui.push(en_preview + " …", "", None)  # preview only English

                # --- VAD path (uses int16 conversion internally)
                segment = vad.push(frame)
                if segment is not None:
                    import numpy as _np
                    dur_ms = int(len(segment)/2 / cfg.capture.sample_rate * 1000)
                    rms = _rms_dbfs(_np.frombuffer(segment, dtype=_np.int16))
                    _log(f"segment ready: {dur_ms} ms, rms={rms:.1f} dBFS")

                    # clear force/preview state
                    force_buf.clear()
                    force_ms = 0
                    preview_buf.clear()
                    last_preview_ts = 0

                    # STT → Translate → UI
                    try:
                        out = stt.transcribe_pcm16_mono(segment)
                        english = out.get("text","").strip()
                    except Exception as e:
                        english = f"(STT 오류: {e})"
                    speaker_id = diarizer.assign_speaker(segment, cfg.capture.sample_rate)
                    korean = trn.translate(english) if english else ""
                    ts = time.strftime("%H:%M:%S")
                    if cfg.debug.log_segments:
                        print(f"\n[{ts}] EN: {english}\n[{ts}] KO: {korean}\n")
                    if cfg.debug.write_wav_segments and english:
                        fname = os.path.join(seg_dir, f"{ts.replace(':','-')}_{dur_ms}ms.wav")
                        write_wav(fname, segment, cfg.capture.sample_rate)
                    ui.push(english, korean, speaker_id)
                    if english:
                        _post_event("stt.text", {
                            "text": english,
                            "translation": korean,
                            "source": "VSRG",
                            "model": cfg.stt.model,
                        })
                    continue

                # --- Forced segmentation bookkeeping (when VAD misses)
                if force_enabled:
                    force_buf.append(((frame_mono * 32767.0).astype(np.int16)).tobytes())
                    if len(force_buf) > max_buf_frames:
                        force_buf.popleft()

                    if rms_frame >= cfg.force.rms_speech_threshold_dbfs:
                        force_ms += frame_ms
                    else:
                        # loud → quiet; if long enough, force cut
                        if force_ms >= cfg.force.min_forced_segment_ms:
                            segment = b''.join(force_buf)
                            _log(f"[force] loud->quiet: {force_ms} ms, buf={int(len(segment)/2/cfg.capture.sample_rate*1000)} ms")
                            force_buf.clear()
                            force_ms = 0
                            preview_buf.clear()
                            last_preview_ts = 0
                            try:
                                out = stt.transcribe_pcm16_mono(segment)
                                english = out.get("text","").strip()
                            except Exception as e:
                                english = f"(STT 오류: {e})"
                            speaker_id = diarizer.assign_speaker(segment, cfg.capture.sample_rate)
                            korean = trn.translate(english) if english else ""
                            ts = time.strftime("%H:%M:%S")
                            if cfg.debug.log_segments:
                                print(f"\n[{ts}] EN: {english}\n[{ts}] KO: {korean}\n")
                            if cfg.debug.write_wav_segments and english:
                                fname = os.path.join(seg_dir, f"{ts.replace(':','-')}_forced.wav")
                                write_wav(fname, segment, cfg.capture.sample_rate)
                            ui.push(english, korean, speaker_id)
                            if english:
                                _post_event("stt.text", {
                                    "text": english,
                                    "translation": korean,
                                    "source": "VSRG/forced",
                                    "model": cfg.stt.model,
                                })
                            continue
                        force_ms = 0

                    # sustained loud without VAD end
                    if force_ms >= cfg.force.sustained_loud_ms:
                        segment = b''.join(force_buf)
                        _log(f"[force] sustained loud: {force_ms} ms, buf={int(len(segment)/2/cfg.capture.sample_rate*1000)} ms")
                        force_buf.clear()
                        force_ms = 0
                        preview_buf.clear()
                        last_preview_ts = 0
                        try:
                            out = stt.transcribe_pcm16_mono(segment)
                            english = out.get("text","").strip()
                        except Exception as e:
                            english = f"(STT 오류: {e})"
                        speaker_id = diarizer.assign_speaker(segment, cfg.capture.sample_rate)
                        korean = trn.translate(english) if english else ""
                        ts = time.strftime("%H:%M:%S")
                        if cfg.debug.log_segments:
                            print(f"\n[{ts}] EN: {english}\n[{ts}] KO: {korean}\n")
                        if cfg.debug.write_wav_segments and english:
                            fname = os.path.join(seg_dir, f"{ts.replace(':','-')}_sustained.wav")
                            write_wav(fname, segment, cfg.capture.sample_rate)
                        ui.push(english, korean, speaker_id)
                        if english:
                            _post_event("stt.text", {
                                "text": english,
                                "translation": korean,
                                "source": "VSRG/sustained",
                                "model": cfg.stt.model,
                            })
        except KeyboardInterrupt:
            print("[INFO] Interrupted.")
        finally:
            audio.stop()

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    ui.loop()


def main():
    parser = argparse.ArgumentParser(description="VSRG Translator Pro (System Output → EN → KO)")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config")
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    cfg = load_config(args.config)
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
