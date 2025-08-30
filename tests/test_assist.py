import sys
import numpy as np
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "assist", Path(__file__).resolve().parents[1] / "STT" / "assist.py"
)
assist = importlib.util.module_from_spec(spec)
sys.modules["assist"] = assist
spec.loader.exec_module(assist)
AssistTranscriber = assist.AssistTranscriber
AssistConfig = assist.AssistConfig


def test_select_device_passthrough():
    assert assist._select_input_device(3, None) == 3


def test_select_device_by_name(monkeypatch):
    class DummySD:
        def __init__(self):
            self.default = type("D", (), {"device": (None, None)})

        def query_devices(self):
            return [
                {"name": "NoMic", "max_input_channels": 0},
                {"name": "Mic B", "max_input_channels": 2},
            ]

    dummy = DummySD()
    monkeypatch.setattr(assist, "sd", dummy)
    assert assist._select_input_device(None, "mic b") == 1


def test_select_device_auto(monkeypatch):
    class DummySD:
        def __init__(self):
            self.default = type("D", (), {"device": (None, None)})

        def query_devices(self):
            return [
                {"name": "NoMic", "max_input_channels": 0},
                {"name": "Mic B", "max_input_channels": 2},
            ]

    dummy = DummySD()
    monkeypatch.setattr(assist, "sd", dummy)
    assert assist._select_input_device(None, None) == 1


class DummyModel:
    def __init__(self):
        self.model_size = "dummy"

    def transcribe(self, audio, language="en"):
        class Seg:
            text = "hello"
        return [Seg()], None


def collect_events():
    events = []

    def _post(evt_type, payload, prio=5):
        events.append((evt_type, payload, prio))

    return events, _post


def test_transcriber_posts_event():
    events, poster = collect_events()
    model = DummyModel()
    cfg = AssistConfig()
    transcriber = AssistTranscriber(model, poster, cfg)
    pcm = (np.ones(cfg.sample_rate, dtype=np.int16)).tobytes()
    text = transcriber.transcribe(pcm)
    assert text == "hello"
    assert events == [
        (
            "stt.text",
            {"text": "hello", "source": "assist", "stt_model": "dummy"},
            5,
        )
    ]


def test_mix_channels_stereo_to_mono():
    left = np.array([0, 1000, -1000], dtype=np.int16)
    right = np.array([1000, -1000, 0], dtype=np.int16)
    interleaved = np.column_stack((left, right)).ravel().tobytes()
    mono = assist._mix_channels(interleaved, 2)
    out = np.frombuffer(mono, dtype=np.int16)
    expected = np.array([(0 + 1000) / 2, (1000 - 1000) / 2, (-1000 + 0) / 2], dtype=np.int16)
    assert np.array_equal(out, expected)


def test_notify_listening(monkeypatch):
    events = []

    def fake_post(evt_type, payload, prio=5):
        events.append((evt_type, payload, prio))

    called = {}

    def fake_toast(title, msg):
        called[(title, msg)] = True
    import types, sys

    monkeypatch.setattr(assist, "_post_event", fake_post)
    pkg = types.ModuleType("agent")
    sinks = types.ModuleType("agent.sinks")
    sinks.toast_notify = fake_toast
    pkg.sinks = sinks
    sys.modules["agent"] = pkg
    sys.modules["agent.sinks"] = sinks
    assist._notify_listening()
    assert events == [("overlay.toast", {"title": "STT", "text": "마이크 청취 중"}, 5)]
    assert ("STT", "마이크 청취 중") in called
