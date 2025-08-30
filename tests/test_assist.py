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
    assert assist._select_input_device(3) == 3


def test_select_device_auto(monkeypatch):
    class DummySD:
        def __init__(self):
            self.default = type("D", (), {"device": (None, None)})

        def query_devices(self):
            return [
                {"max_input_channels": 0},
                {"max_input_channels": 2},
            ]

    dummy = DummySD()
    monkeypatch.setattr(assist, "sd", dummy)
    assert assist._select_input_device(None) == 1


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
