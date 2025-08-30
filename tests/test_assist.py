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
