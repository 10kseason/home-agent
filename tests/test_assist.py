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
load_config = assist.load_config


def test_select_device_passthrough():
    assert assist._select_input_device(3, None) == 3


def test_load_config_default():
    cfg = load_config()
    assert cfg.model == "base"
    assert "capture" in cfg.commands


def test_load_config_override(tmp_path):
    cfg_file = tmp_path / "Assist-config.yaml"
    cfg_file.write_text(
        """
stt:
  model: tiny
assist:
  commands:
    hello: [hi]
""",
        encoding="utf-8",
    )
    cfg = load_config(str(cfg_file))
    assert cfg.model == "tiny"
    assert cfg.commands["hello"] == ["hi"]


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
    pcm = (np.full(cfg.sample_rate, 5000, dtype=np.int16)).tobytes()
    text = transcriber.transcribe(pcm)
    assert text == "hello"
    assert events == [
        (
            "stt.text",
            {"text": "hello", "source": "assist", "stt_model": "dummy", "assist": True},
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
    assert events == [("overlay.toast", {"title": "Assist-STT", "text": "마이크 청취 중"}, 5)]
    assert ("Assist-STT", "마이크 청취 중") in called


def test_transcriber_skips_silence():
    events, poster = collect_events()
    model = DummyModel()
    cfg = AssistConfig()
    transcriber = AssistTranscriber(model, poster, cfg)
    pcm = (np.zeros(cfg.sample_rate, dtype=np.int16)).tobytes()
    text = transcriber.transcribe(pcm)
    assert text == ""
    assert events == []


class YouModel:
    def __init__(self):
        self.model_size = "dummy"

    def transcribe(self, audio, language="en"):
        class Seg:
            text = "You"
        return [Seg()], None


def test_transcriber_filters_you():
    events, poster = collect_events()
    model = YouModel()
    cfg = AssistConfig()
    transcriber = AssistTranscriber(model, poster, cfg)
    pcm = (np.full(cfg.sample_rate, 5000, dtype=np.int16)).tobytes()
    text = transcriber.transcribe(pcm)
    assert text == ""
    assert events == []


def test_resample_pcm16_length():
    pcm = np.arange(8000, dtype=np.int16).tobytes()
    out = assist._resample_pcm16(pcm, 8000, 16_000)
    assert len(out) == 16_000 * 2


class LenModel:
    def __init__(self):
        self.model_size = "dummy"
        self.last_len = None

    def transcribe(self, audio, language="en"):
        self.last_len = len(audio)
        class Seg:
            text = "hi"
        return [Seg()], None


def test_transcriber_resamples(monkeypatch):
    model = LenModel()
    cfg = AssistConfig()
    transcriber = AssistTranscriber(model, lambda *args: None, cfg)
    pcm = (np.full(8000, 5000, dtype=np.int16)).tobytes()
    text = transcriber.transcribe(pcm, sample_rate=8000)
    assert text == "hi"
    assert model.last_len == cfg.sample_rate


class AssistModel:
    def __init__(self):
        self.model_size = "dummy"

    def transcribe(self, audio, language="en"):
        class Seg:
            text = "assist tell me a joke"

        return [Seg()], None


def test_llm_called_on_assist_command(monkeypatch):
    events, poster = collect_events()
    cfg = AssistConfig()
    model = AssistModel()
    transcriber = AssistTranscriber(model, poster, cfg)

    recorded = {}

    def fake_llm(prompt):
        recorded["prompt"] = prompt
        return "why did the chicken?"

    monkeypatch.setattr(transcriber, "_call_llm", fake_llm)
    pcm = (np.full(cfg.sample_rate, 5000, dtype=np.int16)).tobytes()
    transcriber.transcribe(pcm)
    assert recorded["prompt"] == "tell me a joke"
    assert events[0][0] == "stt.text"
    assert events[1][0] == "cmd.detected" and events[1][1]["cmd"] == "assist"
    assert events[2] == (
        "llm.chat",
        {"text": "why did the chicken?", "model": ""},
        5,
    )
