import importlib.util
from pathlib import Path
from PIL import Image
import numpy as np
import sys

spec = importlib.util.spec_from_file_location(
    "capture_assist", Path(__file__).resolve().parents[1] / "Capture-assist" / "capture_assist.py"
)
capture_assist = importlib.util.module_from_spec(spec)
sys.modules["capture_assist"] = capture_assist
spec.loader.exec_module(capture_assist)
OCRAssistConfig = capture_assist.OCRAssistConfig
load_config = capture_assist.load_config


def test_load_config_values(tmp_path):
    cfg_file = tmp_path / "Assist-config.yaml"
    cfg_file.write_text(
        """
assist:
  announce_text: done
capture:
  monitor: 2
  region: [10, 20, 30, 40]
ocr:
  langs: [en]
  gpu: true
""",
        encoding="utf-8",
    )
    cfg = load_config(str(cfg_file))
    assert cfg.monitor == 2
    assert cfg.region == [10, 20, 30, 40]
    assert cfg.langs == ["en"]
    assert cfg.gpu is True
    assert cfg.announce_text == "done"


def test_run_ocr_uses_config(monkeypatch):
    calls = {}

    class DummyReader:
        def __init__(self, langs, gpu=False):
            calls["langs"] = langs
            calls["gpu"] = gpu

        def readtext(self, img, detail=0):
            return ["ok"]

    monkeypatch.setattr(capture_assist, "easyocr", type("M", (), {"Reader": DummyReader}))
    img = Image.fromarray(np.zeros((1, 1, 3), dtype=np.uint8))
    cfg = OCRAssistConfig(langs=["en"], gpu=False)
    text = capture_assist._run_ocr(img, cfg)
    assert calls["langs"] == ["en"]
    assert calls["gpu"] is False
    assert text == "ok"


def test_refine_with_jan_supports_lmstudio(monkeypatch):
    called = {}

    class DummyResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": "refined"}}]}

    def fake_post(url, headers=None, json=None, timeout=None):
        called["url"] = url
        return DummyResp()

    monkeypatch.setenv("LM_STUDIO_ENDPOINT", "http://lmstudio:1234/v1")
    monkeypatch.setattr(capture_assist.requests, "post", fake_post)
    out = capture_assist._refine_with_jan("hi")
    assert called["url"] == "http://lmstudio:1234/v1/chat/completions"
    assert out == "refined"


def test_main_announces_delay_and_progress(monkeypatch):
    messages = []
    monkeypatch.setattr(capture_assist, "_notify", lambda msg: messages.append(msg))
    monkeypatch.setattr(capture_assist, "_capture_screen", lambda cfg: Image.new("RGB", (1, 1)))
    monkeypatch.setattr(capture_assist, "_run_ocr", lambda img, cfg: "")
    sleeps = []
    monkeypatch.setattr(capture_assist.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(sys, "argv", ["capture_assist.py"])
    capture_assist.main()
    assert messages[0].startswith("5초")
    assert messages[1] == "이미지를 OCR 중입니다.."
    assert sleeps == [5]


def test_main_overlay_and_toast_on_text(monkeypatch):
    messages = []
    overlays = []
    monkeypatch.setattr(capture_assist, "_notify", lambda msg: messages.append(msg))
    monkeypatch.setattr(
        capture_assist, "_overlay_toast", lambda msg, title="Assist-Capture": overlays.append((title, msg))
    )
    monkeypatch.setattr(capture_assist, "_post_event", lambda t, p, _prio=5: None)
    monkeypatch.setattr(capture_assist, "_capture_screen", lambda cfg: Image.new("RGB", (1, 1)))
    monkeypatch.setattr(capture_assist, "_run_ocr", lambda img, cfg: "hello")
    monkeypatch.setattr(capture_assist.time, "sleep", lambda s: None)
    monkeypatch.setattr(sys, "argv", ["capture_assist.py"])
    monkeypatch.setattr(capture_assist, "speak", lambda text, lang="ko": None)
    capture_assist.main()
    assert ("Assist-Capture", "[Capture-assist] hello") in overlays
    assert messages[-1] == "EasyOCR로 OCR했어요. Overlay 확인 해주세요."


def test_main_cleans_log(monkeypatch, tmp_path):
    log_file = tmp_path / "capture_assist.log"
    log_file.write_text("old", encoding="utf-8")
    monkeypatch.setattr(capture_assist, "LOG_PATH", log_file)
    monkeypatch.setattr(capture_assist, "_notify", lambda msg: None)
    monkeypatch.setattr(capture_assist, "_overlay_toast", lambda msg, title="Assist-Capture": None)
    monkeypatch.setattr(capture_assist, "_post_event", lambda t, p, _prio=5: None)
    monkeypatch.setattr(capture_assist, "_capture_screen", lambda cfg: Image.new("RGB", (1, 1)))
    monkeypatch.setattr(capture_assist, "_run_ocr", lambda img, cfg: "hello")
    monkeypatch.setattr(capture_assist.time, "sleep", lambda s: None)
    monkeypatch.setattr(sys, "argv", ["capture_assist.py"])
    monkeypatch.setattr(capture_assist, "speak", lambda text, lang="ko": None)
    capture_assist.main()
    assert not log_file.exists()
