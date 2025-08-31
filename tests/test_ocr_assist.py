import importlib.util
from pathlib import Path
from PIL import Image
import numpy as np
import sys

spec = importlib.util.spec_from_file_location(
    "ocr_assist", Path(__file__).resolve().parents[1] / "OCR" / "OCR-Assist.py"
)
ocr_assist = importlib.util.module_from_spec(spec)
sys.modules["ocr_assist"] = ocr_assist
spec.loader.exec_module(ocr_assist)
OCRAssistConfig = ocr_assist.OCRAssistConfig
load_config = ocr_assist.load_config


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

    monkeypatch.setattr(ocr_assist, "easyocr", type("M", (), {"Reader": DummyReader}))
    img = Image.fromarray(np.zeros((1, 1, 3), dtype=np.uint8))
    cfg = OCRAssistConfig(langs=["en"], gpu=False)
    text = ocr_assist._run_ocr(img, cfg)
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
    monkeypatch.setattr(ocr_assist.requests, "post", fake_post)
    out = ocr_assist._refine_with_jan("hi")
    assert called["url"] == "http://lmstudio:1234/v1/chat/completions"
    assert out == "refined"
