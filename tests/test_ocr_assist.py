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
  lang: en
  use_angle_cls: false
  device: gpu
""",
        encoding="utf-8",
    )
    cfg = load_config(str(cfg_file))
    assert cfg.monitor == 2
    assert cfg.region == [10, 20, 30, 40]
    assert cfg.lang == "en"
    assert cfg.use_angle_cls is False
    assert cfg.device == "gpu"
    assert cfg.announce_text == "done"


def test_run_ocr_uses_config(monkeypatch):
    calls = {}

    class DummyOCR:
        def __init__(self, **kwargs):
            calls.update(kwargs)

        def ocr(self, img, cls=True):
            return [[(None, ("ok", 0.9))]]

    monkeypatch.setattr(ocr_assist, "PaddleOCR", DummyOCR)
    img = Image.fromarray(np.zeros((1, 1, 3), dtype=np.uint8))
    cfg = OCRAssistConfig(lang="en", use_angle_cls=False, device="cpu")
    text = ocr_assist._run_ocr(img, cfg)
    assert calls["lang"] == "en"
    assert calls["use_angle_cls"] is False
    assert text == "ok"
