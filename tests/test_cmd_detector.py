import sys, importlib.util
from pathlib import Path

from STT import cmd_detector

spec = importlib.util.spec_from_file_location(
    "mictrans", Path(__file__).resolve().parents[1] / "Mic-trans-assist" / "mictrans.py"
)
mictrans = importlib.util.module_from_spec(spec)
sys.modules["mictrans"] = mictrans
spec.loader.exec_module(mictrans)
AssistConfig = mictrans.AssistConfig


def test_detect_command_basic():
    cmd_detector._reset_state()
    cfg = AssistConfig()
    assert cmd_detector.detect_command("지금 캡쳐 해줘", cfg) == "capture"


def test_detect_command_cooldown(monkeypatch):
    cmd_detector._reset_state()
    cfg = AssistConfig()
    t = [0.0]
    monkeypatch.setattr(cmd_detector.time, "time", lambda: t[0])
    assert cmd_detector.detect_command("캡쳐", cfg) == "capture"
    assert cmd_detector.detect_command("캡쳐", cfg) is None
    t[0] += (cfg.detection["cooldown_ms"] / 1000) + 0.01
    assert cmd_detector.detect_command("캡쳐", cfg) == "capture"
