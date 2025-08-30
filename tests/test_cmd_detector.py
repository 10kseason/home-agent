from STT import cmd_detector
from STT.assist import AssistConfig


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
