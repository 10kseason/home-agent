import os, sys

sys.path.append(os.path.abspath('.'))
import src.core.executor as executor


class Elem:
    def __init__(self, caps):
        self.capabilities = caps
        self.invoked = False
        self.value = None
        self.selected = None

    def invoke(self):
        self.invoked = True
        return "invoked"

    def set_value(self, v):
        self.value = v
        return v

    def select(self, t):
        self.selected = t
        return t


def test_invoke_prefers_uia(monkeypatch):
    e = Elem({"invoke"})
    monkeypatch.setattr(executor, "resolve_uia", lambda tid: e)
    res = executor.execute({"action": "invoke", "target": {"id": "1"}, "params": {}})
    assert res == "invoked" and e.invoked


def test_set_value_fallback(monkeypatch):
    monkeypatch.setattr(executor, "resolve_uia", lambda tid: None)
    monkeypatch.setattr(executor, "type_fallback", lambda v: f"typed:{v}")
    res = executor.execute({
        "action": "set_value",
        "target": {"id": "1"},
        "params": {"value": "hi"},
    })
    assert res == "typed:hi"


def test_danger_cancel(monkeypatch):
    monkeypatch.setattr(executor, "is_dangerous", lambda d: True)
    monkeypatch.setattr(executor, "confirm_with_wake", lambda: False)
    res = executor.execute({"action": "invoke", "target": {"id": "1"}, "params": {}})
    assert res == "cancelled"
