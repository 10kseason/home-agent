import types
import sys
import os
import asyncio
import pytest

# Ensure project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Stub PyQt5 modules for headless testing
class _QtDummyBase:
    def __init__(self, *args, **kwargs):
        pass


class _QtDummyModule(types.ModuleType):
    def __getattr__(self, name):  # returns subclassable dummy classes
        if name == "pyqtSignal":
            return lambda *a, **k: (lambda *a, **k: None)
        if name == "pyqtSlot":
            return lambda *a, **k: (lambda func: func)
        return _QtDummyBase


pyqt = types.ModuleType("PyQt5")
sys.modules.setdefault("PyQt5", pyqt)
sys.modules.setdefault("PyQt5.QtCore", _QtDummyModule("QtCore"))
sys.modules.setdefault("PyQt5.QtGui", _QtDummyModule("QtGui"))
sys.modules.setdefault("PyQt5.QtWidgets", _QtDummyModule("QtWidgets"))

import Overlay.overlay_app as app
from Overlay.overlay_app import Orchestrator


def test_heuristic_route_only_websearch():
    orch = Orchestrator({}, window=None)
    # STT/OCR should not be handled by heuristics anymore
    assert orch._heuristic_route("자막 켜줘") == []
    assert orch._heuristic_route("STT 켜줘") == []
    # Web search remains keyword-based
    res = orch._heuristic_route("고양이 검색")
    assert res == [{"name": "web.search", "args": {"q": "고양이 검색"}}]


def test_call_tools_prefers_llm(monkeypatch):
    # Simplify security hooks
    monkeypatch.setattr(app, "pre_ingest_external", lambda text, meta: {"safe_summary": text})
    monkeypatch.setattr(app, "before_upload", lambda text, is_file=False: {"masked_text": text})
    orch = Orchestrator({}, window=None)

    async def fake_chat(llm, msgs):
        # Simulate LLM suggesting OCR start via function call
        return {"content": "", "tool_calls": [{"type": "function", "function": {"name": "ocr.start", "arguments": "{}"}}]}

    monkeypatch.setattr(orch, "_chat_complete_raw", fake_chat)
    out = asyncio.run(orch.call_tools("화면의 글자를 읽어 줘"))
    assert {"name": "ocr.start", "args": {}} in out["tool_calls"]
