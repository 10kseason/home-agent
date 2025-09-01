import asyncio
from types import SimpleNamespace
from fastapi.testclient import TestClient

from agent.server import create_app

class DummyBus:
    def __init__(self):
        self.events = []
    async def publish(self, e):
        self.events.append(e)
    def subscribe(self, prefix, handler):
        pass
    async def run(self):
        pass

def _make_app(assist_mode=False):
    bus = DummyBus()
    ctx = SimpleNamespace(bus=bus, config={}, assist_mode=assist_mode)
    app = create_app(ctx, plugins=[])
    return app, bus

def test_tool_endpoint_legacy():
    app, bus = _make_app()
    with TestClient(app) as client:
        r = client.post("/tool/call", json={"name": "ocr.start", "args": {}})
        assert r.status_code == 200
    assert [e.type for e in bus.events] == ["ocr.start"]

def test_llm_response_tool_calls():
    app, bus = _make_app()
    payload = {"tool_calls": [{"function": {"name": "stt_start", "arguments": "{}"}}]}
    with TestClient(app) as client:
        r = client.post("/llm/response", json=payload)
        assert r.status_code == 200
    assert [e.type for e in bus.events] == ["stt.start"]
