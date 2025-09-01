import asyncio
from types import SimpleNamespace

from agent import tool_gateway
from agent.tool_gateway import handle_tool_calls


class DummyBus:
    def __init__(self):
        self.events = []

    async def publish(self, e):
        self.events.append(e)


def test_tool_gateway_routes_events():
    bus = DummyBus()
    ctx = SimpleNamespace(assist_mode=False)
    tool_calls = [
        {"function": {"name": "assist_on", "arguments": "{}"}},
        {"idempotency_key": "abc", "function": {"name": "stt_assist_start", "arguments": "{}"}},
        {"function": {"name": "ocr_capture", "arguments": "{\"region\": \"full\"}"}},
        {"function": {"name": "assist_off", "arguments": "{}"}},
    ]
    asyncio.run(handle_tool_calls(tool_calls, ctx, bus))
    types = [e.type for e in bus.events]
    assert types == [
        "assist.on",
        "assist.on",
        "stt_assist.start",
        "ocr_assist.start",
        "assist.off",
        "stt.stop",
        "ocr.stop",
    ]
    corr = bus.events[1].payload["correlation_id"]
    assert corr == bus.events[2].payload["correlation_id"]
    assert bus.events[1].payload["idempotency_key"] == "abc"


def test_tool_gateway_basic_stt_ocr():
    bus = DummyBus()
    ctx = SimpleNamespace(assist_mode=False)
    tool_calls = [
        {"function": {"name": "stt_start", "arguments": "{}"}},
        {"function": {"name": "ocr_start", "arguments": "{}"}},
        {"function": {"name": "stt_stop", "arguments": "{}"}},
        {"function": {"name": "ocr_stop", "arguments": "{}"}},
    ]
    asyncio.run(handle_tool_calls(tool_calls, ctx, bus))
    assert [e.type for e in bus.events] == [
        "stt.start",
        "ocr.start",
        "stt.stop",
        "ocr.stop",
    ]


def test_tool_schemas_loaded():
    names = {t["function"]["name"] for t in tool_gateway.TOOLS}
    assert {
        "stt_assist_start",
        "stt_assist_stop",
        "ocr_capture",
        "ocr_assist_start",
        "ocr_assist_stop",
        "assist_on",
        "assist_off",
        "stt_start",
        "stt_stop",
        "ocr_start",
        "ocr_stop",
    }.issubset(names)
    assert tool_gateway.SCHEMAS["ocr_capture"]["properties"]["region"]["type"] == "string"


def test_ocr_capture_always_assist():
    bus = DummyBus()
    ctx = SimpleNamespace(assist_mode=False)
    tool_calls = [{"function": {"name": "ocr_capture", "arguments": "{}"}}]
    asyncio.run(handle_tool_calls(tool_calls, ctx, bus))
    assert [e.type for e in bus.events] == ["ocr_assist.start"]


def test_stt_ocr_start_assist_mode():
    bus = DummyBus()
    ctx = SimpleNamespace(assist_mode=True)
    tool_calls = [
        {"function": {"name": "stt_start", "arguments": "{}"}},
        {"function": {"name": "ocr_start", "arguments": "{}"}},
    ]
    asyncio.run(handle_tool_calls(tool_calls, ctx, bus))
    assert [e.type for e in bus.events] == [
        "stt_assist.start",
        "ocr_assist.start",
    ]


def test_assist_on_routes():
    bus = DummyBus()
    ctx = SimpleNamespace(assist_mode=False)
    tool_calls = [{"function": {"name": "assist_on", "arguments": "{}"}}]
    asyncio.run(handle_tool_calls(tool_calls, ctx, bus))
    assert [e.type for e in bus.events] == ["assist.on"]
