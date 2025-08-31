import asyncio
from types import SimpleNamespace
import sys, pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from agent.event_bus import EventBus
from agent.plugins.assist_cmd_plugin import AssistCommandPlugin
from agent.schemas import Event

async def dispatch_all(bus: EventBus):
    while not bus.queue.empty():
        _, _, e = await bus.queue.get()
        for prefix, handlers in bus.subscribers.items():
            if e.type.startswith(prefix):
                for h in handlers:
                    await h(e)


def test_cmd_detected_toast():
    async def runner():
        ctx = SimpleNamespace()
        ctx.bus = EventBus(dedup_window=0)
        ctx.config = {}
        plugin = AssistCommandPlugin(ctx)
        toasts = []

        async def collect(ev):
            toasts.append((ev.payload.get("title"), ev.payload.get("text")))

        ctx.bus.subscribe("overlay.", collect)
        await plugin.handle(Event(type="cmd.detected", payload={"cmd": "capture"}))
        await dispatch_all(ctx.bus)
        assert toasts[0] == ("📋 Cmd Detected", "capture")

    asyncio.run(runner())


def test_capture_repeat():
    async def runner():
        ctx = SimpleNamespace()
        ctx.bus = EventBus(dedup_window=0)
        ctx.config = {}
        plugin = AssistCommandPlugin(ctx)
        events = []
        async def collect(ev):
            events.append(ev.type)
        async def noop(ev):
            pass
        ctx.bus.subscribe("ocr_assist.", collect)
        ctx.bus.subscribe("overlay.", noop)
        await plugin.handle(Event(type="cmd.detected", payload={"cmd": "capture"}))
        await dispatch_all(ctx.bus)
        await plugin.handle(Event(type="cmd.detected", payload={"cmd": "repeat"}))
        await dispatch_all(ctx.bus)
        assert events == ["ocr_assist.start", "ocr_assist.start"]
    asyncio.run(runner())


def test_summarize_then_translate(monkeypatch):
    async def runner():
        ctx = SimpleNamespace()
        ctx.bus = EventBus(dedup_window=0)
        ctx.config = {
            "llm_summary": {"model": "sum"},
            "translate": {"model": "trans"},
        }
        plugin = AssistCommandPlugin(ctx)
        plugin.last_ocr = "original"

        async def fake_sum(text):
            assert text == "original"
            return "summary"
        inputs = []
        async def fake_trans(text):
            inputs.append(text)
            return "translated"
        plugin._summarize = fake_sum
        plugin._translate = fake_trans

        events = []
        async def collect(ev):
            events.append(ev.type)
        async def noop(ev):
            pass
        ctx.bus.subscribe("llm.", collect)
        ctx.bus.subscribe("overlay.", noop)

        await plugin.handle(Event(type="cmd.detected", payload={"cmd": "summarize"}))
        await dispatch_all(ctx.bus)
        await plugin.handle(Event(type="cmd.detected", payload={"cmd": "translate"}))
        await dispatch_all(ctx.bus)

        assert events == ["llm.summary", "llm.translation"]
        assert inputs == ["summary"]
    asyncio.run(runner())


def test_focus_command():
    async def runner():
        ctx = SimpleNamespace()
        ctx.bus = EventBus(dedup_window=0)
        ctx.config = {}
        plugin = AssistCommandPlugin(ctx)
        states = []
        async def collect(ev):
            states.append(ev.payload.get("on"))
        async def noop(ev):
            pass
        ctx.bus.subscribe("FocusMode.", collect)
        ctx.bus.subscribe("overlay.", noop)
        await plugin.handle(Event(type="cmd.detected", payload={"cmd": "focus"}))
        await dispatch_all(ctx.bus)
        assert states == [True]
    asyncio.run(runner())
