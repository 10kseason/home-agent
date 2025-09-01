import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import asyncio
from types import SimpleNamespace

from agent.event_bus import EventBus
from agent.server import create_app
from agent.schemas import Event


async def dispatch_all(bus: EventBus):
    while not bus.queue.empty():
        _, _, e = await bus.queue.get()
        for prefix, handlers in bus.subscribers.items():
            if e.type.startswith(prefix):
                for h in handlers:
                    await h(e)


async def _run(ctx, subscribe_prefix):
    app = create_app(ctx, plugins=[])
    events = []
    try:
        async with app.router.lifespan_context(app):
            async def collect(ev):
                events.append(ev.type)
            ctx.bus.subscribe(subscribe_prefix, collect)
            handler = ctx.bus.subscribers["cmd.detected"][0]
            await handler(Event(type="cmd.detected", payload={"cmd": "capture"}))
            await dispatch_all(ctx.bus)
    except asyncio.CancelledError:
        pass
    return events


def test_cmd_detected_to_ocr():
    ctx = SimpleNamespace(config={}, bus=EventBus(dedup_window=0), assist_mode=False)
    events = asyncio.run(_run(ctx, "ocr."))
    assert events == ["ocr.start"]

def test_cmd_detected_to_capture_assist():
    ctx = SimpleNamespace(config={}, bus=EventBus(dedup_window=0), assist_mode=True)
    events = asyncio.run(_run(ctx, "capture_assist."))
    assert events == ["capture_assist.start"]

