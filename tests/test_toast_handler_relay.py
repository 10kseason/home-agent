import asyncio
from types import SimpleNamespace

import agent.server as server
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


def test_toast_handler_relays_to_bus():
    ctx = SimpleNamespace(config={}, bus=EventBus(dedup_window=0), assist_mode=False)
    app = create_app(ctx, plugins=[])
    relayed = []

    async def capture(ev):
        relayed.append(ev.payload)

    ctx.bus.subscribe("overlay.toast", capture)

    async def _run():
        async with app.router.lifespan_context(app):
            await ctx.bus.publish(Event(type="overlay.toast", payload={"title": "hi", "text": "there"}))
            await dispatch_all(ctx.bus)

    asyncio.run(_run())

    assert any(p.get("_relay") for p in relayed)
