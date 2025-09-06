import asyncio
from types import SimpleNamespace
from loguru import logger

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


def test_overlay_toast_logged():
    ctx = SimpleNamespace(config={}, bus=EventBus(dedup_window=0), assist_mode=False)
    app = create_app(ctx, plugins=[])
    messages = []
    token = logger.add(lambda m: messages.append(m), level="INFO")
    try:
        async def _run():
            async with app.router.lifespan_context(app):
                await ctx.bus.publish(Event(type="overlay.toast", payload={"title": "hi", "text": "there"}))
                await dispatch_all(ctx.bus)
        asyncio.run(_run())
    finally:
        logger.remove(token)
    assert any("[toast]" in str(m) for m in messages)
