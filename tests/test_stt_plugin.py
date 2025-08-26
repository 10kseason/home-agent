import asyncio
from types import SimpleNamespace
from agent.event_bus import EventBus
from agent.plugins.stt_plugin import STTPlugin
from agent.schemas import Event


def test_stt_plugin_does_not_republish():
    async def runner():
        ctx = SimpleNamespace()
        ctx.bus = EventBus()
        stt = STTPlugin(ctx)

        calls = []

        async def dummy_translator(ev):
            calls.append(ev.payload.get("text"))

        ctx.bus.subscribe("stt.text", stt.handle)
        ctx.bus.subscribe("stt.text", dummy_translator)

        ev = Event(type="stt.text", payload={"text": "hi hi"}, priority=5)
        await ctx.bus.publish(ev)

        async def dispatch_once(bus):
            _, _, e = await bus.queue.get()
            for prefix, handlers in bus.subscribers.items():
                if e.type.startswith(prefix):
                    for h in handlers:
                        await h(e)

        while not ctx.bus.queue.empty():
            await dispatch_once(ctx.bus)

        assert calls == ["hi"]

    asyncio.run(runner())
