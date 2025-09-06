import asyncio
from types import SimpleNamespace

import agent.server as server
from agent.event_bus import EventBus
from agent.schemas import Event


class DummyProc:
    def poll(self):
        return None


def test_mictrans_restart(monkeypatch):
    async def runner():
        ctx = SimpleNamespace(bus=EventBus(), config={}, assist_mode=False)
        spawns = []
        terminations = []

        def dummy_spawn(cfg, key):
            spawns.append(key)
            return DummyProc()

        async def dummy_terminate(proc, name="mictrans", timeout=3.0):
            terminations.append(name)

        monkeypatch.setattr(server, "_spawn_tool", dummy_spawn)
        monkeypatch.setattr(server, "_terminate_proc", dummy_terminate)

        app = server.create_app(ctx)
        async with app.router.lifespan_context(app):
            handler = None
            for prefix, h in app.state._plugin_unsubs:
                if prefix == "mictrans.":
                    handler = h
                    break
            await handler(Event(type="mictrans.start", payload={}))
            await handler(Event(type="mictrans.start", payload={}))

        assert spawns == ["mictrans.start", "mictrans.start"]
        assert terminations == ["mictrans"]

    try:
        asyncio.run(runner())
    except asyncio.CancelledError:
        pass
