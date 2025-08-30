import asyncio
from types import SimpleNamespace

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import agent.server as server
from agent.event_bus import EventBus
from agent.schemas import Event


class DummyProc:
    def __init__(self):
        self.dead = False

    def poll(self):
        return None if not self.dead else 0

    def terminate(self):
        self.dead = True


def test_stt_stop_ignored(monkeypatch):
    async def runner():
        ctx = SimpleNamespace()
        ctx.bus = EventBus()
        ctx.config = {}
        ctx.assist_mode = True

        app = server.create_app(ctx)
        async with app.router.lifespan_context(app):
            stt_handler = None
            for prefix, h in app.state._plugin_unsubs:
                if prefix == "stt.":
                    stt_handler = h
                    break
            app.state.stt_proc = DummyProc()
            await stt_handler(Event(type="stt.stop", payload={}))
            assert app.state.stt_proc is not None

    try:
        asyncio.run(runner())
    except asyncio.CancelledError:
        pass


def test_ticker_restarts_stt(monkeypatch):
    async def runner():
        ctx = SimpleNamespace()
        ctx.bus = EventBus()
        ctx.config = {}
        ctx.assist_mode = False

        spawns = []

        def dummy_spawn(cfg, key):
            spawns.append(key)
            return DummyProc()

        monkeypatch.setattr(server, "_spawn_tool", dummy_spawn)

        orig_sleep = asyncio.sleep

        async def fast_sleep(_):
            await orig_sleep(0)

        monkeypatch.setattr(server.asyncio, "sleep", fast_sleep)

        app = server.create_app(ctx)
        async with app.router.lifespan_context(app):
            assist_handler = None
            for prefix, h in app.state._plugin_unsubs:
                if prefix == "assist.":
                    assist_handler = h
                    break
            await assist_handler(Event(type="assist.on", payload={}))
            app.state.stt_proc.dead = True
            await asyncio.sleep(0.01)
            assert spawns.count("stt_assist.start") >= 2
            await assist_handler(Event(type="assist.off", payload={}))

    try:
        asyncio.run(runner())
    except asyncio.CancelledError:
        pass

