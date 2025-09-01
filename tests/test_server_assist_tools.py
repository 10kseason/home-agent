import asyncio
from types import SimpleNamespace

import agent.server as server
from agent.event_bus import EventBus
from agent.schemas import Event


class DummyProc:
    def poll(self):
        return None

    def terminate(self):
        pass


def test_assist_tool_events(monkeypatch):
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

        app = server.create_app(ctx)
        async with app.router.lifespan_context(app):
              stt_handler = None
              ocr_handler = None
              for prefix, h in app.state._plugin_unsubs:
                  if prefix == "mictrans.":
                      stt_handler = h
                  elif prefix == "capture_assist.":
                      ocr_handler = h
              assert stt_handler is not None
              assert ocr_handler is not None

              await stt_handler(Event(type="mictrans.start", payload={}))
              await ocr_handler(Event(type="capture_assist.start", payload={}))

        assert spawns == ["mictrans.start", "capture_assist.start"]

    try:
        asyncio.run(runner())
    except asyncio.CancelledError:
        pass

