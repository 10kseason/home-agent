import asyncio
from types import SimpleNamespace

from agent.plugins.overlay_send_plugin import OverlaySendPlugin
from agent.schemas import Event


def test_overlay_send_plugin_republishes(monkeypatch):
    published = []

    class DummyBus:
        async def publish(self, e):
            published.append(e)

    ctx = SimpleNamespace(bus=DummyBus())
    plugin = OverlaySendPlugin(ctx)

    asyncio.run(plugin.handle(Event(type="overlay.send", payload={"title": "T", "text": "hi"})))

    assert len(published) == 1
    e = published[0]
    assert e.type == "overlay.toast"
    assert e.payload["title"] == "T"
    assert e.payload["text"] == "hi"
    assert e.payload["_relay"] is True
