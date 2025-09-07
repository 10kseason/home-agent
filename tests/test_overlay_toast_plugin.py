import asyncio
from types import SimpleNamespace

from agent.plugins.overlay_toast_plugin import OverlayToastPlugin
from agent.schemas import Event


def test_overlay_toast_plugin_emits_toast(monkeypatch):
    ctx = SimpleNamespace()
    plugin = OverlayToastPlugin(ctx)
    shown = []

    def fake_win10(title, msg):
        shown.append((title, msg))
        return True

    monkeypatch.setattr("agent.plugins.overlay_toast_plugin._toast_win10", fake_win10)
    monkeypatch.setattr("agent.plugins.overlay_toast_plugin._toast_winotify", lambda t, m: False)

    # non-relayed event should not trigger toast
    asyncio.run(plugin.handle(Event(type="overlay.toast", payload={"title": "T", "text": "one"})))
    assert shown == []

    # relayed event should trigger toast once
    asyncio.run(plugin.handle(Event(type="overlay.toast", payload={"title": "T", "text": "two", "_relay": True})))
    assert shown == [("T", "two")]
