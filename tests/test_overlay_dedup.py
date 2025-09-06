from types import SimpleNamespace

from Overlay.overlay_app import EventHandler


class DummyEmitter:
    def __init__(self):
        self.calls = []

    def emit(self, who, msg):
        self.calls.append((who, msg))


def test_overlay_toast_dedup():
    window = SimpleNamespace(appended=DummyEmitter())
    handler = EventHandler(window)

    handler._handle_overlay_event("overlay.toast", {"title": "T", "text": "hello"})
    handler._handle_overlay_event("overlay.toast", {"title": "T", "text": "hello"})

    assert len(window.appended.calls) == 1

    handler._handle_overlay_event("overlay.toast", {"title": "T", "text": "bye"})
    assert len(window.appended.calls) == 2
