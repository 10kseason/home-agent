from Overlay.overlay_app import EventHandler


def test_stt_ocr_events_ignored(monkeypatch):
    handler = EventHandler(window=None)
    emitted = []
    handler._emit_safe = lambda who, msg: emitted.append((who, msg))

    assert handler.handle_event("stt.text", {"text": "hi"}) is False
    assert handler.handle_event("ocr.result", {"text": "hola"}) is False
    assert handler.handle_event("capture_assist.text", {"text": "cap", "source": "easy"}) is False

    assert emitted == []
