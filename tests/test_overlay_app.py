from Overlay.overlay_app import EventHandler


def test_capture_assist_event_labeled_assist(monkeypatch):
    handler = EventHandler(window=None)
    emitted = []
    handler._emit_safe = lambda who, msg: emitted.append((who, msg))
    assert handler.handle_event("capture_assist.result", {"text": "hello"}) is True
    assert emitted == [("Assist-Capture", "hello")]


def test_ocr_event_labeled_ocr(monkeypatch):
    handler = EventHandler(window=None)
    emitted = []
    handler._emit_safe = lambda who, msg: emitted.append((who, msg))
    assert handler.handle_event("ocr.result", {"text": "hola"}) is True
    assert emitted == [("👁️ OCR", "hola")]
