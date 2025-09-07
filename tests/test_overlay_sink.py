import asyncio
from agent.plugins.overlay_sink import EnhancedOverlaySink
from agent.schemas import Event


def test_capture_assist_event_forwarded(monkeypatch):
    sink = EnhancedOverlaySink()
    captured = {}

    async def fake_post(url, payload):
        captured['url'] = url
        captured['payload'] = payload
        return True

    sink._post_with_retry = fake_post  # type: ignore
    event = Event(type="capture_assist.text", payload={"text": "hello", "assist": True})
    asyncio.run(sink.handle(event))

    assert captured['payload']['type'] == 'capture_assist.result'
    inner = captured['payload']['payload']
    assert inner['text'] == 'hello'
    assert inner['assist'] is True


def test_mictrans_event_forwarded(monkeypatch):
    sink = EnhancedOverlaySink()
    captured = {}

    async def fake_post(url, payload):
        captured['payload'] = payload
        return True

    sink._post_with_retry = fake_post  # type: ignore
    event = Event(type="mictrans.text", payload={"text": "hi", "translation": "안녕"})
    asyncio.run(sink.handle(event))

    assert captured['payload']['type'] == 'stt.result'
    inner = captured['payload']['payload']
    assert inner['original'] == 'hi'
    assert inner['translation'] == '안녕'
    assert '안녕' in inner['text']


def test_ocr_event_forwarded(monkeypatch):
    sink = EnhancedOverlaySink()
    captured = {}

    async def fake_post(url, payload):
        captured['payload'] = payload
        return True

    sink._post_with_retry = fake_post  # type: ignore
    event = Event(type="ocr.text", payload={"text": "generic"})
    asyncio.run(sink.handle(event))

    assert captured['payload']['type'] == 'ocr.result'
    inner = captured['payload']['payload']
    assert inner['text'] == 'generic'


def test_overlay_toast_relay(monkeypatch):
    sink = EnhancedOverlaySink()
    captured = []

    async def fake_post(url, payload):
        captured.append((url, payload))
        return True

    sink._post_with_retry = fake_post  # type: ignore

    # original toast without _relay should be ignored
    event = Event(type="overlay.toast", payload={"title": "T", "text": "hi"})
    asyncio.run(sink.handle(event))
    assert captured == []

    # relayed toast should be forwarded to overlay
    event2 = Event(type="overlay.toast", payload={"title": "T", "text": "hi", "_relay": True})
    asyncio.run(sink.handle(event2))

    assert captured and captured[0][1]["type"] == "overlay.toast"
    assert captured[0][1]["payload"]["title"] == "T"
