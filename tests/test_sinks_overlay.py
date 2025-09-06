import agent.sinks as sinks


def test_toast_notify_posts_overlay(monkeypatch):
    events = []
    monkeypatch.setattr(sinks, "_post_event", lambda t, p, prio=5: events.append((t, p, prio)))
    monkeypatch.setattr(sinks, "_toast_win10", lambda t, m: True)
    monkeypatch.setattr(sinks, "_toast_winotify", lambda t, m: True)
    sinks.toast_notify("Title", "Hello")
    assert events == [("overlay.toast", {"title": "Title", "text": "Hello"}, 5)]
