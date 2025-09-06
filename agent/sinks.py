from loguru import logger
import threading
import os
import requests as _rq

_EVENT_URL = (
    os.environ.get("EVENT_URL")
    or os.environ.get("AGENT_EVENT_URL")
    or os.environ.get("OVERLAY_EVENT_URL")
    or "http://127.0.0.1:8350/event"
)
_EVENT_KEY = os.environ.get("EVENT_KEY") or os.environ.get("AGENT_EVENT_KEY")

def _toast_win10(title: str, msg: str) -> bool:
    try:
        from win10toast import ToastNotifier  # pip install win10toast
        ToastNotifier().show_toast(title, msg, duration=4, threaded=False)
        return True
    except Exception as e:
        logger.debug(f"[win10toast] fallback due to: {e}")
        return False

def _toast_winotify(title: str, msg: str) -> bool:
    try:
        from winotify import Notification, audio  # pip install winotify (optional)
        n = Notification(app_id="Luna Local Agent", title=title, msg=msg)
        n.set_audio(audio.Default, loop=False)
        n.show()
        return True
    except Exception as e:
        logger.debug(f"[winotify] fallback due to: {e}")
        return False

def _post_event(_type: str, _payload: dict, _prio: int = 5) -> None:
    try:
        headers = {"Content-Type": "application/json"}
        if _EVENT_KEY:
            headers["X-Agent-Key"] = _EVENT_KEY
        _rq.post(
            _EVENT_URL,
            json={"type": _type, "payload": _payload, "priority": _prio},
            headers=headers,
            timeout=3,
        )
    except Exception:
        pass

def _overlay_toast(message: str, title: str) -> None:
    message = (message or "").strip()
    if not message:
        return
    if len(message) > 240:
        message = message[:239] + "…"
    _post_event("overlay.toast", {"title": title, "text": message})

def toast_notify(title: str, msg: str):
    """Threaded toast notification helper (non-blocking)."""
    _overlay_toast(msg, title)

    def _run():
        if _toast_win10(title, msg):
            return
        if _toast_winotify(title, msg):
            return
        logger.info(f"[TOAST] {title}: {msg}")
    threading.Thread(target=_run, daemon=True).start()

def write_log(text: str, log_file: str | None = None):
    """Write to a given file or fall back to loguru logger."""
    if log_file:
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(text + "\n")
        except Exception as e:
            logger.warning(f"[write_log] failed to write file: {e}; falling back to logger")
            logger.info(text)
    else:
        logger.info(text)
