from loguru import logger
import threading

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

def toast_notify(title: str, msg: str):
    """Threaded toast notification helper (non-blocking)."""
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
