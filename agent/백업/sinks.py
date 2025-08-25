from loguru import logger
import threading
from .sinks import toast_notify, write_log

async def _sink_web_result(e: Event):
    q = (e.payload or {}).get("query", "")
    n = len((e.payload or {}).get("items", []))
    toast_notify("Web Search", f"{q} — {n}개 결과")
    write_log(f"[web.search] {q} ({n} results)")
    ctx.bus.subscribe("web.search.", _sink_web_result)

def _toast_win10(title: str, msg: str) -> bool:
    try:
        from win10toast import ToastNotifier
        # threaded=False로 호출 → 내부 WNDPROC 이슈 회피
        ToastNotifier().show_toast(title, msg, duration=4, threaded=False)
        return True
    except Exception as e:
        logger.debug(f"[win10toast] fallback due to: {e}")
        return False

def _toast_winotify(title: str, msg: str) -> bool:
    try:
        from winotify import Notification, audio  # pip install winotify (선택)
        n = Notification(app_id="Luna Local Agent", title=title, msg=msg)
        n.set_audio(audio.Default, loop=False)
        n.show()
        return True
    except Exception as e:
        logger.debug(f"[winotify] fallback due to: {e}")
        return False

def toast_notify(title: str, msg: str):
    # 별도 쓰레드에서 안전하게 실행 (버스/이벤트 루프 비블로킹)
    def _run():
        if _toast_win10(title, msg):
            return
        if _toast_winotify(title, msg):
            return
        logger.info(f"[TOAST] {title}: {msg}")

    threading.Thread(target=_run, daemon=True).start()

def write_log(text: str, log_file: str | None = None):
    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(text + "\n")
    else:
        logger.info(text)
