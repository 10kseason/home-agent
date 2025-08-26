#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LM Studio OCR → Translation Snipping Tool
-----------------------------------------
핫키로 영역 캡처 → LM Studio로 보내 OCR 및 번역.

모드
- 일반 모드: OCR 모델(비전/혹은 OCR LLM) → 텍스트 추출 → 번역 모델로 번역 (2단계)
- 고속 모드: lfm2-vl-1.6b로 OCR 후 언어별 모델을 사용해 한국어로 번역 (2단계)

완료 시
- 결과 창은 띄우지 않음 (요청사항)
- 번역 결과를 클립보드로 복사 + Windows 토스트만 표시
"""

from __future__ import annotations

import sys
import os
import io
import json
import base64
import re
# ---- Luna Agent bridge (공통) ----
import requests as _rq
_AGENT_URL = os.environ.get("AGENT_EVENT_URL", "http://127.0.0.1:8765/plugin/event")
_AGENT_KEY = os.environ.get("AGENT_EVENT_KEY")

def _post_event(_type, _payload, _prio=5):

    try:

        headers = {'Content-Type': 'application/json'}

        if _AGENT_KEY:

            headers['X-Agent-Key'] = _AGENT_KEY

        _rq.post(_AGENT_URL, json={

            'type': _type, 'payload': _payload, 'priority': _prio

        }, headers=headers, timeout=3)

    except Exception:

        pass


import requests
from PIL import Image
from mss import mss

# GUI / Hotkey
from PySide6.QtCore import Qt, QRect, QPoint, Signal, QThread, QTimer
from PySide6.QtGui import QPixmap, QIcon, QAction, QShortcut, QKeySequence
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLineEdit, QMessageBox, QCheckBox, QComboBox, QSystemTrayIcon,
    QMenu, QMainWindow
)

# 전역 핫키 (Windows/Linux 권장, macOS는 권한 필요)
try:
    import keyboard  # type: ignore
    HAS_KEYBOARD = True
except Exception:
    HAS_KEYBOARD = False

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")

DEFAULT_CONFIG = {
    "server_url": "http://localhost:1234/v1",
    "api_key": "lm-studio",

    # 모델
    "ocr_model": "qwen2.5-vl",
    "translate_model": "qwen2.5-7b",

    # 고속 모드
    "fast_vlm_mode": False,
    "fast_vlm_model": "lfm2-vl-1.6b",

    "target_language": "Korean",
    "hotkey": "ctrl+alt+o",

    # 알림/동작
    "copy_to_clipboard": True,
    "notify_on_finish": True,
    "notify_content": "translation",   # translation | ocr | both

    # 창/팝업 관련(토스트만 쓰므로 기본 꺼둠)
    "show_intermediate_ocr": False,
    "show_result_window": False,       # 결과 창 표시 안 함
    "show_result_popup": False,        # 작은 팝업 표시 안 함
    "popup_content": "translation"
}


def load_config() -> dict:
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            for k, v in DEFAULT_CONFIG.items():
                if k not in data:
                    data[k] = v
            return data
        except Exception:
            pass
    return DEFAULT_CONFIG.copy()


def save_config(cfg: dict):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


# -------- 화면 캡처 오버레이 --------
class SnipOverlay(QWidget):
    regionSelected = Signal(int, int, int, int)  # left, top, width, height

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Snip Overlay")
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setWindowState(Qt.WindowFullScreen)
        self._origin: QPoint | None = None
        self._current: QRect | None = None
        self.setCursor(Qt.CrossCursor)

    def paintEvent(self, event):
        from PySide6.QtGui import QPainter, QColor, QPen, QBrush
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(0, 0, 0, 80))
        if self._current:
            p.setPen(QPen(QColor(255, 255, 255, 220), 2))
            p.setBrush(QBrush(QColor(0, 120, 215, 60)))
            p.drawRect(self._current)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._origin = e.globalPosition().toPoint()

    def mouseMoveEvent(self, e):
        if not self._origin:
            return
        cur = e.globalPosition().toPoint()
        x1, y1 = min(self._origin.x(), cur.x()), min(self._origin.y(), cur.y())
        x2, y2 = max(self._origin.x(), cur.x()), max(self._origin.y(), cur.y())
        self._current = QRect(QPoint(x1, y1), QPoint(x2, y2))
        self.update()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton and self._current:
            r = self._current
            self.regionSelected.emit(r.left(), r.top(), r.width(), r.height())
            self.hide()
            self._origin = None
            self._current = None
        elif e.button() == Qt.RightButton:
            self.hide()
            self._origin = None
            self._current = None


# -------- OCR+번역 워커 --------
class WorkerOCRTranslate(QThread):
    finished = Signal(str, str)  # (ocr_text, translated_text)
    error = Signal(str)

    def __init__(self, cfg: dict, image_bytes: bytes, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.image_bytes = image_bytes

    # ---- 공통 호출 (Chat Completions) ----
    def _post_chat(self, payload: dict) -> dict:
        url = self.cfg["server_url"].rstrip("/") + "/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.cfg.get('api_key', 'lm-studio')}",
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()

    # ---- 고속 모드: lfm2-vl-1.6b OCR 후 언어별 번역 ----
    def _run_fast_vlm(self, image_bytes: bytes) -> tuple[str, str]:
        b64 = base64.b64encode(image_bytes).decode("ascii")
        model_name = self.cfg.get("fast_vlm_model") or "lfm2-vl-1.6b"
        payload = {
            "model": model_name,
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an OCR engine. Extract ALL visible text from the image. Keep original line breaks. Output plain text only.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract text (OCR). Return plain text only."},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"}},
                    ],
                },
            ],
        }
        data = self._post_chat(payload)
        ocr_text = ((data.get("choices") or [{}])[0].get("message", {}).get("content", "")).strip()

        lang = self._detect_lang(ocr_text)
        if lang == "en":
            model = "hyperclovax-seed-text-instruct-1.5b"
        elif lang in ("ja", "zh"):
            model = "qwen/qwen3-4b-thinking-2507"
        else:
            model = self.cfg["translate_model"]

        translated = self._run_translate(ocr_text, model)
        return ocr_text, translated

    # ---- 일반 모드: OCR → 번역 ----
    def _run_ocr(self, image_bytes: bytes) -> str:
        b64 = base64.b64encode(image_bytes).decode("ascii")
        model_name = self.cfg["ocr_model"]
        payload = {
            "model": model_name,
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an OCR engine. Extract ALL visible text from the image. Keep original line breaks. Output plain text only.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract text (OCR). Return plain text only."},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"}},
                    ],
                },
            ],
        }
        data = self._post_chat(payload)
        return ((data.get("choices") or [{}])[0].get("message", {}).get("content", "")).strip()

    def _run_translate(self, text: str, model_name: str | None = None) -> str:
        model_name = model_name or self.cfg["translate_model"]
        target = self.cfg["target_language"]
        system_prompt = (
            "You are a professional translator. Translate the user's text into the target language. "
            "Preserve line breaks. Output only the translation."
        )
        user = f"Target language: {target}\n\n---\n{text}\n---"
        payload = {
            "model": model_name,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user},
            ],
        }
        data = self._post_chat(payload)
        return ((data.get("choices") or [{}])[0].get("message", {}).get("content", "")).strip()

    def _detect_lang(self, text: str) -> str:
        if re.search("[\u3040-\u30ff]", text):
            return "ja"
        if re.search("[\u4e00-\u9fff]", text):
            return "zh"
        return "en"

    def run(self):
        try:
            if self.cfg.get("fast_vlm_mode", False):
                ocr_text, translated = self._run_fast_vlm(self.image_bytes)
            else:
                ocr_text = self._run_ocr(self.image_bytes)
                translated = self._run_translate(ocr_text)
            self.finished.emit(ocr_text, translated)
        except Exception as e:
            self.error.emit(f"{type(e).__name__}: {e}")


# -------- 메인 윈도우 --------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.cfg = load_config()
        self.setWindowTitle("LM Studio OCR → Translation")
        self.setMinimumWidth(780)
        self._notified_hide = False

        # 폼
        self.ed_server = QLineEdit(self.cfg["server_url"])
        self.ed_key = QLineEdit(self.cfg["api_key"])
        self.ed_key.setEchoMode(QLineEdit.Password)
        self.ed_ocr_model = QLineEdit(self.cfg["ocr_model"])
        self.ed_trans_model = QLineEdit(self.cfg["translate_model"])
        self.ed_target_lang = QLineEdit(self.cfg["target_language"])
        self.ed_hotkey = QLineEdit(self.cfg["hotkey"])

        # 고속 모드
        self.chk_fast = QCheckBox("고속 OCR 모드")
        self.chk_fast.setChecked(self.cfg.get("fast_vlm_mode", False))
        self.ed_fast_model = QLineEdit(self.cfg.get("fast_vlm_model", ""))
        self.ed_fast_model.setPlaceholderText("예: lfm2-vl-1.6b")
        self.chk_fast.toggled.connect(self._on_fast_toggled)
        self._on_fast_toggled(self.chk_fast.isChecked())

        # 알림/옵션
        self.chk_copy = QCheckBox("번역 결과를 자동으로 클립보드에 복사")
        self.chk_copy.setChecked(self.cfg["copy_to_clipboard"])
        self.chk_notify = QCheckBox("완료 시 Windows 알림 표시")
        self.chk_notify.setChecked(self.cfg.get("notify_on_finish", True))
        self.cmb_notify_content = QComboBox()
        self.cmb_notify_content.addItems(["translation", "ocr", "both"])
        self.cmb_notify_content.setCurrentText(self.cfg.get("notify_content", "translation"))

        # (선택) 유지하되 기본 꺼짐
        self.chk_show_ocr = QCheckBox("OCR 중간 결과 팝업 표시")
        self.chk_show_ocr.setChecked(self.cfg["show_intermediate_ocr"])
        self.chk_popup = QCheckBox("완료 시 작은 알림창 표시")
        self.chk_popup.setChecked(self.cfg.get("show_result_popup", False))
        self.cmb_popup_content = QComboBox()
        self.cmb_popup_content.addItems(["translation", "ocr", "both"])
        self.cmb_popup_content.setCurrentText(self.cfg.get("popup_content", "translation"))

        self.btn_save = QPushButton("설정 저장")
        self.btn_register_hotkey = QPushButton("핫키 재등록")
        self.btn_snip_now = QPushButton("지금 캡처하기")

        # 레이아웃
        grid = QGridLayout()
        r = 0
        grid.addWidget(QLabel("LM Studio URL"), r, 0)
        grid.addWidget(self.ed_server, r, 1)
        r += 1
        grid.addWidget(QLabel("API Key(임의 값 가능)"), r, 0)
        grid.addWidget(self.ed_key, r, 1)
        r += 1
        grid.addWidget(QLabel("OCR 모델명 (비전/혹은 OCR LLM)"), r, 0)
        grid.addWidget(self.ed_ocr_model, r, 1)
        r += 1
        grid.addWidget(QLabel("번역 모델명 (텍스트)"), r, 0)
        grid.addWidget(self.ed_trans_model, r, 1)
        r += 1
        grid.addWidget(QLabel("타겟 언어"), r, 0)
        grid.addWidget(self.ed_target_lang, r, 1)
        r += 1
        grid.addWidget(QLabel("전역 핫키"), r, 0)
        grid.addWidget(self.ed_hotkey, r, 1)
        r += 1

        grid.addWidget(self.chk_fast, r, 0, 1, 2)
        r += 1
        grid.addWidget(QLabel("고속 모드용 OCR 모델"), r, 0)
        grid.addWidget(self.ed_fast_model, r, 1)
        r += 1

        grid.addWidget(self.chk_copy, r, 0, 1, 2)
        r += 1
        grid.addWidget(QLabel("토스트 알림 내용"), r, 0)
        grid.addWidget(self.cmb_notify_content, r, 1)
        r += 1
        grid.addWidget(self.chk_notify, r, 0, 1, 2)
        r += 1

        # (선택) 유지하되 기본 꺼짐
        grid.addWidget(self.chk_show_ocr, r, 0, 1, 2)
        r += 1
        grid.addWidget(QLabel("팝업 알림 내용"), r, 0)
        grid.addWidget(self.cmb_popup_content, r, 1)
        r += 1
        grid.addWidget(self.chk_popup, r, 0, 1, 2)
        r += 1

        btns = QHBoxLayout()
        btns.addWidget(self.btn_save)
        btns.addWidget(self.btn_register_hotkey)
        btns.addStretch(1)
        btns.addWidget(self.btn_snip_now)

        wrapper = QWidget()
        v = QVBoxLayout(wrapper)
        v.addLayout(grid)
        v.addLayout(btns)
        self.setCentralWidget(wrapper)

        # 오버레이
        self.overlay = SnipOverlay()
        self.overlay.regionSelected.connect(self.on_region_selected)

        # 트레이
        if QSystemTrayIcon.isSystemTrayAvailable():
            icon = QIcon.fromTheme("camera")
            if icon.isNull():
                pix = QPixmap(32, 32)
                pix.fill(Qt.transparent)
                icon = QIcon(pix)
            self.tray = QSystemTrayIcon(icon, self)
            menu = QMenu()
            act_show = QAction("열기", self)
            act_show.triggered.connect(self.showNormal)
            act_snip = QAction("캡처", self)
            act_snip.triggered.connect(self.trigger_snip)
            act_quit = QAction("종료", self)
            act_quit.triggered.connect(QApplication.instance().quit)
            menu.addAction(act_show)
            menu.addAction(act_snip)
            menu.addSeparator()
            menu.addAction(act_quit)
            self.tray.setContextMenu(menu)
            self.tray.setToolTip("LM Studio OCR → Translation")
            self.tray.activated.connect(self.on_tray_activated)
            self.tray.show()
        else:
            self.tray = None

        # 시그널
        self.btn_save.clicked.connect(self.on_save)
        self.btn_snip_now.clicked.connect(self.trigger_snip)
        self.btn_register_hotkey.clicked.connect(self.register_hotkey)

        # 핫키 등록 + 포커스 폴백
        self.register_hotkey(auto=True)
        self.shortcut = QShortcut(QKeySequence(self.cfg["hotkey"]), self)
        self.shortcut.activated.connect(self.trigger_snip)

    def _on_fast_toggled(self, checked: bool):
        """고속 모드 토글 시 기본 OCR/번역 설정 비활성화"""
        self.ed_ocr_model.setEnabled(not checked)
        self.ed_trans_model.setEnabled(not checked)

    # ---------- 알림 유틸 ----------
    def _truncate(self, s: str, max_len: int = 240) -> str:
        s = (s or "").strip()
        return s if len(s) <= max_len else s[: max_len - 1] + "…"

    def _build_notify_text(self, ocr_text: str, translated: str) -> str:
        choice = (self.cfg.get("notify_content") or "translation").lower()
        if choice == "ocr":
            return self._truncate(ocr_text)
        if choice == "both":
            return self._truncate(f"OCR:\n{ocr_text}\n\nTRANSLATION:\n{translated}", 260)
        return self._truncate(translated or ocr_text)

    def _show_windows_notification(self, title: str, message: str):
        # Lunar Bridge → Overlay toast
        _post_event("overlay.toast", {"title": title, "text": message})
        if getattr(self, "tray", None):
            try:
                self.tray.showMessage(title, message, QSystemTrayIcon.Information, 8000)
                return
            except Exception:
                pass
        try:
            from win10toast import ToastNotifier  # type: ignore
            ToastNotifier().show_toast(title, message, duration=8, threaded=True)
            return
        except Exception:
            pass

    # ---------- 이벤트 ----------
    def on_save(self):
        self.cfg["server_url"] = self.ed_server.text().strip() or "http://localhost:1234/v1"
        self.cfg["api_key"] = self.ed_key.text().strip() or "lm-studio"
        self.cfg["ocr_model"] = self.ed_ocr_model.text().strip()
        self.cfg["translate_model"] = self.ed_trans_model.text().strip()
        self.cfg["target_language"] = self.ed_target_lang.text().strip() or "Korean"
        self.cfg["hotkey"] = self.ed_hotkey.text().strip() or "ctrl+alt+o"

        self.cfg["fast_vlm_mode"] = self.chk_fast.isChecked()
        self.cfg["fast_vlm_model"] = self.ed_fast_model.text().strip()

        self.cfg["copy_to_clipboard"] = self.chk_copy.isChecked()
        self.cfg["notify_on_finish"] = self.chk_notify.isChecked()
        self.cfg["notify_content"] = self.cmb_notify_content.currentText()

        # (선택) 유지하되 기본 꺼짐
        self.cfg["show_intermediate_ocr"] = self.chk_show_ocr.isChecked()
        self.cfg["show_result_popup"] = self.chk_popup.isChecked()
        self.cfg["popup_content"] = self.cmb_popup_content.currentText()
        save_config(self.cfg)

        # 폴백 단축키 재설정
        try:
            self.shortcut.setKey(QKeySequence(self.cfg["hotkey"]))
        except Exception:
            pass

        self.register_hotkey(auto=False)
        QMessageBox.information(self, "저장됨", "설정을 저장하고 핫키를 재등록했습니다.")

    def register_hotkey(self, auto=False):
        if not HAS_KEYBOARD:
            if not auto:
                QMessageBox.warning(
                    self,
                    "핫키 라이브러리 없음",
                    "keyboard 라이브러리가 감지되지 않았습니다.\n"
                    "전역 핫키를 쓰려면 'pip install keyboard' 후 프로그램을 재시작하세요.",
                )
            return
        try:
            keyboard.clear_all_hotkeys()
            keyboard.add_hotkey(self.cfg["hotkey"], lambda: QTimer.singleShot(0, self.trigger_snip))
            if not auto:
                QMessageBox.information(self, "등록됨", f"전역 핫키 '{self.cfg['hotkey']}' 등록 완료.")
        except Exception as e:
            if not auto:
                QMessageBox.critical(self, "오류", f"핫키 등록 실패: {e}")

    def trigger_snip(self):
        self.overlay.show()

    def on_region_selected(self, left: int, top: int, width: int, height: int):
        try:
            with mss() as sct:
                bbox = {"left": int(left), "top": int(top), "width": int(width), "height": int(height)}
                shot = sct.grab(bbox)
                img = Image.frombytes("RGB", shot.size, shot.rgb)
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                image_bytes = buf.getvalue()
        except Exception as e:
            QMessageBox.critical(self, "캡처 실패", f"화면 캡처에 실패했습니다.\n{e}")
            return

        self.worker = WorkerOCRTranslate(self.cfg, image_bytes)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def on_finished(self, ocr_text: str, translated: str):
        # 결과 창/작은 팝업/중간 OCR 팝업은 전부 생략 — 토스트 + 복붙만!
        if self.cfg.get("copy_to_clipboard", True):
            QApplication.clipboard().setText((translated or ocr_text or "").strip())

        model_info = {
            "ocr_model": self.cfg.get("fast_vlm_model") if self.cfg.get("fast_vlm_mode", False) else self.cfg.get("ocr_model"),
            "translate_model": self.cfg.get("translate_model"),
        }
        _post_event(
            "ocr.text",
            {
                "text": translated or ocr_text,
                "ocr": ocr_text,
                "source": "HomeOCR",
                **model_info,
            },
        )

        if self.cfg.get("notify_on_finish", True):
            self._show_windows_notification(
                "번역 완료 (클립보드 복사됨)",
                self._build_notify_text(ocr_text, translated),
            )

    def on_error(self, msg: str):
        QMessageBox.critical(self, "오류", f"처리 중 오류가 발생했습니다.\n{msg}")

    # 트레이 아이콘 클릭 → 창 토글
    def on_tray_activated(self, reason):
        if reason == QSystemTrayIcon.Trigger:
            if self.isHidden() or not self.isActiveWindow():
                self.showNormal()
                self.activateWindow()
                self.raise_()
            else:
                self.hide()

    # X 버튼 → 트레이로 최소화
    def closeEvent(self, event):
        if getattr(self, "tray", None):
            event.ignore()
            self.hide()
            self.tray.showMessage(
                "백그라운드로 전환됨",
                "트레이 아이콘에서 다시 열 수 있습니다.",
                QSystemTrayIcon.Information,
                3000,
            )
        else:
            super().closeEvent(event)


# -------- 엔트리 --------
def main():
    app = QApplication(sys.argv)
    app.setApplicationName("LM Studio OCR → Translation")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
