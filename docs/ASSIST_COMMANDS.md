# Assist Commands / 보조 명령

This document summarizes the assistive modules and the command plugin that react to speech in accessibility mode.

## STT & OCR Assist
- **STT Assist** (`STT/assist.py`) captures microphone audio in real time and posts `stt.text` events with `assist: true`. The overlay labels these transcripts as **Assist-STT**.
- **OCR Assist** (`OCR/OCR-Assist.py`) takes a screenshot, performs OCR, and emits `ocr.text` events marked with `assist: true`. The overlay displays the text as **Assist-OCR**.

## Command Plugin
`agent/plugins/assist_cmd_plugin.py` watches STT transcripts for the following Korean keywords and triggers the matching tools:

| Keyword | Action |
|---------|-------|
| "캡처" | Run OCR Assist and show the result on the overlay |
| "요약" | Summarize the latest OCR text with the local LLM |
| "번역" | Translate the OCR text (or its summary if one exists) |
| "다시" | Repeat the previous assist action |
| "집중모드" | Toggle focus mode through the focus-assist tool |

All actions are recorded via the agent server so sessions can be reviewed or repeated later.
