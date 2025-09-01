# Assist Commands / 보조 명령

This document summarizes the assistive modules and the command plugin that react to speech in accessibility mode.

## MicTrans & Capture Assist
- **MicTrans** (`Mic-trans-assist/mictrans.py`) captures microphone audio as voice input and posts `mictrans.text` events with `assist: true`. The overlay labels these transcripts as **Assist-MicTrans**.
- **Capture Assist** (`Capture-assist/capture_assist.py`) waits five seconds, takes a screenshot, performs OCR, and emits `capture_assist.text` events marked with `assist: true`. The overlay displays the text as **Assist-Capture**.

## Command Plugin
`agent/plugins/assist_cmd_plugin.py` watches STT transcripts for the following Korean keywords and triggers the matching tools:

| Keyword | Action |
|---------|-------|
| "캡처" | Run Capture Assist and show the result on the overlay |
| "요약" | Summarize the latest OCR text with the local LLM |
| "번역" | Translate the OCR text (or its summary if one exists) |
| "다시" | Repeat the previous assist action |
| "집중모드" | Toggle focus mode through the focus-assist tool |

All actions are recorded via the agent server so sessions can be reviewed or repeated later.

## Local LLM & Path Notes
Both assist scripts resolve their `Assist-config.yaml` relative to the script
file, so they run correctly even when launched from another working
directory. When the overlay enters assist mode and a local model (e.g. LM
Studio or Ollama) calls `mictrans.start` or `capture_assist.start`, the agent
spawns these tools using the relative paths in `config.yaml`.

- **MicTrans** can forward wake-word prompts to a local LLM if `llm.endpoint`
  and `llm.model` are set in `Mic-trans-assist/Assist-config.yaml`.
- **Capture Assist** refines EasyOCR output when `LM_STUDIO_ENDPOINT`/
  `OLLAMA_ENDPOINT` (and corresponding `*_MODEL` variables) are defined in the
  environment.

These hooks allow a local LLM to orchestrate capture, summarization and
translation without path issues.
