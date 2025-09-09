<p align="right"><a href="README.md">한국어(ko)</a> | <b>English (en)</b></p>

# Luna Home Agent (Experimental)

This repository contains a local Agent + Overlay that orchestrates STT, OCR, capture assist, and simple tools on Windows. It is designed to run with local runtimes (LM Studio/Ollama etc.) and to be portable via relative paths.

## Quick Start

- Install dependencies (creates .venv):
  - Double‑click `Python-env-installer.bat`
- Run the server:
  - Double‑click `run-server.bat`

The Agent auto‑launches the Overlay (if enabled in `config.yaml`). All tool paths are relative and resolved at runtime.

## What Works Now

- STT (System/monitor audio) — VSRG translator script; can monitor system audio (e.g., via OBS monitor or audio loopback). Emits `stt.text` and shows as `stt.result` on the Overlay.
- OCR (Vision‑Language pipeline) — launches a VL OCR pipeline (configurable) and posts results.
- Capture Assist (EasyOCR) — immediate screenshot capture (no 5s delay), EasyOCR, optional Jan‑Nano refinement, and `capture_assist.result` to the Overlay.
- MicTrans (microphone STT) — lightweight Whisper (CPU‑friendly), shows as “Assist‑MicTrans”; detects assist commands (e.g., “capture”).
- Web Search (experimental, known issues) — provides search + basic summaries; still unstable.

## VRAM ≤ 12GB Friendly Policy

- When the heavy STT (English→Korean translator) is running, the Agent blocks OCR(VL) and Capture‑Assist to protect VRAM/RAM. MicTrans remains allowed.
- Toggle policy with env: `BLOCK_OCR_WHILE_STT=0` before starting the server.

## Language + Paths

- All tool/script paths use relative paths in `config.yaml`.
- The server discovers folders/scripts and persists them into `paths.cache.json` (created on first run). If the cache is missing at startup, the server creates it and restarts automatically.
- Optional user override:
  - Place a `paths.user.json` (or set `HOME_AGENT_PATHS`) to pin folders/scripts. This layer has highest priority. All paths are normalized to forward slashes.

## Recent Updates (Summary)

- Relative‑path portability across Agent/Overlay/Tools (Pathlib resolution).
- First‑run path cache with auto‑restart; periodic persistence on shutdown.
- MicTrans → Overlay bridge for reliable `stt.result` display.
- Command routing via server (cmd.detected), duplicate suppression (2s TTL).
- Capture Assist: immediate shot (removed the 5s wait); duplicate toasts filtered in Overlay.
- Translator text to feed is OFF by default (`translate.emit_text_event=false`).
- Batch scripts (ASCII‑only): `Python-env-installer.bat`, `run-server.bat`.

## Docs

- Install & Configure: `docs/INSTALL_AND_CONFIG.md`
- Architecture: `docs/ARCHITECTURE.md`
- Events: `docs/EVENTS.md`
- Assist Pipelines: `docs/ASSIST_PIPELINES.md`
- Troubleshooting: `docs/TROUBLESHOOTING.md`

## Notes

- LM Studio/Ollama endpoints are used for LLMs; model weights are not included.
- Web search remains experimental and may be unstable.

