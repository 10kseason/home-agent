# Install & Configure

This guide covers installation, quick start, and key configuration knobs.

## Prerequisites
- Windows 10/11
- Python 3.10+ (recommended venv)
- Optional: LM Studio or compatible OpenAI server for LLMs
- Optional: GPU for OCR/LLM; MicTrans runs on CPU by default

## Quick Start
- Create and activate a virtual environment
- `pip install -r requirements.txt`
- Edit `config.yaml` minimally:
  - Ensure `server.host/port` (default `127.0.0.1:8765`)
  - Verify `overlay.enable=true` and script/cwd paths
  - Check `tools.*` paths (Python and script paths must exist)
- Launch agent: `python -m agent.main`
- Use Overlay hotkeys or `/help` inside the Overlay

## Key Configuration
- Commands
  - `commands.enable_detected: true` — turn on voice command routing
  - `commands.router_handle_detected: true` — server handles `cmd.detected`
- Translator
  - `translate.emit_text_event: false` — do not clutter feed by default
- VRAM Policy
  - `orchestrator.block_ocr_while_stt: true` — block OCR/Capture while the heavy STT is running
  - Override per-session: `BLOCK_OCR_WHILE_STT=0`
- Tools
  - `tools.mictrans.start.env.ASSIST_ENABLE_DETECTED: '1'`
  - Ensure `AGENT_EVENT_URL` is `http://127.0.0.1:8765/event`
- Overlay
  - Overlay quits Agent when launched by Agent and the overlay process exits
  - Duplicate toasts titled `Assist-Capture` / `Capture-assist` are filtered

## Environment Variables
- Overlay:
  - `OVERLAY_HOST` / `OVERLAY_PORT` (default `127.0.0.1:8350`)
  - `OVERLAY_EVENT_URL`, `OVERLAY_TOAST_URL`
- Tools & Plugins:
  - `AGENT_EVENT_URL`, `AGENT_EVENT_KEY`
  - `ASSIST_ENABLE_DETECTED` (`1` to emit `cmd.detected`)
  - `BLOCK_OCR_WHILE_STT` (`0` to allow OCR during STT)

## Validation
- Health: `http://127.0.0.1:8350/health` → `{ ok: true }`
- Post a toast: `POST /overlay/event { type: 'overlay.toast', payload: { title, text } }`
- Post MicTrans text: `POST /event { type: 'mictrans.text', payload: { text, assist: true } }`
- Expect: Overlay feed shows “Assist‑MicTrans”
