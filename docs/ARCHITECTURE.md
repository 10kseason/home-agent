# System Architecture

This document explains how the Agent, Overlay, plugins and tools work together.

## Components
- Agent Server (FastAPI)
  - Entrypoint: `agent/main.py` → `agent/server.py`
  - Event loop: `agent/event_bus.py` (PriorityQueue + DedupStore)
  - Plugin loader: `agent/plugins/__init__.py` (auto-subscribes by `handles` prefixes)
- Overlay App (PyQt)
  - Entrypoint: `Overlay/overlay_app.py`
  - HTTP endpoints: `/event`, `/overlay/event`, proxy `/v1/chat/completions`
  - Event feed renderer (`EventHandler`) and internal tool orchestrator
- Shared Tools
  - Configured in `config.yaml: tools.*` (kind=process) with `command/args/cwd/env`
  - Examples: `stt.start`, `mictrans.start`, `ocr.start`, `capture_assist.start`

## Data Flow (Events)
- Tools and plugins emit events to the Agent `/event` endpoint.
- Agent’s EventBus routes by prefix to plugin handlers and internal routers.
- Overlay Sink plugin forwards events to the Overlay HTTP endpoints.
- Overlay renders unified results (`stt.result`, `ocr.result`, `capture_assist.result`).

## Key Plugins
- Overlay Sink: `agent/plugins/overlay_sink.py`
  - Subscribes to `stt.`, `mictrans.`, `ocr.`, `capture_assist.`, `translator.`, `overlay.`
  - Normalizes to Overlay-friendly events and posts to `/event` or `/overlay/event`.
  - Health check + retry, length truncation, duplication control.
- Translator: `agent/plugins/translator_plugin.py`
  - Cleans output (`<think>` removal), deduplicates double lines, optional `translator.text` emission.
- Assist Command: `agent/plugins/assist_cmd_plugin.py`
  - Listens to `cmd.detected` and executes capture/summarize/translate/focus.
  - Disabled when server router is configured to handle `cmd.detected`.

## Server Routers and Policies
- Router (`agent/server.py`) can handle:
  - `stt.*`, `mictrans.*` launch/stop
  - `ocr.*`, `capture_assist.*` launch/stop
  - `cmd.detected` → capture/assist/stop routing (configurable)
- VRAM policy (8–12GB): block OCR(VL)/Capture Assist while STT(영한 번역기) is running
  - Config: `orchestrator.block_ocr_while_stt` (default true) or env `BLOCK_OCR_WHILE_STT=0` to disable

## Fallbacks
- Toast relay: server re-publishes `overlay.toast` with `_relay=true` to avoid loops.
- Overlay forwarder (when Sink missing or forced): server normalizes and posts `stt./ocr./capture_assist.` to Overlay.
- MicTrans Bridge: server always bridges `mictrans.text` → Overlay (`stt.result`) for robust display.

## Deduplication
- EventBus: windowed SHA1 by type+payload
- Overlay feed: 5s text-hash window to hide duplicate messages
- STT/cmd detection: `cooldown_ms` + `duplicate_ttl_ms` for the same command

---

See also:
- `docs/EVENTS.md` for event contracts
- `docs/ASSIST_PIPELINES.md` for MicTrans/Capture flows
- `docs/TROUBLESHOOTING.md` for common issues
