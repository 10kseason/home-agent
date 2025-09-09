# Assist Pipelines

This doc describes the assistive flows for MicTrans (microphone STT) and Capture Assist (EasyOCR).

## MicTrans (Assist STT)
- Code: `Mic-trans-assist/mictrans.py`
- Capture loopback: uses `sounddevice` with user microphone, chunked by `block_ms` (default 1000ms)
- Transcribe: faster-whisper (CPU-friendly `int8` by default)
- Emit: `mictrans.text` with `assist: true`
- Display: Overlay Sink (or server bridge) sends `stt.result` → feed shows as “Assist‑MicTrans”
- Commands: cmd detection on the transcript → emits `cmd.detected` with a 2s duplicate TTL
  - Recommended routing: server router handles `cmd.detected`
  - Fallback (Overlay): EventHandler also detects “캡쳐/capture/스크린샷…” inside MicTrans text and triggers `capture_assist.start` (2s cooldown)
- Close behavior: closing the MicTrans window triggers `mictrans.stop` and gracefully shuts down the stream/thread

### Configuration
- `config.yaml → commands.*`
  - `enable_detected: true` — turn cmd routing on
  - `router_handle_detected: true` — server handles `cmd.detected` (plugin ignores)
- `Mic-trans-assist/Assist-config.yaml → assist.*`
  - `enable_detected: true` — emit `cmd.detected` from MicTrans
- Environment
  - `ASSIST_ENABLE_DETECTED=1` — allow command emission
  - `AGENT_EVENT_URL` — event URL for the Agent server

## Capture Assist (EasyOCR)
- Code: `Capture-assist/capture_assist.py`
- Immediate capture (no delay): shows a toast “촬영합니다”, grabs the screen (or region), runs EasyOCR, refines via Jan Nano (optional), and posts results
- Emit:
  - `capture_assist.text` (`assist: true`)
  - Also mirrors `stt.text` for downstream flows if needed
- Display: Overlay Sink sends `capture_assist.result` to the feed as “Assist‑Capture”
- Toasts: extra toasts titled “Assist‑Capture”/“Capture‑assist” are filtered by the Overlay to prevent duplicates

### Configuration
- `Capture-assist/Assist-config.yaml`
  - `capture.monitor` (1-based), `capture.region=[x,y,w,h]`
  - `ocr.langs=['ko','en']`, `ocr.gpu=false`
  - `event.url`, `event.key` override Agent endpoint/key if needed

## STT (영한 번역기) coexistence
- Large STT pipeline can be heavy on VRAM. To protect systems ≤12GB:
  - While STT(영한 번역기) is running, OCR(VL) and Capture Assist will not start
  - Config: `orchestrator.block_ocr_while_stt=true` (default); set env `BLOCK_OCR_WHILE_STT=0` to disable
  - MicTrans is always allowed and can be used with Capture Assist

## Translator Text
- Default OFF: `config.yaml → translate.emit_text_event: false`
- Turn on to emit `translator.text` feed entries after translation

## Duplicate Handling Reference
- Agent EventBus: time-window dedup by type+payload hash (see `limits`)
- Overlay feed: 5s duplicate text filter
- Commands: `cooldown_ms` and `duplicate_ttl_ms` suppress repeated triggers
