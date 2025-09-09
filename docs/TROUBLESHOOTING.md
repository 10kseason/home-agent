# Troubleshooting

This guide lists common problems and how to diagnose/fix them.

## Overlay shows OS toast but feed is empty
- Cause: Raw events were not normalized, or Sink not loaded
- Fix:
  - Ensure `agent/plugins/overlay_sink.py` is loading (see server logs)
  - Fallback: set `OVERLAY_FORCE_FALLBACK=1` and restart Agent
  - Verify Overlay health: `http://127.0.0.1:8350/health`

## “queued” only on POST, nothing appears
- Cause: Port/host mismatch between Overlay and Agent
- Fix:
  - Set `OVERLAY_HOST/OVERLAY_PORT` or ensure Overlay’s `agent.event_url` points to `http://127.0.0.1:8765/event`
  - Confirm `config.yaml → overlay.enable=true` when relying on auto-launch

## MicTrans text not visible
- Cause: Missing event URL env for MicTrans process
- Fix:
  - Ensure `config.yaml → tools.mictrans.start.env.ASSIST_ENABLE_DETECTED/AGENT_EVENT_URL` are nested under the tool spec
  - Restart Agent so envs apply to the subprocess
  - Server also bridges `mictrans.text` to `stt.result`; check server logs for bridge posts

## “캡쳐” said but Capture Assist didn’t start
- Check VRAM policy: if STT(영한 번역기) is running, OCR/Capture is blocked (MicTrans allowed)
  - Toast: “STT 실행 중이라 캡처를 시작하지 않습니다.”
  - Disable: `BLOCK_OCR_WHILE_STT=0` before starting Agent, or stop STT
- Ensure router handles `cmd.detected`:
  - `config.yaml → commands.router_handle_detected: true`
  - Logs show: `[router] subscribed to 'cmd.*' and 'cmd.detected'`
- Overlay fallback also detects “캡쳐/capture/스크린샷” inside MicTrans text (2s cooldown)

## Duplicate Capture messages / double toasts
- Cause: Both Sink and server fallback posted, or extra toasts from scripts
- Fix:
  - Server fallback does not subscribe to `capture_assist.*` when Sink is loaded
  - Overlay filters `Assist-Capture`/`Capture-assist` toasts (feed already shows result)

## Translator text spam
- `config.yaml → translate.emit_text_event: false` (default). Set to true only if needed

## MicTrans stays after window close
- Fixed: MicTrans UI close triggers `mictrans.stop` and stops streams/threads
- If you still see a lingering process, confirm you run the updated script and restart Agent

## Quick test commands
- POST a toast:
  - `Invoke-RestMethod http://127.0.0.1:8350/overlay/event -Method Post -ContentType 'application/json; charset=utf-8' -Body (@{ type='overlay.toast'; payload=@{ title='연결'; text='OK' } } | ConvertTo-Json -Depth 5)`
- POST MicTrans text:
  - `Invoke-RestMethod http://127.0.0.1:8765/event -Method Post -ContentType 'application/json; charset=utf-8' -Body (@{ type='mictrans.text'; payload=@{ text='테스트입니다'; assist=$true } } | ConvertTo-Json -Depth 5)`
- Start/stop tools:
  - `stt.start/stop`, `mictrans.start/stop`, `ocr.start/stop`, `capture_assist.start/stop`
