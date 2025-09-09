# Event Contracts

This doc lists the core event types exchanged between tools, the Agent, and the Overlay. Use these to avoid ambiguity and to prevent argument collisions.

## Normalized Result Events (Overlay feed)
- `stt.result`
  - payload: `{ text: str, original: str, translation: str, confidence: number, assist: bool }`
  - source: MicTrans or general STT; `assist=true` is shown as “Assist‑MicTrans”.
- `ocr.result`
  - payload: `{ text: str, bbox?: [x,y,w,h], confidence?: number, assist?: bool }`
  - general OCR pipeline
- `capture_assist.result`
  - payload: `{ text: str, bbox?: [x,y,w,h], confidence?: number, assist: true }`
  - EasyOCR assist pipeline (assist UI)

## Raw Ingress Events (from tools)
- `stt.text` — general STT transcript (may include `translation`)
- `mictrans.text` — MicTrans transcript (`assist: true`)
- `ocr.text` — generic OCR text
- `capture_assist.text` — EasyOCR assist text (`assist: true`)

These are consumed by the Sink/forwarders and transformed into `*.result` events for the Overlay.

## Overlay UI
- `overlay.toast`: `{ title: str, text: str, _relay?: true }`
  - The Agent relays toasts with `_relay=true` so the plugin can avoid loops.
  - Overlay filters internal duplicates (e.g., “Assist‑Capture” toasts) because the feed already shows the capture result.

## Translator
- `translator.text`: `{ text: str, source: str }`
  - Default OFF. Enable via `config.yaml: translate.emit_text_event=true`.

## Commands
- `cmd.detected`: `{ cmd: 'capture'|'summarize'|'translate'|'focus'|'repeat'|'assist', ts: number }`
  - Emitted by MicTrans/STT command detectors.
  - Routed by the server (recommended) or by `AssistCommandPlugin` (fallback).

## Server Control
- `stt.start|stt.stop`, `mictrans.start|mictrans.stop`
- `ocr.start|ocr.stop`, `capture_assist.start|capture_assist.stop`

## Deduplication Rules
- Agent EventBus: map of `sha1(type+sorted(payload))` within a time window (see `config.yaml: limits`)
- Overlay feed: 5s text-hash window to avoid repeat messages
- Commands: `cooldown_ms` + `duplicate_ttl_ms` (default 2000ms) to avoid re-triggering

## Reserved Names and Guidelines
- Name new events with unique prefixes to avoid collisions, e.g., `capture_assist.text`, `mictrans.text`.
- Keep payloads minimal and typed (numbers as numbers, not strings).
- For user-visible messages prefer:
  - A feed event (`*.result`) for the main content
  - A single toast (`overlay.toast`) only when needed
