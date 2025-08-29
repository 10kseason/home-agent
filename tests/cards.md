Assumptions: scenarios executed manually with mocks
Risks: high variance in real apps
Alternatives: automated GUI tests
Rationale: validate critical flows quickly
1. UIA success, LLM skipped: element with stable_id resolved and read aloud.
2. OCR fallback: UIA returns none; ROI OCR via PaddleOCR (ko/en/ja/zh) captures text and is read.
3. Multiple candidates: clarify once, user chooses option, proceed.
4. Barge-in: while TTS speaking, user says "멈춰" then new command handled.
5. Preserve-lock: numbers/units/url/currency/code remain identical after translation.
6. Dangerous command: utter "삭제" requires wake word and explicit yes.
7. ROI backoff: first OCR empty, second attempt with 1.4x ROI succeeds.
8. DPI/multi-monitor: coordinate conversion correct on high DPI dual display.
9. IME input: SetValue used instead of coordinate typing for text fields.
10. Timeout retry: initial LLM call times out >600ms, second attempt with +50% budget; if fails, ask clarify.
11. Accessibility toggle: config sets `accessibility: 1` or overlay `/보조모드` switches mode.

Checklist:
- [x] Think Harder
- [x] Think Deeper
- [x] More Information
- [x] Check Again
