Assumptions / Risks / Alternatives / Rationale
- Home-agent accessibility mode skeleton exists but modules are placeholders; risk of misalignment with UIA/OCR integration.
- Low-VRAM profile is assumed; alternative is to tune `config.yaml` if GPU changes.
- JSON schema validation is strict to prevent LLM hallucinations; may need alt route for partial responses.
- Pseudocode outlines are not yet wired; functional code must respect AGENTS.md tests.
- Rolling-window cache TTLs rely on timers; race conditions need review.

# ✅ Implemented
- Central `config.yaml` with profiles (default + low_vram) and rolling window/LLM budgets.
- System prompt `prompts/llm_system.txt` and JSON schemas (`schemas/request.json`, `schemas/response.json`).
- Pseudocode stubs under `src/core/` (stable_id, roi_ocr, ranking, rule_nlu, runner, context_packer, executor, preserve_lock, barge_in).
- Manual test cards in `tests/cards.md` and SLO notes in `metrics/slo.md`.

# 🚧 To Build
1. Implement real modules replacing pseudocode:
   - UIA stable ID generation and cache.
   - ROI OCR engine with PaddleOCR multi-language top‑K and backoff.
   - Deterministic ranking formula and rule-based NLU → DSL parser.
   - Two-stage LLM runner with budget/timeout logic and retry.
   - Executor with UIA-first actions, coordinate fallback, and danger confirmation.
   - Preserve-Lock regex checks integrated into translation flow.
   - Barge-in audio gate + reflex command path.
2. Event bus & state machine (IDLE/TARGETING/READING/…) per plan.
3. Overlay badge rendering for candidate clarify flow.
4. Privacy guards: in-memory TTL purge, PII tokenization, `<think>` drop.
5. Coordinate conversion utilities (logical↔physical, multi-monitor DPI).

# 🧪 Tests Needed
- pip install -r requirements.txt (PaddleOCR/paddlepaddle 2.5.1+).
- pytest -q (unit tests once modules exist).
- Manual cards in `tests/cards.md` covering UIA success, OCR fallback, multi-monitor, ROI backoff, barge-in, preserve-lock edge cases, dangerous command confirmation, timeout retry→clarify, IME SetValue, DPI scaling.

# 🔁 Ongoing Checks
- Update `requirements.txt` when adding modules (paddle, tts, uia).
- Keep config/profile numbers in sync with docs.
- Verify JSON schemas against any response format changes.
- Ensure privacy/SLO metrics align with `metrics/slo.md` thresholds.

# 📌 Notes for Next Agent
- Follow root `AGENTS.md`: run `pip install -r requirements.txt || true` and `pytest -q` before committing.
- Use ripgrep (`rg`) for searches, avoid `ls -R`/`grep -R`.
- Maintain clean git state; commit to main branch only, no amend.
