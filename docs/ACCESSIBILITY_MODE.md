Assumptions: features implemented as skeleton; PaddleOCR available; config/profiles loaded.
Risks: heavy dependencies and incomplete modules may mislead adopters.
Alternatives: Tesseract OCR or single profile without barge-in.
Rationale: document accessibility mode so future contributors continue work.

# Accessibility Mode Update

This guide documents the first accessibility-mode update that wires
configuration, prompts, schemas and core modules for rapid cursor text
reading and voice control. It is intended for both humans and agentic
contributors.

## Key Components

- **Configuration** – `config.yaml` defines rolling window TTLs, LLM budgets,
  ranking weights, translation preserve rules and profile overrides such as
  `profiles.low_vram`. Set `accessibility: 1` to enable the mode or toggle
  live from the overlay using `/보조모드`.
- **Prompts** – `prompts/llm_system.txt` enforces single candidate selection
  and two-stage `<think>` reasoning.
- **Schemas** – `schemas/request.json` and `schemas/response.json` describe
  strict payload formats validated with `jsonschema`.
- **Core Modules** – pseudocode skeletons under `src/core/` outline stable ID
  generation, ROI‑OCR, ranking, rule-based NLU, two‑stage runner, barge‑in and
  preserve‑lock logic.
- **Testing & Metrics** – `tests/cards.md` lists manual test scenarios while
  `metrics/slo.md` defines latency and failure SLOs.

## Setup

1. Install dependencies: `pip install -r requirements.txt`
2. Run the suite: `pytest -q`
3. Adjust `config.yaml` or profile overrides as needed.

# Checklist:
# - [x] Think Harder
# - [x] Think Deeper
# - [x] More Information
# - [x] Check Again
