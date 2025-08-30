# Build & Test
setup: pip install -r requirements.txt || true
test: pytest -q

# Preparation
- Read `For Agent readme.txt` before making changes; it directs you to review `홈 에이전트 첫 업데이트 용 장애보조 모드 계획서..txt`.
- After completing a feature or doc update from the readme, create or update `Agent memory for working.txt` noting the work done and reminding that remaining items still need implementation.

# Rules
- Do not leak secrets.
- Ask for approval before sensitive actions.
- Keep changes minimal and documented.

# Commands
- Run tests: pytest -q
- Lint (optional): ruff .

## Repository Structure (Quick Reference)
- `agent/` – event-driven agent server; `main.py` launches the server and `plugins/` houses event handlers.
- `docs/` – project documentation (see `docs/REPO_STRUCTURE.md` for a full directory tree).
- `src/` – core library modules such as `src/core/context_packer.py` for preparing conversation context.
- `STT/` – speech-to-text utilities like `assist.py` for live microphone capture and `VSRG-Ts-to-kr.py` for subtitle translation.
- `OCR/` – optical character recognition tools; `main.py` provides a GUI for screenshot capture and translation.
- `Overlay/` – desktop overlay application and its plugin system.
- `tests/` – pytest suites verifying modules (e.g., `test_assist.py`, `test_context_packer.py`).
- `tools/` – shared runtime tool registry and security hooks used by both the agent and overlay.
- `run_tests.bat` – Windows helper script to install dependencies and run lint/tests in one step.
- `Agent memory for working.txt` – running log of completed work and remaining tasks.
