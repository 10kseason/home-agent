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
