# Build & Test
setup: pip install -r requirements.txt || true
test: pytest -q

# Rules
- Do not leak secrets.
- Ask for approval before sensitive actions.
- Keep changes minimal and documented.

# Commands
- Run tests: pytest -q
- Lint (optional): ruff .
