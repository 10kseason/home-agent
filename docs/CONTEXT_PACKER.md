# Context Packer

The context packer module composes system prompts from several fragments while
keeping each section within a fixed character budget.

## Caps dataclass
`Caps` defines the maximum number of characters allowed for each section.  This
prevents a single source from overwhelming the final prompt.

## Helpers
- `truncate(text, cap)` – safely shortens a string or `None` to `cap`
  characters.
- `format_candidates(candidates)` – turns an iterable of objects with `id`
  and `text` attributes into `id:text` lines.
- `pack(...)` – joins the clipped pieces into a newline‑delimited prompt.

## Example
```python
from src.core.context_packer import Caps, pack

caps = Caps(system=10, user_utterance=10)
text = pack("system info", "state", "digest", [], "hello world", "target", caps)
```

## Batch execution
Windows users can run `run_tests.bat` in the project root to install
requirements and execute the unit tests in one step.
