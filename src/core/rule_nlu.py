"""
Assumptions: hotpath commands cover 80%
Risks: unseen phrases slip through
Alternatives: send all to LLM
Rationale: reduce latency by skipping LLM
"""

from typing import Iterable, Optional


def rule_nlu(utter: str, candidates: Iterable) -> Optional[dict]:
    """Very small rule set for common utterances.

    Returns a DSL-like dict describing the action or ``None`` if the utterance
    should fall back to the LLM intent classifier.
    """

    utter = utter.strip()
    if utter in ("멈춰", "정지", "stop"):
        return {"action": "stop"}
    if utter.startswith("번역"):
        return {"action": "translate"}
    if utter in ("다음 줄", "다음", "next"):
        return {"action": "next_line"}
    if utter in ("이전 줄", "이전", "prev"):
        return {"action": "prev_line"}
    if utter.startswith("철자"):
        return {"action": "spell"}
    if utter.startswith("숫자"):
        return {"action": "numbers"}

    matches = [c for c in candidates if getattr(c, "text", "") and c.text in utter]
    if len(matches) == 1:
        return {"action": "read", "target": {"id": matches[0].id}}
    if len(matches) > 1:
        return {"action": "clarify", "options": [m.text for m in matches[:3]]}
    return None
# Checklist:
# - [x] Think Harder
# - [x] Think Deeper
# - [x] More Information
# - [x] Check Again
