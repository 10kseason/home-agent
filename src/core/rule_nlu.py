"""
Assumptions: hotpath commands cover 80%
Risks: unseen phrases slip through
Alternatives: send all to LLM
Rationale: reduce latency by skipping LLM
"""

def rule_nlu(utter, candidates):
    if utter in ("멈춰", "정지"): return {"action": "stop"}
    if utter.startswith("번역"): return {"action": "translate"}
    matches = [c for c in candidates if c.text in utter]
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
