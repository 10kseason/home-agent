"""
Assumptions: UIA resolution succeeds most of time
Risks: wrong element gets activated
Alternatives: always ask confirmation
Rationale: UIA pattern then coordinate fallback for robustness
"""

def execute(dsl):
    if is_dangerous(dsl) and not confirm_with_wake():
        return "cancelled"
    tid = dsl.get("target", {}).get("id")
    el = resolve_uia(tid)
    a = dsl["action"]
    if a == "invoke":
        if el and has_invoke(el):
            return el.invoke()
        return click_center(tid)
    if a == "set_value":
        val = dsl["params"]["value"]
        if el and has_value(el):
            return el.set_value(val)
        return type_fallback(val)
    if a == "select":
        txt = dsl["params"].get("text")
        if el and has_selection(el):
            return el.select(txt)
        return ocr_click_by_text(txt)
    if a in ("key", "hotkey"):
        return send_key(dsl["params"])
# Checklist:
# - [x] Think Harder
# - [x] Think Deeper
# - [x] More Information
# - [x] Check Again
