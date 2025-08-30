"""Minimal DSL executor.

The executor prefers UIA actions and falls back to coordinates or OCR when an
appropriate UI Automation pattern is unavailable.  It assumes the environment
already has a local LLM stack (e.g. LM Studio or Ollama) for higher level
reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class Element:
    """Tiny stand‑in for a UI Automation element."""

    id: str
    capabilities: set[str]

    def invoke(self) -> str:
        return f"invoked:{self.id}"

    def set_value(self, value: str) -> str:
        return value

    def select(self, text: str) -> str:
        return text


# ----- helpers --------------------------------------------------------------


def is_dangerous(dsl: Dict[str, Any]) -> bool:
    return dsl.get("dangerous", False)


def confirm_with_wake() -> bool:
    return True


def resolve_uia(tid: Optional[str]) -> Optional[Element]:
    return Element(tid, {"invoke", "value", "selection"}) if tid else None


def has_invoke(el: Element) -> bool:
    return "invoke" in el.capabilities


def has_value(el: Element) -> bool:
    return "value" in el.capabilities


def has_selection(el: Element) -> bool:
    return "selection" in el.capabilities


def click_center(tid: str) -> str:
    return f"click:{tid}"


def type_fallback(value: str) -> str:
    return f"typed:{value}"


def ocr_click_by_text(text: str) -> str:
    return f"ocr:{text}"


def send_key(params: Dict[str, Any]) -> Dict[str, Any]:
    return params


# ----- main -----------------------------------------------------------------


def execute(dsl: Dict[str, Any]) -> Any:
    """Execute a simple DSL action dictionary."""

    if is_dangerous(dsl) and not confirm_with_wake():
        return "cancelled"

    tid = dsl.get("target", {}).get("id")
    el = resolve_uia(tid)
    action = dsl["action"]

    if action == "invoke":
        if el and has_invoke(el):
            return el.invoke()
        return click_center(tid)

    if action == "set_value":
        val = dsl["params"]["value"]
        if el and has_value(el):
            return el.set_value(val)
        return type_fallback(val)

    if action == "select":
        txt = dsl["params"].get("text", "")
        if el and has_selection(el):
            return el.select(txt)
        return ocr_click_by_text(txt)

    if action in ("key", "hotkey"):
        return send_key(dsl["params"])

    raise ValueError(f"unknown action: {action}")
