"""Runtime registry for YAML-defined tools."""
from __future__ import annotations
from typing import Dict, Any, List

from ..tool import HANDLERS  # re-export
from ..security.orchestrator_hooks import before_tool

REGISTRY = HANDLERS

class ToolRegistry:
    def __init__(self, schemas: Dict[str, Dict[str, Any]]):
        self.schemas = schemas

    @classmethod
    def from_yaml(cls, path: str) -> "ToolRegistry":
        try:
            import yaml  # type: ignore
            items = yaml.safe_load(open(path, "r", encoding="utf-8"))
        except Exception:
            items = []
        schemas = {it["name"]: it for it in items if isinstance(it, dict) and "name" in it}
        return cls(schemas)

    def tools_array(self) -> List[Dict[str, Any]]:
        arr = []
        for name, sch in self.schemas.items():
            if name in HANDLERS:
                arr.append({
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": sch.get("desc", ""),
                        "parameters": sch.get("input_schema", {"type": "object"}),
                    },
                })
        return arr

    def execute(self, name: str, args: Dict[str, Any]):
        if name not in HANDLERS:
            raise RuntimeError(f"Tool not implemented: {name}")
        before_tool(name, args)
        return HANDLERS[name](args)
