
from __future__ import annotations
from typing import Dict, Any, List
import sys
from pathlib import Path

# Ensure project root on path
here = Path(__file__).resolve()
root = here.parents[2]
sys.path.insert(0, str(root))

# Import HANDLERS and before_tool hook
from tools.tool import HANDLERS
from tools.security.orchestrator_hooks import before_tool

class ToolRegistry:
    """YAML 스키마와 구현된 HANDLERS를 연결하는 런타임 레지스트리."""
    def __init__(self, schemas: Dict[str, Dict[str, Any]]):
        self.schemas = schemas

    @classmethod
    def from_yaml(cls, path: str) -> "ToolRegistry":
        try:
            import yaml
            items = yaml.safe_load(open(path,"r",encoding="utf-8"))
        except Exception:
            items = []
        schemas = { it["name"]: it for it in items if isinstance(it, dict) and "name" in it }
        return cls(schemas)

    def tools_array(self) -> List[Dict[str, Any]]:
        arr = []
        for name, sch in self.schemas.items():
            if name in HANDLERS:
                arr.append({"type":"function","function":{
                    "name": name,
                    "description": sch.get("desc",""),
                    "parameters": sch.get("input_schema", {"type":"object"})
                }})
        return arr

    def execute(self, name: str, args: Dict[str, Any]):
        if name not in HANDLERS:
            raise RuntimeError(f"Tool not implemented: {name}")
        before_tool(name, args)  # 레이트리밋/승인
        return HANDLERS[name](args)
