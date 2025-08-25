# tools/runtime_registry.py
from __future__ import annotations
from typing import Dict, Any
import sys
from pathlib import Path

# -*- coding: utf-8 -*-
from __future__ import annotations
from ..tool import HANDLERS  # re-export
REGISTRY = HANDLERS


# 경로 설정 (list.py와 동일한 방식)
here = Path(__file__).resolve()
root = here.parents[2]  # 프로젝트 루트 추정: …/home-agent

# 안전망: 만약 root/tools/tool.py 가 없으면 한 단계 더 올림
if not (root / "tools" / "tool.py").exists():
    root = root.parent
sys.path.insert(0, str(root))

try:
    from tools.tool import HANDLERS
except ImportError:
    # 같은 디렉토리의 tool.py 시도
    try:
        from .tool import HANDLERS
    except ImportError:
        # 상대 경로로 시도
        import os
        sys.path.insert(0, os.path.dirname(__file__))
        from tool import HANDLERS

from tools.security.orchestrator_hooks import before_tool

class ToolRegistry:
    def __init__(self, schemas: Dict[str, Dict[str, Any]]):
        self.schemas = schemas

    @classmethod
    def from_yaml(cls, path: str):
        try:
            import yaml
            items = yaml.safe_load(open(path,"r",encoding="utf-8"))
        except Exception:
            items = []
        schemas = { it["name"]: it for it in items if isinstance(it, dict) and "name" in it }
        return cls(schemas)

    def tools_array(self):
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