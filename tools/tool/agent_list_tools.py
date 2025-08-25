# tools/tool/agent_list_tools.py
from __future__ import annotations
import json, yaml, os
from pathlib import Path
from . import HANDLERS as _H

TOOLS_YAML = Path("tools/config/tools.yaml")

def agent_list_tools(_args):
    schemas = []
    if TOOLS_YAML.exists():
        items = yaml.safe_load(TOOLS_YAML.read_text(encoding="utf-8"))
        if isinstance(items, list):
            for it in items:
                nm = it.get("name")
                if nm in _H:
                    schemas.append({
                        "name": nm,
                        "desc": it.get("desc",""),
                        "parameters": it.get("input_schema", {"type":"object"})
                    })
    names = sorted(list(_H.keys()))
    return {"implemented": names, "schemas": schemas}

TOOL_HANDLERS = {"agent.list_tools": agent_list_tools}
