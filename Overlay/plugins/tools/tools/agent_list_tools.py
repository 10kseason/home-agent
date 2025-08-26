# tools/tool/agent_list_tools.py
from __future__ import annotations
import json, sys
from pathlib import Path
from . import HANDLERS as _H

TOOLS_YAML = Path(__file__).resolve().parents[1] / "config" / "tools.yaml"

def _load_yaml_items(path: Path):
    try:
        import yaml
        items = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(items, dict):
            items = items.get("tools") or []
        return items
    except Exception:
        # ultra-light fallback: name만 파싱
        items = []
        for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            s = ln.strip()
            if s.startswith("- name:"):
                nm = s.split(":",1)[1].strip()
                items.append({"name": nm})
        return items

def agent_list_tools(_args):
    schemas = []
    tools = []
    if TOOLS_YAML.exists():
        items = _load_yaml_items(TOOLS_YAML) or []
        for it in items:
            if not isinstance(it, dict):
                continue
            nm = it.get("name")
            if nm in _H:
                desc = it.get("desc", "")
                params = it.get("input_schema") or {"type": "object", "properties": {}}
                schemas.append({
                    "name": nm,
                    "desc": desc,
                    "parameters": params,
                })
                tools.append({
                    "type": "function",
                    "function": {
                        "name": nm,
                        "description": desc,
                        "parameters": params,
                    },
                })
    names = sorted(list(_H.keys()))
    return {"implemented": names, "schemas": schemas, "tools": tools}

TOOL_HANDLERS = {"agent.list_tools": agent_list_tools}

if __name__ == "__main__":
    print(json.dumps(agent_list_tools({}), ensure_ascii=False))
