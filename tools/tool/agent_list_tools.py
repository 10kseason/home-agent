
from __future__ import annotations
import json
from pathlib import Path
from . import HANDLERS as _H

TOOLS_YAML = Path(__file__).resolve().parents[1] / "config" / "tools.yaml"

def _load_yaml_items(path: Path):
    try:
        import yaml
        return yaml.safe_load(path.read_text(encoding="utf-8"))
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
    if TOOLS_YAML.exists():
        items = _load_yaml_items(TOOLS_YAML) or []
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

if __name__ == "__main__":
    print(json.dumps(agent_list_tools({}), ensure_ascii=False))
