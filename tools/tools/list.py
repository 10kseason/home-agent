# -*- coding: utf-8 -*-
"""
List registered tools + schemas from tools/config/tools.yaml
"""
from __future__ import annotations
import sys, json
from pathlib import Path

def main():
    here = Path(__file__).resolve()
    root = here.parents[2]  # .../home-agent
    sys.path.insert(0, str(root))

    from tools.tool import HANDLERS
    try:
        import yaml  # type: ignore
    except Exception:
        print(json.dumps({"ok": False, "error": "pyyaml not installed"}))
        return

    cfg = root / "tools" / "config" / "tools.yaml"
    tools = []
    if cfg.exists():
        data = yaml.safe_load(cfg.read_text(encoding="utf-8")) or []
        for it in data:
            nm = it.get("name")
            if not nm: continue
            tools.append({
                "type":"function",
                "function":{
                    "name": nm,
                    "description": it.get("desc",""),
                    "parameters": it.get("input_schema", {"type":"object","properties":{}})
                }
            })
    out = {"ok": True, "implemented": sorted(list(HANDLERS.keys())), "tools": tools}
    print(json.dumps(out, ensure_ascii=False))

if __name__ == "__main__":
    main()
