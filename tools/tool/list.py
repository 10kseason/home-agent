# tools/list.py
from __future__ import annotations
import json, yaml, sys
from pathlib import Path

def main():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from tools.tool import HANDLERS
    schemas = []
    ty = root / "tools" / "config" / "tools.yaml"
    if ty.exists():
        items = yaml.safe_load(ty.read_text(encoding="utf-8"))
        if isinstance(items, list):
            for it in items:
                nm = it.get("name")
                if nm in HANDLERS:
                    schemas.append({
                        "type":"function",
                        "function":{
                            "name": nm,
                            "description": it.get("desc",""),
                            "parameters": it.get("input_schema", {"type":"object"})
                        }
                    })
    print("# Implemented tools")
    print(json.dumps(sorted(list(HANDLERS.keys())), ensure_ascii=False, indent=2))
    print("\n# Tools array for OpenAI/vLLM")
    print(json.dumps(schemas, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
