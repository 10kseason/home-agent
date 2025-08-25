
from __future__ import annotations
import importlib, pkgutil, inspect
from typing import Callable, Dict, Any, List, Optional

# Central registry of tool handlers
HANDLERS: Dict[str, Callable[[Dict[str, Any]], Any]] = {}

def _merge_tool_handlers(module) -> int:
    added = 0
    if hasattr(module, "TOOL_HANDLERS") and isinstance(module.TOOL_HANDLERS, dict):
        for k, v in module.TOOL_HANDLERS.items():
            if not callable(v): continue
            HANDLERS[k] = v
            added += 1
    return added

# Auto-import sibling modules to populate HANDLERS
def _autodiscover():
    pkg = __name__
    for m in pkgutil.iter_modules(__path__):  # type: ignore[name-defined]
        name = f"{pkg}.{m.name}"
        if m.name in ("__init__",): continue
        mod = importlib.import_module(name)
        _merge_tool_handlers(mod)

# Run discovery at import-time
_autodiscover()

# Optional: expose tool schemas from tools/config/tools.yaml
def list_tools_from_yaml() -> List[Dict[str, Any]]:
    from pathlib import Path
    try:
        import yaml  # type: ignore
    except Exception:
        return []
    cfg = Path(__file__).resolve().parents[1] / "config" / "tools.yaml"
    out: List[Dict[str, Any]] = []
    if cfg.exists():
        data = yaml.safe_load(cfg.read_text(encoding="utf-8")) or []
        for it in data:
            nm = it.get("name")
            if not nm: continue
            out.append({
                "type": "function",
                "function": {
                    "name": nm,
                    "description": it.get("desc", ""),
                    "parameters": it.get("input_schema", {"type":"object","properties":{}})
                }
            })
    return out

# Overlay bridge: if your Overlay orchestrator exposes register_tool(name, schema, handler),
# you can call register(overlay) to attach all tools declared in tools.yaml that are implemented.
def register(overlay) -> Dict[str, Any]:
    schemas = list_tools_from_yaml()
    count = 0
    for s in schemas:
        fn = s.get("function", {})
        nm = fn.get("name")
        if nm in HANDLERS:
            overlay.register_tool(nm, fn, HANDLERS[nm])
            count += 1
    return {"registered": count, "available": sorted(HANDLERS.keys())}
