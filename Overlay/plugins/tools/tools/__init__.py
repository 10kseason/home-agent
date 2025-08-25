# tools/tool/__init__.py
from __future__ import annotations
import importlib, pkgutil

HANDLERS = {}  # {"tool.name": callable(args_dict)->dict}

for m in pkgutil.iter_modules(__path__):
    if m.name.startswith(("_",)):  # 내부 모듈 스킵
        continue
    mod = importlib.import_module(f"{__name__}.{m.name}")
    d = getattr(mod, "TOOL_HANDLERS", None)
    if isinstance(d, dict):
        HANDLERS.update(d)
