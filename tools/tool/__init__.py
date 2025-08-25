# tools/tool/__init__.py
from __future__ import annotations
import importlib, pkgutil

# 모든 모듈의 TOOL_HANDLERS 딕셔너리를 합쳐서 노출
HANDLERS = {}

for m in pkgutil.iter_modules(__path__):
    if m.name.startswith("_"):  # 내부 모듈 스킵
        continue
    mod = importlib.import_module(f"{__name__}.{m.name}")
    if hasattr(mod, "TOOL_HANDLERS") and isinstance(mod.TOOL_HANDLERS, dict):
        for k, v in mod.TOOL_HANDLERS.items():
            HANDLERS[k] = v
