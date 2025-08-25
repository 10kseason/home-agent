from __future__ import annotations
import importlib
import pkgutil
import yaml
from typing import Dict, Any, Optional, Type
from .base import ToolPlugin, ToolContext


class ToolRegistry:
    def __init__(self, ctx: Optional[ToolContext] = None):
        self._tools: Dict[str, ToolPlugin] = {}
        self._ctx = ctx or ToolContext()

    @classmethod
    def from_yaml(cls, yaml_path: str, ctx: Optional[ToolContext] = None) -> "ToolRegistry":
        reg = cls(ctx=ctx)
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        for item in (cfg.get("tools") or []):
            if not item.get("enabled", True):
                continue
            name = item["name"]
            module = item["module"]
            klass = item["class"]
            default_args = item.get("default_args") or {}
            plugin = reg._load_plugin(module, klass)
            plugin.name = name  # 강제 이름 지정(파일 클래스명과 분리)
            # 기본 인자 보관(필요 시 사용)
            setattr(plugin, "_default_args", default_args)
            reg._tools[name] = plugin
        return reg

    def _load_plugin(self, module_path: str, class_name: str) -> ToolPlugin:
        mod = importlib.import_module(module_path)
        plugin_cls: Type[ToolPlugin] = getattr(mod, class_name)
        if not issubclass(plugin_cls, ToolPlugin):
            raise TypeError(f"{module_path}.{class_name} is not a ToolPlugin")
        return plugin_cls()

    def list_tools(self) -> Dict[str, Any]:
        out = {}
        for name, plugin in self._tools.items():
            out[name] = {
                "description": getattr(plugin, "description", ""),
                "input_schema": getattr(plugin, "input_schema", {}),
            }
        return out

    def invoke(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if name not in self._tools:
            return {"ok": False, "error": f"tool not found: {name}", "tool": name}
        plugin = self._tools[name]
        try:
            plugin.validate(args or {})
            return plugin.run(args or {}, self._ctx)
        except Exception as e:
            self._ctx.log("error", f"[tool:{name}] {e}")
            return {"ok": False, "tool": name, "error": str(e)}
