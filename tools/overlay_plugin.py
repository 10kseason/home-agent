from __future__ import annotations
from typing import Dict, Any, Callable, List
from pathlib import Path

from .tool.runtime_registry import ToolRegistry


class YamlToolsPlugin:
    """Expose YAML-defined tools from tools/config/tools.yaml as overlay plugin."""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.name = self.__class__.__name__
        self.enabled = True
        self.handlers: Dict[str, Callable[[Dict[str, Any]], Any]] = {}
        self._setup()

    def _setup(self) -> None:
        cfg_path = Path(__file__).resolve().parent / "config" / "tools.yaml"
        self.registry = ToolRegistry.from_yaml(str(cfg_path))
        for name in self.registry.schemas.keys():
            self.handlers[name] = self._make_handler(name)

    def _make_handler(self, name: str) -> Callable[[Dict[str, Any]], Any]:
        def _handler(args: Dict[str, Any]) -> Any:
            return self.registry.execute(name, args)
        return _handler

    def get_handlers(self) -> Dict[str, Callable[[Dict[str, Any]], Any]]:
        return self.handlers

    def get_tools_schema(self) -> List[Dict[str, Any]]:
        return self.registry.tools_array()

    def on_event(self, event_type: str, payload: Dict[str, Any]) -> bool:
        return False

    def cleanup(self) -> None:
        pass

