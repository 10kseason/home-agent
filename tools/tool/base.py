from __future__ import annotations
import abc
import json
import time
from typing import Any, Dict, Optional

try:
    import jsonschema  # optional
except Exception:
    jsonschema = None


class ToolContext:
    """
    실행 컨텍스트: logger/config/event_bus 등을 주입.
    - logger: .info/.warning/.error 메서드를 가진 로거(없으면 None 허용)
    - config: dict 형태의 전역 설정(없으면 {})
    - event_bus: callable(event_name:str, payload:dict) 형태(없으면 람다)
    """
    def __init__(self, logger=None, config: Optional[Dict[str, Any]] = None, event_bus=None):
        self.logger = logger
        self.config = config or {}
        self.event_bus = event_bus or (lambda event, payload: None)

    def log(self, level: str, msg: str):
        if self.logger:
            getattr(self.logger, level, self.logger.info)(msg)


class ToolPlugin(abc.ABC):
    """모든 플러그인의 공통 인터페이스"""
    name: str = "tool"
    description: str = ""
    input_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {},
        "additionalProperties": True,
    }

    def validate(self, args: Dict[str, Any]):
        if jsonschema is None:
            return  # best-effort: jsonschema 미설치 시 스킵
        try:
            jsonschema.validate(args, self.input_schema)
        except Exception as e:
            raise ValueError(f"[{self.name}] schema validation failed: {e}")

    @abc.abstractmethod
    def run(self, args: Dict[str, Any], ctx: ToolContext) -> Dict[str, Any]:
        """동기 실행. 결과는 JSON-직렬화 가능한 dict."""
        raise NotImplementedError

    # 선택: 공용 유틸
    def ok(self, **data) -> Dict[str, Any]:
        return {"tool": self.name, "ok": True, "data": data, "ts": time.time()}

    def fail(self, message: str, **data) -> Dict[str, Any]:
        return {"tool": self.name, "ok": False, "error": message, "data": data, "ts": time.time()}
