from typing import Dict, Any
from .base import ToolPlugin, ToolContext

class Echo(ToolPlugin):
    name = "echo"
    description = "에코/디버그: 입력을 그대로 반환"
    input_schema = {
        "type": "object",
        "properties": {
            "message": {"type": "string"},
            "meta": {"type": "object"}
        },
        "required": ["message"],
        "additionalProperties": True
    }

    def run(self, args: Dict[str, Any], ctx: ToolContext) -> Dict[str, Any]:
        ctx.event_bus("tool.echo", {"args": args})
        return self.ok(echo=args)
