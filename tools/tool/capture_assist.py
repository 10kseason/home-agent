from typing import Dict, Any
from .base import ToolPlugin, ToolContext

class CaptureAssist(ToolPlugin):
    """Forward capture assist text into the agent event bus."""
    name = "capture_assist"
    description = "Capture assist text forwarding"
    input_schema = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "text to forward"},
        },
        "required": ["text"],
    }

    def run(self, args: Dict[str, Any], ctx: ToolContext) -> Dict[str, Any]:
        ctx.event_bus("capture_assist.text", {"text": args["text"]})
        return self.ok()
