from typing import Dict, Any
from .base import ToolPlugin, ToolContext

class MicTrans(ToolPlugin):
    """Forward microphone transcription text into the agent event bus."""
    name = "mictrans"
    description = "Mic transcription forwarding"
    input_schema = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "transcribed text"},
        },
        "required": ["text"],
    }

    def run(self, args: Dict[str, Any], ctx: ToolContext) -> Dict[str, Any]:
        ctx.event_bus("mictrans.text", {"text": args["text"]})
        return self.ok()
