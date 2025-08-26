from typing import Dict, Any
from .base import ToolPlugin, ToolContext

class STT(ToolPlugin):
    name = "stt"
    description = "오디오에서 텍스트로 변환"
    input_schema = {
        "type": "object",
        "properties": {
            "audio_path": {"type": "string", "description": "오디오 파일 경로"}
        },
        "required": ["audio_path"],
    }

    def run(self, args: Dict[str, Any], ctx: ToolContext) -> Dict[str, Any]:
        return self.fail("not implemented")
