from typing import Dict, Any
from .base import ToolPlugin, ToolContext

class OCR(ToolPlugin):
    name = "ocr"
    description = "이미지에서 텍스트를 추출"
    input_schema = {
        "type": "object",
        "properties": {
            "image_path": {"type": "string", "description": "이미지 파일 경로"}
        },
        "required": ["image_path"],
    }

    def run(self, args: Dict[str, Any], ctx: ToolContext) -> Dict[str, Any]:
        return self.fail("not implemented")
