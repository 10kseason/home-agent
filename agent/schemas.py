from pydantic import BaseModel, Field
from typing import Any, Dict, Optional, List
import time
from pydantic import BaseModel, Field
from typing import Any, Dict, Optional

class PluginEventIn(BaseModel):
    type: str = Field(..., description="예: 'web.search.result'")
    payload: Dict[str, Any] = Field(default_factory=dict)
    priority: Optional[int] = None  # 없으면 서버에서 5로
    source: Optional[str] = None    # 없으면 서버가 'plugin' 혹은 path param으로 채움
    # timestamp 는 받으면 쓰고, 없으면 서버가 생성

class Event(BaseModel):
    type: str = Field(..., description="e.g., 'ocr.text', 'stt.text', 'discord.batch'")
    payload: Dict[str, Any] = Field(default_factory=dict)
    priority: int = Field(default=5)  # 1(high) .. 10(low)
    timestamp: float = Field(default_factory=lambda: time.time())
    source: Optional[str] = None

class Result(BaseModel):
    ok: bool = True
    message: str = ""
    data: Optional[Dict[str, Any]] = None
