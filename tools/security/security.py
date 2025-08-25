# 이미 있으면 유지. 없으면 아래처럼 최소 스텁.
import re

def safe_path(p: str) -> bool:
    return not bool(re.search(r"[<>:\"|?*]", p))
