from __future__ import annotations
import json
from typing import Dict, Any
from .security_plugins import (
    guard_prompt_injection_check, secrets_scan,
    RateLimiter, ApprovalGate, MiniScheduler
)

SENSITIVE_TOOLS = {"py.run_sandbox","fs.write","git.ops",
                   "web.browser","screen.capture","clipboard.read","clipboard.write"}

# 전역 인스턴스(필요시 싱글톤/DI로 치환)
rl = RateLimiter()
rl.set_policy("web.fetch_lite", limit=30, window_sec=60)
rl.set_policy("py.run_sandbox", limit=6, window_sec=60)
gate = ApprovalGate()  # UI 연동시 prompt_fn 주입
sched = MiniScheduler(tz="Asia/Seoul", on_fire=lambda t: print("[FIRE]", t.title, t.payload))

def pre_ingest_external(text: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    g = guard_prompt_injection_check(text=text, source=meta.get("source","web"), url=meta.get("url"))
    if g["risk"] == "high":
        resp = gate.ask(
            title="⚠️ 고위험 외부텍스트 감지",
            detail=f"source={meta.get('source')} url={meta.get('url')}\ntriggers={g['triggers']}\n계속 진행할까요?",
            timeout_sec=45
        )
        if not resp["approved"]:
            raise RuntimeError("Blocked by prompt-injection guard")
    return g  # safe_summary를 LLM 컨텍스트에 포함(명령 아님 고지)

def before_tool(tool_name: str, args: Dict[str, Any]):
    ok, retry_after = rl.allow(tool_name)
    if not ok:
        raise RuntimeError(f"Rate limited: retry_after={retry_after:.1f}s")
    if tool_name in SENSITIVE_TOOLS:
        detail = f"{tool_name} 호출\nargs={json.dumps(args)[:400]}"
        resp = gate.ask(title="민감 툴 실행 승인", detail=detail, timeout_sec=45)
        if not resp["approved"]:
            raise RuntimeError("User rejected")

def before_upload(text_or_path: str, is_file: bool=False):
    res = secrets_scan(text=None, file_path=text_or_path, mask=True) if is_file else secrets_scan(text=text_or_path, mask=True)
    if res["findings"]:
        msg = f"시크릿 패턴 {len(res['findings'])}건 발견. 마스킹된 상태로 전송할까요?"
        resp = gate.ask("시크릿 감지", msg, timeout_sec=45)
        if not resp["approved"]:
            raise RuntimeError("Upload blocked by secrets.scan")
    return res  # masked_text 사용 권장

# 스케줄러 도우미
def sched_create(args: Dict[str, Any]): return sched.create(**args)
def sched_cancel(args: Dict[str, Any]): return sched.cancel(args["id"])
def sched_list(args: Dict[str, Any]):   return sched.list()
