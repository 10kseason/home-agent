# tools/tool/security_tools.py
from __future__ import annotations
import json, sys
from typing import Dict, Any
from tools.security.orchestrator_hooks import (
    guard_prompt_injection_check, secrets_scan, rl, gate,
    sched_create, sched_cancel, sched_list
)

def _guard_prompt_injection_check(args: Dict[str, Any]):
    return guard_prompt_injection_check(**args)

def _secrets_scan(args: Dict[str, Any]):
    return secrets_scan(**args)

def _rate_limit(args: Dict[str, Any]):
    key = args["key"]
    limit = args.get("limit")
    window = args.get("window_sec")
    get_only = bool(args.get("get_only"))
    if not get_only and (limit is not None or window is not None):
        cur = rl.get_policy(key)
        if cur:
            if limit is None:  limit = cur.limit
            if window is None: window = cur.window_sec
        rl.set_policy(key, int(limit or 0), int(window or 0))
    pol = rl.get_policy(key)
    if not pol:
        return {"key": key, "limit": 0, "window_sec": 0}
    return {"key": key, "limit": pol.limit, "window_sec": pol.window_sec}

def _approval_gate(args: Dict[str, Any]):
    return gate.ask(args["title"], args["detail"], int(args.get("timeout_sec", 45)))

TOOL_HANDLERS = {
    "guard.prompt_injection_check": _guard_prompt_injection_check,
    "security.secrets_scan": _secrets_scan,
    "rate.limit": _rate_limit,
    "approval.gate": _approval_gate,
    "sched.create": sched_create,
    "sched.cancel": sched_cancel,
    "sched.list": sched_list,
}

if __name__ == "__main__":
    payload = json.loads(sys.stdin.read() or "{}")
    name = payload.get("name")
    args = payload.get("args", {})
    try:
        fn = TOOL_HANDLERS[name]
        print(json.dumps(fn(args), ensure_ascii=False))
    except Exception as e:
        print(json.dumps({"error": str(e)}, ensure_ascii=False))
