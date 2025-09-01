from __future__ import annotations
import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

from loguru import logger

try:
    import jsonschema
except Exception:  # pragma: no cover - best effort if jsonschema missing
    jsonschema = None

from .schemas import Event

# Load tool definitions and derive JSON Schemas
TOOLS_PATH = Path(__file__).resolve().parent.parent / "schemas" / "tool_calls.json"
with TOOLS_PATH.open("r", encoding="utf-8") as f:
    TOOLS: List[Dict[str, Any]] = json.load(f)
SCHEMAS: Dict[str, Dict[str, Any]] = {
    t["function"]["name"]: t["function"]["parameters"] for t in TOOLS
}

# Tool name to event mapping
TOOL_ROUTES = {
    "stt_assist_start": lambda args, ctx: [
        ("assist.on", {"reason": "tool", "args": args}),
        ("stt_assist.start", {"reason": "tool"}),
    ],
    "ocr_capture": lambda args, ctx: [
        ("ocr_assist.start", {"region": args.get("region")})
    ],
    "assist_on": lambda args, ctx: [("assist.on", {"reason": "tool"})],
    "assist_off": lambda args, ctx: [
        ("assist.off", {"reason": "tool"}),
        ("stt.stop", {"reason": "tool"}),
        ("ocr.stop", {"reason": "tool"}),
    ],
    "stt_start": lambda args, ctx: [
        (
            "stt_assist.start" if getattr(ctx, "assist_mode", False) else "stt.start",
            {"reason": "tool"},
        )
    ],
    "stt_stop": lambda args, ctx: [("stt.stop", {"reason": "tool"})],
    "ocr_start": lambda args, ctx: [
        (
            "ocr_assist.start" if getattr(ctx, "assist_mode", False) else "ocr.start",
            {"reason": "tool"},
        )
    ],
    "ocr_stop": lambda args, ctx: [("ocr.stop", {"reason": "tool"})],
}


def _validate(name: str, args: Dict[str, Any]):
    schema = SCHEMAS.get(name, {"type": "object"})
    if jsonschema is None:
        return
    jsonschema.validate(args, schema)


async def handle_tool_calls(tool_calls: List[Dict[str, Any]], ctx, bus) -> None:
    """Validate and route tool calls to the event bus."""
    for tc in tool_calls or []:
        if not isinstance(tc, dict):
            continue
        func = tc.get("function", tc)
        name = func.get("name")
        raw_args = func.get("arguments", func.get("args") or {})
        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args or "{}")
            except Exception:
                logger.warning(f"[tool-gw] invalid JSON args for {name}")
                args = {}
        else:
            args = raw_args or {}

        _validate(name, args)
        route = TOOL_ROUTES.get(name)
        if not route:
            logger.warning(f"[tool-gw] no route for {name}")
            continue

        events = route(args, ctx)
        corr = tc.get("correlation_id") or str(uuid.uuid4())
        idem = tc.get("idempotency_key")
        priority = tc.get("priority", 2)
        now = time.time()
        for etype, payload in events:
            payload = dict(payload)
            payload.update({"ts": now, "correlation_id": corr, "tool": name})
            if idem:
                payload["idempotency_key"] = idem
            await bus.publish(
                Event(
                    type=etype,
                    payload=payload,
                    priority=priority,
                    source="tool-gw",
                    timestamp=now,
                )
            )


async def process_response(resp: Dict[str, Any], ctx, bus) -> None:
    """Handle LLM response dict with optional fallback for legacy JSON."""
    tool_calls = resp.get("tool_calls")
    if not tool_calls and resp.get("content"):
        try:
            parsed = json.loads(resp["content"])
        except Exception:
            return
        if "tool_calls" in parsed:
            logger.warning("[tool-gw] deprecated JSON content tool call path")
            tool_calls = parsed.get("tool_calls")
    if tool_calls:
        await handle_tool_calls(tool_calls, ctx, bus)
