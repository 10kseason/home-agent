# -*- coding: utf-8 -*-
"""
tools.tool — runtime tool handlers matching Overlay/plugins/tools/config/tools.yaml
"""
from __future__ import annotations
import re, os, json, time, uuid
from pathlib import Path
from typing import Dict, Any, List, Callable, Tuple
import httpx

# --- simple <think> tag remover (translation tool uses) ---
THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

def _strip_think(text: str) -> str:
    """Remove <think> blocks from text"""
    if not text:
        return ""
    return THINK_RE.sub("", text)

# ---------- registry ----------
HANDLERS: Dict[str, Callable[[Dict[str, Any]], Any]] = {}

def register(name: str):
    def _wrap(fn: Callable[[Dict[str, Any]], Any]):
        HANDLERS[name] = fn
        return fn
    return _wrap

# ---------- utils ----------
ROOT = Path(__file__).resolve().parents[1]  # .../home-agent
OUT_DIR = ROOT / "agent_output"
KB_DIR  = OUT_DIR / "kb"
for d in (OUT_DIR, KB_DIR):
    d.mkdir(parents=True, exist_ok=True)

def _redact(text: str) -> tuple[str, List[str]]:
    if not text:
        return "", []
    triggers = []
    patterns = [
        (r"sk-[A-Za-z0-9]{20,}", "OpenAIKey"),
        (r"AKIA[0-9A-Z]{16}", "AWSKey"),
        (r"(?i)password\s*[:=]\s*\S+", "Password"),
        (r"[0-9a-f]{32,64}", "HexSecret"),
        (r"[A-Za-z0-9_.+-]+@[A-Za-z0-9-]+\.[A-Za-z0-9-.]+", "Email"),
        (r"(0\d{1,2}[- ]?\d{3,4}[- ]?\d{4})", "KRPhone"),
    ]
    red = text
    for pat, label in patterns:
        if re.search(pat, red):
            triggers.append(label)
            red = re.sub(pat, f"<{label}>", red)
    return red, triggers

def _risk_score(text: str) -> float:
    score = 0.0
    if not text: return 0.0
    tf = [
        ("ignore previous", 0.25),
        ("disregard.*instructions", 0.25),
        ("system prompt", 0.2),
        ("resets? (the )?instructions", 0.2),
        ("```", 0.05),
        ("<script", 0.2),
        ("file://", 0.15),
        ("bash -c", 0.2),
        ("powershell", 0.2),
        ("curl http", 0.1),
        ("http://127.0.0.1|localhost", 0.2),
    ]
    low = text.lower()
    for pat, w in tf:
        if re.search(pat, low):
            score += w
    return max(0.0, min(1.0, score))

def _risk_label(score: float) -> str:
    if score >= 0.6: return "high"
    if score >= 0.3: return "medium"
    return "low"

# ---------- guard.prompt_injection_check ----------
@register("guard.prompt_injection_check")
def guard_prompt_injection_check(args: Dict[str, Any]):
    text = str(args.get("text","") or "")
    red, trig = _redact(text)
    score = _risk_score(text)
    return {
        "risk": _risk_label(score),
        "score": round(score, 3),
        "triggers": trig,
        "safe_summary": (red[:1000] if red else "")
    }

# ---------- sched.* (file-backed lightweight impl) ----------
SCHED_DB = OUT_DIR / "sched.jsonl"
def _sched_write(rec: Dict[str, Any]):
    with SCHED_DB.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False)+"\n")
def _sched_read_all() -> List[Dict[str, Any]]:
    if not SCHED_DB.exists(): return []
    return [json.loads(ln) for ln in SCHED_DB.read_text(encoding="utf-8").splitlines() if ln.strip()]

@register("sched.create")
def sched_create(args: Dict[str, Any]):
    rid = args.get("id") or str(uuid.uuid4())[:8]
    rec = {
        "id": rid,
        "title": args.get("title") or "",
        "time": args.get("time") or "NOW",
        "every": args.get("every") or "",
        "message": args.get("message") or "",
        "created": int(time.time())
    }
    _sched_write({"op":"create", **rec})
    return {"ok": True, **rec}

@register("sched.cancel")
def sched_cancel(args: Dict[str, Any]):
    rid = args.get("id") or ""
    title = args.get("title") or ""
    _sched_write({"op":"cancel","id":rid,"title":title,"ts":int(time.time())})
    return {"ok": True, "id": rid, "title": title}

@register("sched.list")
def sched_list(args: Dict[str, Any]):
    return {"ok": True, "items": _sched_read_all()[-100:]}

# ---------- security.secrets_scan ----------
@register("security.secrets_scan")
def secrets_scan(args: Dict[str, Any]):
    text = str(args.get("text","") or "")
    red, trig = _redact(text)
    return {"ok": True, "triggers": trig, "masked_text": red}

# ---------- rate.limit ----------
_RATE_DB = OUT_DIR / "ratelimit.json"
def _rl_load():
    if not _RATE_DB.exists(): return {}
    try: return json.loads(_RATE_DB.read_text(encoding="utf-8"))
    except: return {}
def _rl_save(data):
    _RATE_DB.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

@register("rate.limit")
def rate_limit(args: Dict[str, Any]):
    key = args.get("key") or "default"
    per_min = int(args.get("per_min", 5))
    data = _rl_load()
    now_min = int(time.time() // 60)
    ent = data.get(key) or {"min": now_min, "count":0}
    if ent["min"] != now_min:
        ent = {"min": now_min, "count":0}
    ent["count"] += 1
    allowed = ent["count"] <= per_min
    data[key] = ent
    _rl_save(data)
    return {"allowed": bool(allowed), "used": ent["count"], "limit": per_min, "reset_in_sec": (60 - int(time.time()%60))}

# ---------- approval.gate ----------
APPROVAL_DB = OUT_DIR / "approvals.jsonl"
@register("approval.gate")
def approval_gate(args: Dict[str, Any]):
    action = args.get("action") or ""
    approver = args.get("approver") or ""
    approve = bool(args.get("approve", False))
    rec = {"action": action, "approver": approver, "approve": approve, "ts": int(time.time())}
    with APPROVAL_DB.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False)+"\n")
    status = "approved" if approve else "pending"
    return {"status": status, **rec}

# ---------- agent.list_tools ----------
@register("agent.list_tools")
def agent_list_tools(args: Dict[str, Any]):
    cfg = ROOT / "tools" / "config" / "tools.yaml"
    items: List[Dict[str, Any]] = []
    schemas: List[Dict[str, Any]] = []
    if cfg.exists():
        try:
            import yaml  # type: ignore
            data = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
            raw_items = data.get("tools") if isinstance(data, dict) else data
            for it in (raw_items or []):
                if not isinstance(it, dict):
                    continue
                nm = it.get("name")
                if not nm:
                    continue
                desc = it.get("desc", "")
                params = it.get("input_schema") or {"type": "object", "properties": {}}
                items.append({
                    "type": "function",
                    "function": {
                        "name": nm,
                        "description": desc,
                        "parameters": params,
                    },
                })
                schemas.append({"name": nm, "desc": desc, "parameters": params})
        except Exception as e:
            return {"ok": False, "error": f"yaml_parse_failed: {e}"}
    return {
        "ok": True,
        "tools": items,
        "schemas": schemas,
        "implemented": sorted(list(HANDLERS.keys())),
    }

# ---------- web.fetch_lite ----------
@register("web.fetch_lite")
def web_fetch_lite(args: Dict[str, Any]):
    url = args.get("url") or args.get("href")
    if not url: return {"ok": False, "error": "missing url"}
    try:
        r = httpx.get(url, timeout=10.0, headers={"User-Agent":"LunaOverlay/1.0"})
        text = r.text[:4000]
        m = re.search(r"<title>(.*?)</title>", text, re.I|re.S)
        title = (m.group(1).strip() if m else "")
        return {"ok": True, "status": r.status_code, "title": title, "length": len(r.text), "sample": text[:500]}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ---------- kb.ingest ----------
@register("kb.ingest")
def kb_ingest(args: Dict[str, Any]):
    text = args.get("text") or ""
    if not text: return {"ok": False, "error": "missing text"}
    rid = str(uuid.uuid4())[:8]
    path = KB_DIR / f"{rid}.txt"
    path.write_text(text, encoding="utf-8")
    return {"ok": True, "id": rid, "path": str(path)}

# ---------- kb.search ----------
@register("kb.search")
def kb_search(args: Dict[str, Any]):
    query = args.get("query") or ""
    top_k = int(args.get("top_k") or 5)
    if not query: return {"ok": False, "error": "missing query"}
    results = []
    for fp in KB_DIR.glob("*.txt"):
        try:
            t = fp.read_text(encoding="utf-8", errors="ignore")
            score = sum(t.lower().count(q.strip().lower()) for q in query.split() if q.strip())
            if score > 0:
                results.append({"path": str(fp), "score": float(score), "preview": t[:160]})
        except: pass
    results.sort(key=lambda x: x["score"], reverse=True)
    return {"ok": True, "items": results[:top_k]}

# ---------- translation (LLM-based) ----------
@register("translation")
def translation_tool(args: Dict[str, Any]):
    text = (args.get("text") or "").strip()
    target = (args.get("target") or args.get("to") or "ko").strip()
    if not text:
        return {"ok": False, "error": "missing text"}

    endpoint = model = api_key = ""
    try:  # load from agent/config.yaml if available
        import yaml  # type: ignore
        cfg = yaml.safe_load((ROOT / "agent" / "config.yaml").read_text(encoding="utf-8")) or {}
        trans = cfg.get("translate") or {}
        endpoint = trans.get("endpoint") or ""
        model = trans.get("model") or ""
        api_key = trans.get("api_key") or ""
    except Exception:
        pass
    endpoint = os.environ.get("TRANSLATE_ENDPOINT", endpoint)
    model = os.environ.get("TRANSLATE_MODEL", model)
    api_key = os.environ.get("TRANSLATE_API_KEY", api_key)
    if not endpoint or not model:
        return {"ok": False, "error": "translate_not_configured"}

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    prompt = f"Translate to {target}. If already in {target}, return original. Text:\n{text}"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful translator."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }
    try:
        timeout = httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=30.0)
        r = httpx.post(f"{endpoint}/chat/completions", headers=headers, json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        translated = _strip_think((data.get("choices", [{}])[0].get("message", {}).get("content") or "").strip())
        return {"ok": True, "translation": translated}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ---------- tool.list (static list in Korean) ----------
@register("tool.list")
def tool_list(args: Dict[str, Any]):
    items = [
        "Translation – 주어진 문장을 지정한 언어로 번역합니다.",
        "DDG Search – DuckDuckGo를 사용하여 안전하게 웹 검색을 수행합니다.",
        "OCR 8B – 이미지 속 텍스트를 추출하고, 한국어 텍스트만 별도로 반환합니다(고속 모드 제외).",
    ]
    return {"ok": True, "tools": items}

# ---------- file.watch (stub) ----------
@register("file.watch")
def file_watch(args: Dict[str, Any]):
    return {"ok": False, "error": "file.watch not supported in lightweight handler"}

# ---------- mail.* (stubs for safety) ----------
@register("mail.imap_list")
def mail_imap_list(args: Dict[str, Any]):
    return {"ok": False, "error": "IMAP listing disabled in this build"}

@register("mail.smtp_send")
def mail_smtp_send(args: Dict[str, Any]):
    return {"ok": False, "error": "SMTP send disabled in this build"}

# ---------- sched.persist (Windows schtasks PREVIEW ONLY) ----------
@register("sched.persist")
def sched_persist(args: Dict[str, Any]):
    title = (args.get("title") or "").strip()
    delete = bool(args.get("delete", False))
    every = (args.get("every") or "").upper()
    time_s = args.get("time") or "NOW"
    cmd = args.get("cmd") or ""
    if not title:
        return {"ok": False, "error": "title required"}
    if delete:
        return {"ok": True, "preview": f'schtasks /Delete /TN "{title}" /F', "note": "preview only; not executed"}
    else:
        if not cmd:
            return {"ok": False, "error": "cmd required"}
        if every.startswith("MINUTE:"):
            minutes = every.split(":",1)[1]
            sc = f"/SC MINUTE /MO {minutes}"
        elif every == "HOURLY":
            sc = "/SC HOURLY"
        else:
            sc = "/SC DAILY"
        tpart = f'/ST {time_s}' if time_s and time_s != "NOW" else ""
        preview = f'schtasks /Create /TN "{title}" {sc} {tpart} /TR "{cmd}" /F'
        return {"ok": True, "preview": preview, "note": "preview only; not executed"}
