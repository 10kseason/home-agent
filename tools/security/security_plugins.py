from __future__ import annotations
import re, time, json, threading, heapq, uuid
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta, timezone
try:
    from zoneinfo import ZoneInfo  # py>=3.9
except Exception:
    ZoneInfo = None

# ─────────── Prompt-Injection Guard ───────────

INJECTION_PATTERNS = [
    r"(?i)\bignore (all )?(previous|prior) (instructions|messages)\b",
    r"(?i)\bdisregard\b.*\b(system|safety|guardrails)\b",
    r"(?i)\bexfiltrate\b|\bdata leak\b|\bexpose\b.*\b(api|token|cookie|key)\b",
    r"(?i)\bread(?!ing)\b.*\b(localhost|127\.0\.0\.1|file://|C:\\|/home/)\b",
    r"(?i)\bcopy\b.*\bclipboard\b|\bdump\b.*\bsecrets?\b",
    r"(?i)\blogin\b.*\bcredential\b|\bsession\b.*\bcookie\b",
    r"(?i)\bdo not ask\b.*\buser\b|\bwithout confirmation\b",
]

def _score_injection(text: str) -> Tuple[float, List[str]]:
    triggers, score = [], 0.0
    for pat in INJECTION_PATTERNS:
        if re.search(pat, text or ""):
            triggers.append(pat)
            score += 0.25
    if "http://" in text or "https://" in text: score += 0.05
    if len(text) > 4000: score += 0.05
    return min(score, 1.0), triggers

def guard_prompt_injection_check(text: str, source: str="web", url: Optional[str]=None,
                                 allow_tool_use: bool=False) -> Dict[str, Any]:
    score, triggers = _score_injection(text or "")
    risk = "high" if score >= 0.75 else "medium" if score >= 0.35 else "low"
    safe_summary = (text or "")[:500]
    return {"risk": risk, "score": round(score, 2),
            "triggers": triggers, "safe_summary": safe_summary}

# ─────────── Secrets Scanner ───────────

SECRET_REGEXES: Dict[str, str] = {
    "AWS_ACCESS_KEY_ID": r"\bAKIA[0-9A-Z]{16}\b",
    "AWS_SECRET_ACCESS_KEY": r"\b(?<![A-Za-z0-9/+=])[A-Za-z0-9/+=]{40}(?![A-Za-z0-9/+=])\b",
    "GCP_API_KEY": r"\bAIza[0-9A-Za-z\-_]{35}\b",
    "OPENAI_KEY": r"\b(sk|rk)-[A-Za-z0-9]{20,}\b",
    "GITHUB_TOKEN": r"\bghp_[A-Za-z0-9]{36}\b",
    "SLACK_TOKEN": r"\bxox[baprs]-[A-Za-z0-9-]{10,48}\b",
    "JWT": r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b",
}

def _mask(s: str) -> str:
    if len(s) <= 6: return "*" * len(s)
    return s[:3] + "*"*(len(s)-6) + s[-3:]

def secrets_scan(text: Optional[str]=None, file_path: Optional[str]=None, mask: bool=True) -> Dict[str, Any]:
    content, path = text or "", None
    if file_path:
        try:
            with open(file_path, "rb") as f: raw = f.read()
            content = raw.decode("utf-8", errors="ignore"); path = file_path
        except Exception as e:
            content += f"\n[secrets_scan warning: {e}]"
    findings = []
    for name, pat in SECRET_REGEXES.items():
        for m in re.finditer(pat, content or ""):
            val = m.group(0)
            findings.append({"type": name, "value": _mask(val) if mask else val,
                             "span": [m.start(), m.end()], "path": path or ""})
    masked_text = content
    if mask and findings:
        matches = sorted([ (m.start(), m.end(), m.group(0)) for p in SECRET_REGEXES.values()
                           for m in re.finditer(p, content) ], key=lambda x: x[0], reverse=True)
        for s, e, v in matches:
            masked_text = masked_text[:s] + _mask(v) + masked_text[e:]
    return {"findings": findings, "masked_text": masked_text}

# ─────────── Rate Limiter ───────────

@dataclass
class _Bucket:
    limit: int
    window_sec: int
    hits: List[float] = field(default_factory=list)

class RateLimiter:
    def __init__(self):
        self._buckets: Dict[str, _Bucket] = {}
        self._lock = threading.Lock()
    def set_policy(self, key: str, limit: int, window_sec: int):
        with self._lock: self._buckets[key] = _Bucket(limit, window_sec)
    def get_policy(self, key: str) -> Optional[_Bucket]:
        return self._buckets.get(key)
    def allow(self, key: str) -> Tuple[bool, float]:
        now = time.time()
        with self._lock:
            b = self._buckets.get(key)
            if not b: return True, 0.0
            cutoff = now - b.window_sec
            b.hits = [t for t in b.hits if t >= cutoff]
            if len(b.hits) < b.limit:
                b.hits.append(now); return True, 0.0
            retry_after = b.hits[0] + b.window_sec - now
            return False, max(0.0, retry_after)

# ─────────── Approval Gate ───────────

class ApprovalGate:
    """승인 UI 콜백을 주입 가능. 없으면 콘솔 입력 사용."""
    def __init__(self, prompt_fn=None):
        self.prompt_fn = prompt_fn or self._console_prompt
    def _console_prompt(self, title: str, detail: str, timeout_sec: int) -> Tuple[bool, str]:
        print(f"[APPROVAL] {title}\n{detail}\n(y/N, {timeout_sec}s timeout)")
        approved = [None]
        def waiter():
            try: approved[0] = input().strip().lower() in ("y","yes")
            except Exception: approved[0] = False
        t = threading.Thread(target=waiter, daemon=True); t.start(); t.join(timeout=timeout_sec)
        if approved[0] is None: return False, "timeout"
        return bool(approved[0]), "user_input"
    def ask(self, title: str, detail: str, timeout_sec: int=45) -> Dict[str, Any]:
        ok, reason = self.prompt_fn(title, detail, timeout_sec)
        return {"approved": bool(ok), "reason": reason}

# ─────────── Scheduler / Reminder ───────────

@dataclass(order=True)
class _Task:
    next_ts: float
    id: str = field(compare=False)
    title: str = field(compare=False)
    interval_sec: Optional[int] = field(compare=False, default=None)
    rrule: Optional[str] = field(compare=False, default=None)
    payload: Dict[str, Any] = field(compare=False, default_factory=dict)

class MiniScheduler:
    """인메모리 스케줄러: 일회·interval·간단 RRULE(BYHOUR/BYMINUTE)"""
    def __init__(self, tz: str="Asia/Seoul", on_fire=None):
        self.tz = tz
        self.on_fire = on_fire or (lambda task: print(f"[SCHED] {task.id} {task.title} {task.payload}"))
        self._q: List[_Task] = []; self._index: Dict[str, _Task] = {}; self._cv = threading.Condition()
        self._thr = threading.Thread(target=self._run, daemon=True); self._thr.start()
    def _now(self) -> datetime:
        if ZoneInfo and self.tz: return datetime.now(ZoneInfo(self.tz))
        return datetime.now(timezone.utc).astimezone()
    def create(self, title: str, run_at: Optional[str]=None, interval_sec: Optional[int]=None,
               rrule: Optional[str]=None, payload: Optional[Dict[str, Any]]=None) -> Dict[str, Any]:
        now = self._now()
        if run_at: dt = datetime.fromisoformat(run_at)
        elif interval_sec: dt = now + timedelta(seconds=interval_sec)
        elif rrule:
            hh = mm = 0
            for part in rrule.split(";"):
                if part.startswith("BYHOUR="): hh = int(part.split("=")[1])
                if part.startswith("BYMINUTE="): mm = int(part.split("=")[1])
            cand = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
            dt = cand if cand > now else cand + timedelta(days=1)
        else: dt = now + timedelta(seconds=5)
        tid = str(uuid.uuid4())[:8]
        t = _Task(next_ts=dt.timestamp(), id=tid, title=title,
                  interval_sec=interval_sec, rrule=rrule, payload=payload or {})
        with self._cv:
            heapq.heappush(self._q, t); self._index[tid] = t; self._cv.notify()
        return {"id": tid, "next_run": dt.isoformat()}
    def cancel(self, tid: str) -> Dict[str, Any]:
        with self._cv:
            t = self._index.pop(tid, None)
            if not t: return {"ok": False}
            t.next_ts = -1; self._cv.notify()
        return {"ok": True}
    def list(self) -> Dict[str, Any]:
        with self._cv:
            items = [ {"id": t.id, "title": t.title, "next_run": datetime.fromtimestamp(t.next_ts).isoformat(),
                       "interval_sec": t.interval_sec, "rrule": t.rrule, "payload": t.payload}
                     for t in self._q if t.next_ts > 0 ]
        return {"items": items}
    def _reschedule(self, t: _Task):
        now = self._now()
        if t.interval_sec: t.next_ts = (now + timedelta(seconds=t.interval_sec)).timestamp()
        elif t.rrule:
            hh = mm = 0
            for part in t.rrule.split(";"):
                if part.startswith("BYHOUR="): hh = int(part.split("=")[1])
                if part.startswith("BYMINUTE="): mm = int(part.split("=")[1])
            nxt = now.replace(hour=hh, minute=mm, second=0, microsecond=0) + timedelta(days=1)
            t.next_ts = nxt.timestamp()
        else: t.next_ts = -1
    def _run(self):
        while True:
            with self._cv:
                while not self._q: self._cv.wait()
                t = heapq.heappop(self._q)
                if t.next_ts < 0: continue
                wait = max(0.0, t.next_ts - self._now().timestamp())
            if wait > 0: time.sleep(wait)
            try: self.on_fire(t)
            except Exception as e: print(f"[SCHED error] {e}")
            self._reschedule(t)
            with self._cv:
                if t.next_ts > 0: heapq.heappush(self._q, t)
