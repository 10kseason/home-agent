# tools/tool/kb.py
from __future__ import annotations
import os, json, time, re, math, glob
from pathlib import Path

KB_DIR = Path("agent_output/kb")
KB_DIR.mkdir(parents=True, exist_ok=True)
KB_FILE = KB_DIR / "kb.jsonl"

def _tokenize(s: str):
    return re.findall(r"[A-Za-z0-9가-힣]+", (s or "").lower())

def _chunk(text: str, size=800, overlap=150):
    toks = _tokenize(text)
    i = 0
    while i < len(toks):
        yield " ".join(toks[i:i+size])
        i += max(1, size - overlap)

def kb_ingest(args):
    source = args["source"]
    is_path = bool(args.get("is_path"))
    is_url  = bool(args.get("is_url"))
    tags    = list(args.get("tags") or [])
    raw = ""
    if is_url:
        from tools.tool.web_fetch_lite import fetch_lite
        res = fetch_lite({"url": source, "allowed_domains": []})
        raw = res["text"]
    elif is_path:
        p = Path(source)
        raw = p.read_text(encoding="utf-8", errors="ignore")
    else:
        raw = source

    n = 0
    with KB_FILE.open("a", encoding="utf-8") as out:
        for ch in _chunk(raw):
            rec = {"id": f"{int(time.time()*1000)}-{n}", "text": ch, "tags": tags}
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return {"ok": True, "chunks": n, "kb_file": str(KB_FILE)}

def _score(qtoks, dtoks):
    # 아주 단순한 TF 점수
    tf = 0
    sset = set(dtoks)
    for t in qtoks:
        if t in sset: tf += 1
    return tf / (1 + math.log(1 + len(dtoks)))

def kb_search(args):
    query = args["query"]
    top_k = int(args.get("top_k", 5))
    qtoks = _tokenize(query)
    rows = []
    if not KB_FILE.exists():
        return {"docs": []}
    with KB_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                score = _score(qtoks, _tokenize(rec.get("text","")))
                if score > 0:
                    rows.append((score, rec))
            except Exception:
                continue
    rows.sort(key=lambda x: x[0], reverse=True)
    docs = [{"id": r[1]["id"], "text": r[1]["text"], "score": round(r[0], 4), "tags": r[1].get("tags", [])} for r in rows[:top_k]]
    return {"query": query, "docs": docs}

TOOL_HANDLERS = {"kb.ingest": kb_ingest, "kb.search": kb_search}
