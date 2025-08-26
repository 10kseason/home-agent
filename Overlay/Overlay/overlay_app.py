#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Luna Overlay v9 – 견고한 이벤트 수신 및 처리 버전
"""
import yaml
import httpx
import uvicorn
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi import FastAPI, Request
from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import os
import json
import threading
import time
import asyncio
import traceback
import re
import subprocess
import platform
import datetime
import base64
import webbrowser
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

# ---------------- logging ----------------
try:
    from loguru import logger
except Exception:
    class _Dummy:
        def __getattr__(self, name):
            def _p(*a, **k): print(f"[overlay:{name}]", a[0] if a else "")
            return _p
    logger = _Dummy()

# 로그 롤링 설정 (10MB 회전, 14일 보관, zip 압축)
APP_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(APP_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
try:
    logger.remove()  # 기본 stderr 핸들 제거 (원하면 주석 처리)
except Exception:
    pass
logger.add(
    os.path.join(LOG_DIR, "overlay_{time}.log"),
    rotation="10 MB",
    retention="14 days",
    compression="zip",
    enqueue=True,          # 프로세스-세이프 ㅡ
    backtrace=True,        # 예외 backtrace
    diagnose=False,        # 민감정보 노출 방지
    level="INFO"
)

# ---------------- Event Processing ----------------

@dataclass
class EventStats:
    total_received: int = 0
    by_type: Dict[str, int] = field(default_factory=dict)
    last_event_time: float = 0
    errors: int = 0

class EventHandler:
    """이벤트 처리 전담 클래스"""
    
    def __init__(self, window):
        self.window = window
        self.stats = EventStats()
        self.debug_mode = False
        
    def set_debug_mode(self, enabled: bool):
        self.debug_mode = enabled
        
    def handle_event(self, event_type: str, payload: Dict[str, Any]) -> bool:
        """이벤트 처리 메인 로직"""
        try:
            self.stats.total_received += 1
            self.stats.by_type[event_type] = self.stats.by_type.get(event_type, 0) + 1
            self.stats.last_event_time = time.time()
            
            if self.debug_mode:
                logger.info(f"[event] Processing {event_type}: {payload}")
            
            # 이벤트 타입별 처리
            success = False
            if event_type.startswith("stt."):
                success = self._handle_stt(payload)
            elif event_type.startswith("ocr."):
                success = self._handle_ocr(payload)
            elif event_type.startswith("web.search"):
                success = self._handle_web_search(payload)
            elif event_type.startswith("llm.") or event_type.startswith("lm."):
                success = self._handle_llm(payload)
            elif event_type.startswith("overlay."):
                success = self._handle_overlay_event(event_type, payload)
            else:
                success = self._handle_generic(event_type, payload)
                
            if not success:
                self.stats.errors += 1
                
            return success
            
        except Exception as e:
            self.stats.errors += 1
            logger.error(f"[event] Handler error for {event_type}: {e}")
            if self.debug_mode:
                logger.exception("Full traceback:")
            return False
    
    def _emit_safe(self, who: str, msg: str):
        """스레드 안전한 메시지 출력"""
        try:
            if self.window and hasattr(self.window, 'appended'):
                self.window.appended.emit(who, msg)
            else:
                logger.warning(f"[event] No window to emit to: [{who}] {msg}")
        except Exception as e:
            logger.error(f"[event] Emit error: {e}")
    
    def _handle_stt(self, payload: Dict[str, Any]) -> bool:
        """STT 이벤트 처리"""
        text = payload.get("text", "").strip()
        if not text:
            return False
            
        # 기본 텍스트
        display_text = text
        
        # 번역이 있으면 추가
        translation = payload.get("translation", "")
        if translation and translation.strip() != text:
            display_text += f"\n🔄 {translation}"
        
        # 신뢰도 표시
        confidence = payload.get("confidence", 0)
        if confidence > 0 and confidence < 0.8:
            display_text += f" ({confidence:.0%})"
            
        # 언어 정보
        language = payload.get("language", "")
        if language:
            display_text += f" [{language}]"

        # 사용된 STT 모델 정보
        model = payload.get("model", "")
        if model:
            display_text += f" ({model})"

        self._emit_safe("🎤 STT", display_text)
        return True
    
    def _handle_ocr(self, payload: Dict[str, Any]) -> bool:
        """OCR 이벤트 처리"""
        text = payload.get("text", "") or payload.get("ocr", "")
        if not text:
            return False
            
        text = text.strip()
        display_text = text
        
        # 좌표 정보
        bbox = payload.get("bbox", [])
        if bbox and len(bbox) >= 2:
            display_text += f" @({bbox[0]:.0f},{bbox[1]:.0f})"
        
        # 신뢰도
        confidence = payload.get("confidence", 0)
        if confidence > 0:
            display_text += f" ({confidence:.0%})"
            
        self._emit_safe("👁️ OCR", display_text)
        return True
    
    def _handle_web_search(self, payload: Dict[str, Any]) -> bool:
        """웹검색 결과 처리"""
        query = payload.get("query", "")
        items = payload.get("items", [])
        provider = payload.get("provider", "")
        
        if not query:
            return False
        
        # 검색 헤더
        header = f"'{query}'"
        if provider:
            header += f" ({provider})"
        if items:
            header += f" - {len(items)}개 결과"
            
        self._emit_safe("🔍 검색", header)
        
        # 상위 3개 결과 표시
        for i, item in enumerate(items[:3], 1):
            title = item.get("title", "")[:60]
            snippet = item.get("snippet", "")
            url = item.get("url", "")
            
            result_text = f"{i}. {title}"
            if snippet:
                snippet = snippet[:80] + ("..." if len(snippet) > 80 else "")
                result_text += f"\n   {snippet}"
                
            self._emit_safe(f"   결과{i}", result_text)
        
        return True
    
    def _handle_llm(self, payload: Dict[str, Any]) -> bool:
        """LLM 출력 처리"""
        model = payload.get("model", "")
        text = payload.get("text", "") or payload.get("content", "")
        
        if not text:
            return False
        
        # 모델별 레이블
        label = "🤖 LLM"
        model_lower = model.lower()
        if "4b" in model_lower:
            label = "🤖 LLM-4B"
        elif "8b" in model_lower:
            label = "🤖 LLM-8B"
        elif "14b" in model_lower:
            label = "🤖 LLM-14B"
        elif "20b" in model_lower:
            label = "🤖 LLM-20B"
        elif "1.5b" in model_lower:
            label = "🤖 LLM-1.5B"
            
        # 텍스트 길이 제한
        # 추론 모델 출력에서 <think>...</think> 구간 제거
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)

        display_text = text.strip()
        if len(display_text) > 200:
            display_text = display_text[:197] + "..."
        
        # 토큰 수 표시
        tokens = payload.get("tokens", 0)
        if tokens > 0:
            display_text += f" ({tokens}tok)"
            
        self._emit_safe(label, display_text)
        return True
    
    def _handle_overlay_event(self, event_type: str, payload: Dict[str, Any]) -> bool:
        """오버레이 전용 이벤트 처리"""
        if event_type == "overlay.toast":
            title = payload.get("title", "알림")
            text = payload.get("text", "")
            if text:
                self._emit_safe(f"📢 {title}", text)
                return True
        return False
    
    def _handle_generic(self, event_type: str, payload: Dict[str, Any]) -> bool:
        """일반 이벤트 처리"""
        # 이벤트 타입을 읽기 좋게 변환
        display_type = event_type.replace(".", " ").title()
        
        # 페이로드에서 표시할 텍스트 찾기
        text_fields = ["text", "message", "content", "description", "summary"]
        display_text = ""
        
        for field in text_fields:
            if field in payload and payload[field]:
                display_text = str(payload[field])[:200]
                break
        
        if not display_text:
            display_text = str(payload)[:150]
            
        self._emit_safe(f"📋 {display_type}", display_text)
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """통계 정보 반환"""
        return {
            "total_received": self.stats.total_received,
            "by_type": dict(self.stats.by_type),
            "last_event_time": self.stats.last_event_time,
            "errors": self.stats.errors,
            "error_rate": self.stats.errors / max(1, self.stats.total_received)
        }

# ---------------- third-party ----------------

CFG_PATH = os.path.join(APP_DIR, "config.yaml")
MEMO_PATH = os.path.join(APP_DIR, "tool_memory.txt")

# ---------------- defaults ----------------
DEFAULT_CFG = {
    "llm_tools": {"endpoint": "http://127.0.0.1:1234/v1", "model": "qwen3-4b-instruct", "api_key": "", "timeout_seconds": 60},
    "llm_chat":  {"endpoint": "http://127.0.0.1:1234/v1", "model": "qwen3-14b-instruct", "api_key": "", "timeout_seconds": 60},
    "llm_vision": {"endpoint": "http://127.0.0.1:1234/v1", "model": "gemma-3-12b-it-qat", "api_key": "", "timeout_seconds": 60},
    "agent": {"health_url": "http://127.0.0.1:8765/health", "event_url": "http://127.0.0.1:8765/event", "health_interval_seconds": 2, "health_fail_quit_count": 3},
    "tools": {},
    "proxy": {"enable": True, "host": "127.0.0.1", "port": 8350},
    "ui": {"opacity": 0.92, "width": 640, "height": 500, "font_family": "Malgun Gothic", "font_size": 11, "always_on_top": True, "show_on_start": True},
    "debug": {"overlay_echo_raw": False, "log_events": False, "connection_test": True}
}
DEFAULT_MEMORY = (
    'You are a Tool Orchestrator. Respond ONLY with compact JSON:\\n'
    '{"say":"...", "tool_calls":[{"name":"...","args":{...}}]}\\n\\n'
    "Hard rules:\\n"
    "- No chain-of-thought; no explanations. JSON ONLY.\\n"
    "- Prefer short, actionable outputs. Omit 'say' if not needed.\\n"
    "- Never translate content yourself. External pipelines handle translation.\\n"
    "- Do not store memory; assume these rules persist between turns.\\n"
    "- Tool calls work only when response is valid JSON; other text is ignored.\\n\\n"
    "When to call tools:\\n"
    "1) OCR needed (new subtitles, small UI text, non-Korean text visible)\\n"
    "   -> {\"name\":\"ocr.start\",\"args\":{\"hint\":\"subtitle|ui|document\"}}\\n"
    "   Stop when done -> {\"name\":\"ocr.stop\",\"args\":{}}\\n\\n"
    "2) Voice chat / audio capture / live transcription\\n"
    "   -> {\"name\":\"stt.start\",\"args\":{\"mode\":\"realtime\"}}\\n"
    "   Stop when done -> {\"name\":\"stt.stop\",\"args\":{}}\\n\\n"
    "3) Summarize desktop notifications / Discord mentions batch\\n"
    "   -> {\"name\":\"discord.collect.start\",\"args\":{\"interval_sec\":300}}\\n"
    "   Stop when done -> {\"name\":\"discord.collect.stop\",\"args\":{}}\\n\\n"
    "4) Leave a note/event for the Agent (no UI output)\\n"
    "   -> {\"name\":\"agent.event\",\"args\":{\"type\":\"overlay.note\",\"payload\":{\"note\":\"...\",\"source\":\"overlay\"},\"priority\":5}}\\n\\n"
    "5) Open a link in the default browser\\n"
    "   -> {\"name\":\"overlay.open_url\",\"args\":{\"url\":\"https://...\"}}\\n"
)


def ensure_files():
    if not os.path.exists(CFG_PATH):
        with open(CFG_PATH, "w", encoding="utf-8") as f:
            yaml.safe_dump(DEFAULT_CFG, f, allow_unicode=True, sort_keys=False)
    if not os.path.exists(MEMO_PATH):
        with open(MEMO_PATH, "w", encoding="utf-8") as f:
            f.write(DEFAULT_MEMORY)


def load_cfg() -> Dict[str, Any]:
    ensure_files()
    with open(CFG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_tool_memory() -> str:
    try:
        with open(MEMO_PATH, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""

# ---------------- utils ----------------


def openai_tc_to_internal(tc: dict):
    # OpenAI tool_calls → 내부 {"name":..., "args": {...}}
    if not isinstance(tc, dict):
        return None
    if tc.get("type") == "function" and isinstance(tc.get("function"), dict):
        fn = tc["function"]
        name = fn.get("name")
        args = fn.get("arguments") or "{}"
        try:
            if isinstance(args, str):
                args = json.loads(args) if args.strip() else {}
        except Exception:
            args = {}
        return {"name": name, "args": args or {}}
    return None


def strip_think_and_fences(text: str) -> str:
    if not text:
        return ""
    s = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    if s.strip().startswith("```"):
        m = re.search(r"```(?:json|JSON)?(.*?)```", s, flags=re.S)
        if m:
            s = m.group(1)
    return s.strip()


def extract_msg_content(choice_msg: Dict[str, Any]) -> str:
    """
    다양한 응답 포맷에서 텍스트를 최대한 잘 뽑아낸다.
    - choice.message.content: str
    - choice.message.content: list[ {type:'text'|'reasoning'|'output_text', text:'...'} , ... ]
    - (fallback) delta 기반/기타 변형 → 빈 문자열 방지
    """
    if not isinstance(choice_msg, dict):
        return ""
    content = choice_msg.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for p in content:
            if isinstance(p, dict):
                if "text" in p and isinstance(p["text"], str):
                    texts.append(p["text"])
                elif "content" in p and isinstance(p["content"], str):
                    texts.append(p["content"])
        return "\n".join([t for t in texts if t])
    return ""


def contains_image(messages: List[Dict[str, Any]]) -> bool:
    for m in messages or []:
        content = m.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    t = part.get("type", "")
                    if t in ("image", "image_url") or part.get("image_url") or part.get("image_base64"):
                        return True
        elif isinstance(content, dict):
            t = content.get("type", "")
            if t in ("image", "image_url") or content.get("image_url") or content.get("image_base64"):
                return True
    return False


def to_tool_call(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(obj, dict):
        return None
    if "name" in obj:
        return {"name": obj.get("name"), "args": obj.get("args", {}) or {}}
    if obj.get("type") == "function" and isinstance(obj.get("function"), dict):
        fn = obj["function"]
        try:
            args = fn.get("arguments") or "{}"
            if isinstance(args, str):
                args = json.loads(args) if args.strip() else {}
        except Exception:
            args = {}
        return {"name": fn.get("name"), "args": args}
    return None


def parse_and_normalize_tools(raw: str) -> Dict[str, Any]:
    result = {"say": "", "tool_calls": []}
    if not raw:
        return result
    s = strip_think_and_fences(raw)
    try:
        data = json.loads(s)
        if isinstance(data, dict):
            if "tool_calls" in data or "say" in data:
                tcs = data.get("tool_calls") or []
                norm = [tc for tc in (to_tool_call(x) for x in tcs) if tc]
                result["say"] = data.get("say", "") or ""
                if not norm and "name" in data:
                    one = to_tool_call(data)
                    if one:
                        norm = [one]
                result["tool_calls"] = norm
                return result
            one = to_tool_call(data)
            if one:
                result["tool_calls"] = [one]
                return result
            result["say"] = s
            return result
        if isinstance(data, list):
            norm = [tc for tc in (to_tool_call(x) for x in data) if tc]
            result["tool_calls"] = norm
            return result
    except Exception:
        pass
    objs = re.findall(r"\{[^{}]*\{.*?\}[^{}]*\}|\{.*?\}", s, flags=re.S)
    collected = []
    for o in objs:
        try:
            val = json.loads(o)
            tc = to_tool_call(val)
            if tc:
                collected.append(tc)
        except Exception:
            continue
    if collected:
        result["tool_calls"] = collected
        return result
    result["say"] = s
    return result


def openai_function_calls(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for tc in tool_calls or []:
        out.append({
            "type": "function",
            "function": {
                "name": tc.get("name"),
                "arguments": json.dumps(tc.get("args", {}), ensure_ascii=False)
            }
        })
    return out

# ---------------- HealthWatcher ----------------


class HealthWatcher(QtCore.QThread):
    dead = QtCore.pyqtSignal(str)
    okping = QtCore.pyqtSignal()

    def __init__(self, url: str, interval: float, max_fail: int):
        super().__init__()
        self.url = url
        self.interval = interval
        self.max_fail = max_fail
        self._stop = False

    def run(self):
        import httpx as _hx
        fail = 0
        while not self._stop:
            try:
                r = _hx.get(self.url, timeout=3.0)
                js = {}
                try:
                    js = r.json()
                except:
                    js = {}
                ok = (r.status_code == 200) and (js.get("ok")
                                                 is True or js.get("status") in ("ok", "healthy", True))
                if ok:
                    self.okping.emit()
                    fail = 0
                else:
                    fail += 1
            except:
                fail += 1
            if fail >= self.max_fail:
                self.dead.emit("Agent health failed")
                break
            time.sleep(self.interval)

    def stop(self): self._stop = True

# ---------------- Orchestrator ----------------


class Orchestrator:
    def __init__(self, cfg: Dict[str, Any], window=None):
        self.cfg = cfg
        self.window = window  # OverlayWindow 참조
        self.tool_memory = read_tool_memory()
        self.history_tools: List[Dict[str, str]] = [{"role": "system", "content": self._system_prompt_tools()}]
        self.history_chat:  List[Dict[str, str]] = [{"role": "system", "content": self._system_prompt_chat()}]
        self.procs: Dict[str, subprocess.Popen] = {}
        self.lock = threading.Lock()
        
        # defaults for tool endpoints
        cfg.setdefault('tool_endpoints', {})
        te = cfg['tool_endpoints']
        te.setdefault('ocr', {'event_url': 'http://127.0.0.1:8765/event'})
        te.setdefault('stt', {'event_url': 'http://127.0.0.1:8765/event'})

        # optional convenience mappings in cfg['tools']
        cfg.setdefault('tools', {})
        tools_map = cfg['tools']
        tools_map.setdefault('ocr.start', te['ocr']['event_url'])
        tools_map.setdefault('ocr.stop',  te['ocr']['event_url'])
        tools_map.setdefault('stt.start', te['stt']['event_url'])
        tools_map.setdefault('stt.stop',  te['stt']['event_url'])
        te.setdefault('web', {'event_url': 'http://127.0.0.1:8765/event'})
        tools_map.setdefault('web.search', te['web']['event_url'])
        te.setdefault('discord', {'event_url': 'http://127.0.0.1:8765/event'})
        tools_map.setdefault('discord.collect.start', te['discord']['event_url'])
        tools_map.setdefault('discord.collect.stop',  te['discord']['event_url'])

        # alias for ocr/stt names
        # WEB aliases
        try:
            tmap = self.cfg.setdefault('tools', {})
            if 'web' in tmap and 'web.search' not in tmap:
                tmap['web.search'] = tmap['web']
        except Exception:
            pass
        try:
            tmap = self.cfg.setdefault('tools', {})
            # OCR aliases
            if 'ocr' in tmap and 'ocr.start' not in tmap:
                tmap['ocr.start'] = tmap['ocr']
            if 'ocr.start' in tmap and 'ocr' not in tmap:
                tmap['ocr'] = tmap['ocr.start']
            if 'ocr.stop' not in tmap:
                # add process_stop if 'ocr' is a process dict
                spec = tmap.get('ocr')
                if isinstance(spec, dict) and (spec.get('kind','').lower() == 'process' or ('command' in spec)):
                    tmap['ocr.stop'] = {'kind': 'process_stop', 'id': spec.get('id','ocr'), 'timeout': 3.0}
                else:
                    # else fallback to same endpoint/url
                    tmap['ocr.stop'] = tmap.get('ocr.start') or tmap.get('ocr')
            # STT aliases
            if 'stt' in tmap and 'stt.start' not in tmap:
                tmap['stt.start'] = tmap['stt']
            if 'stt.start' in tmap and 'stt' not in tmap:
                tmap['stt'] = tmap['stt.start']
            if 'stt.stop' not in tmap:
                spec = tmap.get('stt')
                if isinstance(spec, dict) and (spec.get('kind','').lower() == 'process' or ('command' in spec)):
                    tmap['stt.stop'] = {'kind': 'process_stop', 'id': spec.get('id','stt'), 'timeout': 3.0}
                else:
                    tmap['stt.stop'] = tmap.get('stt.start') or tmap.get('stt')
        except Exception as e:
            logger.warning(f"[tools] alias mapping failed: {e}")

    def orchestrate(self, user_text: str) -> Dict[str, Any]:
        """
        Execute tool routing and run mapped tools (synchronous wrapper).
        Preserves existing behavior used by worker(): call_tools -> run_tool_calls.
        """
        parsed = asyncio.run(self.call_tools(user_text))
        asyncio.run(self.run_tool_calls(parsed.get("tool_calls") or []))
        return parsed

    def _heuristic_route(self, user_text: str):
        ut = (user_text or "").strip()
        low = ut.lower()

        # 웹검색
        if ("웹검색" in ut) or ("검색" in ut) or ("search" in low):
            q = ut
            for kw in ("웹검색","검색","search"):
                if kw in ut:
                    i = ut.find(kw)+len(kw)
                    q = ut[i:].strip() or ut
                    break
            return [{"name":"web.search","args":{"q": q}}]

        # OCR
        if ("ocr" in low or "자막" in ut) and any(k in ut for k in ("켜","켜줘","start","시작","시작해")):
            return [{"name":"ocr.start","args":{}}]
        if ("ocr" in low or "자막" in ut) and any(k in ut for k in ("꺼","꺼줘","stop","중지","종료","멈춰")):
            return [{"name":"ocr.stop","args":{}}]

        # STT
        if "stt" in low and any(k in ut for k in ("켜","켜줘","start","시작","시작해")):
            return [{"name":"stt.start","args":{}}]
        if "stt" in low and any(k in ut for k in ("꺼","꺼줘","stop","중지","종료","멈춰")):
            return [{"name":"stt.stop","args":{}}]

        return []

    def _system_prompt_tools(self) -> str:
        # tool_memory.txt에서 가져온 상세한 지침 사용
        base_content = '''You are a Tool Orchestrator. Respond ONLY with compact JSON:
{"say":"...", "tool_calls":[{"name":"...","args":{...}}]}

Hard rules:
- No chain-of-thought; no explanations. JSON ONLY.
- Prefer short, actionable outputs. If you have nothing to say, omit "say".
- Never translate content yourself. External pipelines handle translation.
- Do not store memory; assume these rules persist between turns.
- Tool calls work only when response is valid JSON; other text is ignored.

When to call tools:
1) OCR needed (new subtitles, small UI text, non-Korean text visible)
   -> {"name":"ocr.start","args":{"hint":"subtitle|ui|document"}}
   Stop when done -> {"name":"ocr.stop","args":{}}

2) Voice chat / audio capture / live transcription
   -> {"name":"stt.start","args":{"mode":"realtime"}}
   Stop when done -> {"name":"stt.stop","args":{}}

3) Summarize desktop notifications / Discord mentions batch
   -> {"name":"discord.collect.start","args":{"interval_sec":300}}
   Stop when done -> {"name":"discord.collect.stop","args":{}}

4) Leave a note/event for the Agent (no UI output)
   -> {"name":"agent.event","args":{"type":"overlay.note","payload":{"note":"...","source":"overlay"},"priority":5}}

5) Open a link in the default browser
   -> {"name":"overlay.open_url","args":{"url":"https://example.com"}}

Examples:
Q: 자막 켜줘
A: {"say": "", "tool_calls": [{"name": "ocr.start", "args": {"hint": "subtitle"}}]}

Q: STT 끄기
A: {"say": "", "tool_calls": [{"name": "stt.stop", "args": {}}]}

Q: 안녕하세요
A: {"say": "안녕하세요! 무엇을 도와드릴까요?", "tool_calls": []}
'''
        
        # tool_memory.txt 내용 추가
        memory_content = self.tool_memory.strip()
        if memory_content and memory_content != base_content:
            return memory_content  # tool_memory.txt 우선 사용
        else:
            return base_content  # 폴백

    def _system_prompt_chat(self) -> str:
        return ("You are a friendly Korean assistant. Keep replies concise in Korean by default. "
                "Do NOT call tools; just answer helpfully.")

    def reload_memory(self):
        with self.lock:
            self.tool_memory = read_tool_memory()
            self.history_tools = [{"role": "system", "content": self._system_prompt_tools()}]
            self.history_chat  = [{"role": "system", "content": self._system_prompt_chat()}]

    async def _chat_complete_raw(self, llm_cfg: Dict[str, Any], messages: List[Dict[str, str]]) -> Dict[str, Any]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if llm_cfg.get("api_key"):
            headers["Authorization"] = f"Bearer {llm_cfg['api_key']}"
        payload = {"model": llm_cfg["model"], "messages": messages, "temperature": 0.2}
        timeout = httpx.Timeout(connect=5.0, read=float(llm_cfg.get("timeout_seconds", 60)), write=15.0, pool=10.0)
        url = f"{llm_cfg['endpoint'].rstrip('/')}/chat/completions"
        logger.info(f"[llm] POST {url} model={llm_cfg['model']} msgs={len(messages)}")
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(url, headers=headers, json=payload)
            txt = r.text
            try:
                data = r.json()
            except Exception:
                logger.error(f"[llm] non-json response: {txt[:500]}")
                r.raise_for_status()
                data = {}
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        content = extract_msg_content(msg)
        tool_calls = msg.get("tool_calls") or []
        logger.info(f"[llm] got {len(content or '')} chars (finish_reason={choice.get('finish_reason')}); tool_calls={len(tool_calls)}")
        return {"message": msg, "content": (content or ""), "tool_calls": tool_calls}

    async def call_chat(self, user_text: str) -> str:
        # Pin 상태 고려한 LLM 설정 사용
        if self.window and hasattr(self.window, '_get_effective_llm_config'):
            llm = self.window._get_effective_llm_config("chat")
        else:
            llm = self.cfg.get("llm_chat") or self.cfg.get("llm")
        pin_20b = bool(self.window and getattr(self.window, "pin_active", False) and getattr(self.window, "pinned_model", None) == "20b")

        if pin_20b:
            msgs = [{"role": "system", "content": "Reasoning: medium"}, {"role": "user", "content": user_text}]
        else:
            msgs = self.history_chat[-12:] + [{"role": "user", "content": user_text}]

        resp = await self._chat_complete_raw(llm, msgs)
        out = strip_think_and_fences(resp.get("content") or extract_msg_content(resp.get("message") or {}) or "")

        if not pin_20b:
            self.history_chat += [{"role": "user", "content": user_text}, {"role": "assistant", "content": out}]
            if len(self.history_chat) > 16:
                self.history_chat = [self.history_chat[0]] + self.history_chat[-15:]
        return out

    async def call_vision(self, messages: List[Dict[str, Any]]) -> str:
        if self.window and hasattr(self.window, '_get_effective_llm_config'):
            llm = self.window._get_effective_llm_config("vision")
        else:
            llm = self.cfg.get("llm_vision") or self.cfg.get("llm_chat") or self.cfg.get("llm")

        pin_20b = bool(self.window and getattr(self.window, "pin_active", False) and getattr(self.window, "pinned_model", None) == "20b")
        if pin_20b:
            messages = [{"role": "system", "content": "Reasoning: medium"}, *messages]

        resp = await self._chat_complete_raw(llm, messages)
        out = strip_think_and_fences(resp.get("content") or extract_msg_content(resp.get("message") or {}) or "")
        return out

    async def call_tools(self, user_text: str) -> Dict[str, Any]:
        # Pin 상태 고려한 LLM 설정 사용
        if self.window and hasattr(self.window, '_get_effective_llm_config'):
            llm = self.window._get_effective_llm_config("tools")
        else:
            llm = self.cfg.get("llm_tools") or self.cfg.get("llm")
            
        msgs = self.history_tools[-8:] + [{"role":"user","content":user_text}]

        # ① 모델 호출
        resp = await self._chat_complete_raw(llm, msgs)
        out = resp.get("content") or ""
        tc_openai = resp.get("tool_calls") or []

        # ② 텍스트 파싱
        parsed = parse_and_normalize_tools(out)
        say = parsed.get("say") or ""
        tc_from_text = parsed.get("tool_calls") or []

        # ③ 휴리스틱(최우선)
        tc_heur = self._heuristic_route(user_text)

        # ④ 허용 툴만 통과
        allowed = {"ocr.start","ocr.stop","stt.start","stt.stop","web.search"}  # ← agent.event 는 비허용
        def _filter_ok(calls):
            ok=[]; 
            for tc in (calls or []):
                if isinstance(tc, dict) and (tc.get("name") in allowed):
                    ok.append({"name": tc["name"], "args": tc.get("args") or {}})
            return ok

        tool_calls = []
        if tc_heur:
            tool_calls = _filter_ok(tc_heur)
        elif tc_from_text:
            tool_calls = _filter_ok(tc_from_text)
        else:
            for tc in tc_openai:
                it = openai_tc_to_internal(tc)
                if it: tool_calls.append(it)
            tool_calls = _filter_ok(tool_calls)

        self.history_tools += [
            {"role":"user","content":user_text},
            {"role":"assistant","content": out if out else json.dumps(tc_openai, ensure_ascii=False)},
        ]
        if len(self.history_tools) > 12:
            self.history_tools = [self.history_tools[0]] + self.history_tools[-10:]

        return {"say": say, "tool_calls": tool_calls}

    def _proc_launch(self, name: str, spec: Dict[str, Any], call_args: Dict[str, Any] | None = None):
        # format args with placeholders like {q}, {k}, {topk}, {provider}
        def _fmt_args(raw_args, ctx: dict):
            out = []
            if not raw_args:
                return out
            import re as _re
            defaults = {
                "q": (ctx or {}).get("q") or (ctx or {}).get("query") or "",
                "query": (ctx or {}).get("q") or (ctx or {}).get("query") or "",
                "k": str((ctx or {}).get("k") or (ctx or {}).get("topk") or 5),
                "topk": str((ctx or {}).get("topk") or (ctx or {}).get("k") or 5),
                "provider": (ctx or {}).get("provider") or "auto",
                "summarize": str((ctx or {}).get("summarize") if (ctx or {}).get("summarize") is not None else ""),
                "post": str((ctx or {}).get("post") if (ctx or {}).get("post") is not None else ""),
            }
            def _templ(s: str):
                def rep(m):
                    key = m.group(1)
                    return str((ctx or {}).get(key, defaults.get(key, "")))
                return _re.sub(r"\{(\w+)\}", rep, s)
            out = []
            for a in list(raw_args):
                if isinstance(a, str):
                    out.append(_templ(a))
                else:
                    out.append(str(a))
            return out

        pid_name = spec.get("id") or name
        if pid_name in self.procs and self.procs[pid_name] and self.procs[pid_name].poll() is None:
            logger.info(f"[proc] {pid_name} already running (pid={self.procs[pid_name].pid})")
            return
        raw_args = list(spec.get("args", []))
        cmd = [spec.get("command"), *(_fmt_args(raw_args, call_args or {}))]
        cmd = [c for c in cmd if c]
        if not cmd or not cmd[0]:
            raise RuntimeError(f"Invalid process spec for {name}: missing command")
        cwd = spec.get("cwd") or None
        env = os.environ.copy()
        env.update(spec.get("env", {}) or {})
        creationflags = 0
        if platform.system() == "Windows" and spec.get("no_console", True):
            creationflags |= getattr(subprocess, "CREATE_NO_WINDOW", 0)
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, creationflags=creationflags)
        self.procs[pid_name] = proc
        logger.info(f"[proc] launched {pid_name} pid={proc.pid} cmd={cmd} cwd={cwd}")

    def _proc_stop(self, spec: Dict[str, Any]):
        pid_name = spec.get("id")
        if not pid_name:
            raise RuntimeError("process_stop requires 'id'")
        proc = self.procs.get(pid_name)
        if not proc:
            logger.info(f"[proc] {pid_name} not running")
            return
        if proc.poll() is None:
            try:
                if (spec.get("method") or "terminate") == "kill":
                    proc.kill()
                else:
                    proc.terminate()
                try:
                    proc.wait(timeout=float(spec.get("timeout", 3.0)))
                except Exception:
                    proc.kill()
            except Exception as e:
                logger.error(f"[proc] stop error for {pid_name}: {e}")
        logger.info(f"[proc] stopped {pid_name}")
        self.procs.pop(pid_name, None)

    async def run_tool_calls(self, tool_calls: List[Dict[str, Any]]):
        if not tool_calls:
            return
        tcfg = self.cfg.get("tools", {})
        async with httpx.AsyncClient(timeout=10.0) as client:
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                name = (call or {}).get("name", "")
                args = (call or {}).get("args", {}) or {}

                if name == "overlay.open_url":
                    url = args.get("url") or args.get("href")
                    if url:
                        try:
                            webbrowser.open(url)
                            logger.info(f"[tool] overlay.open_url → {url}")
                        except Exception as e:
                            logger.error(f"[tool] overlay.open_url failed: {e}")
                    else:
                        logger.warning("[tool] overlay.open_url missing url")
                    continue

                spec = tcfg.get(name)
                try:
                    if isinstance(spec, dict):
                        kind = (spec.get("kind") or "").lower()
                        if kind == "process":
                            self._proc_launch(name, spec, args)
                        elif kind in ("process_stop", "stop"):
                            merged = {**spec, **({k: v for k, v in args.items() if k in ("id","method","timeout")})}
                            self._proc_stop(merged)
                        else:
                            logger.warning(f"[tool] unknown dict kind for {name}: {kind}")
                        continue
                    if isinstance(spec, str) and spec.startswith("http"):
                        payload = {"name": name, "args": (args or {})}
                        await client.post(spec, json=payload)
                        logger.info(f"[tool] {name} → {spec}")
                    elif name == "agent.event":
                        ev = {"type": args.get("type", "note"), "payload": args.get("payload", {}), "priority": int(args.get("priority", 5))}
                        await client.post(self.cfg["agent"]["event_url"], json=ev)
                        logger.info(f"[tool] agent.event → {ev['type']}")
                    else:
                        logger.warning(f"[tool] no mapping for {name}")
                except Exception as e:
                    logger.error(f"[tool] {name} failed: {e}")

    # ---- sync wrappers ----
    def chat(self, user_text: str) -> str:
        return asyncio.run(self.call_chat(user_text))

    def chat_vision(self, messages: List[Dict[str, Any]]) -> str:
        return asyncio.run(self.call_vision(messages))


# ---------------- Enhanced Proxy Server ----------------


class ProxyServer:
    def __init__(self, orch: Orchestrator, host: str, port: int, event_handler: EventHandler):
        self.orch = orch
        self.event_handler = event_handler
        self.host = host
        self.port = int(port)
        self.app = FastAPI(title="Luna Overlay Proxy")
        self._setup_routes()
        self._server = None
        self._thread = None
        self.connection_test_done = False

    def _setup_routes(self):
        app = self.app
        orch = self.orch
        handler = self.event_handler

        @app.get("/health")
        async def health():
            stats = handler.get_stats()
            return {
                "ok": True, 
                "who": "luna-overlay-proxy",
                "event_stats": stats,
                "uptime": time.time()
            }

        @app.post("/event")
        async def receive_event(req: Request):
            """이벤트 버스에서 오는 OCR/STT/웹검색/LLM 이벤트 수신"""
            try:
                body = await req.json()
                event_type = body.get("type", "")
                payload = body.get("payload", {}) or {}
                
                if not event_type:
                    return JSONResponse({"error": "Missing event type"}, status_code=400)
                
                logger.debug(f"[event] Received {event_type}")
                
                # 이벤트 핸들러로 처리
                success = handler.handle_event(event_type, payload)
                
                return {
                    "ok": success, 
                    "message": "processed" if success else "failed",
                    "type": event_type
                }
                
            except Exception as e:
                logger.error(f"[event] Error processing event: {e}")
                return JSONResponse({"error": str(e)}, status_code=500)

        @app.post("/overlay/event")
        async def overlay_specific_event(req: Request):
            """Overlay 전용 이벤트 (토스트 등)"""
            try:
                body = await req.json()
                event_type = body.get("type", "")
                payload = body.get("payload", {}) or {}
                
                if not event_type:
                    return JSONResponse({"error": "Missing event type"}, status_code=400)
                
                # overlay. 접두사 강제
                if not event_type.startswith("overlay."):
                    event_type = f"overlay.{event_type}"
                
                success = handler.handle_event(event_type, payload)
                
                return {
                    "ok": success,
                    "message": "processed" if success else "failed",
                    "type": event_type
                }
                
            except Exception as e:
                logger.error(f"[overlay_event] Error: {e}")
                return JSONResponse({"error": str(e)}, status_code=500)

        @app.get("/stats")
        async def get_stats():
            """이벤트 처리 통계"""
            return handler.get_stats()
            
        @app.post("/test")
        async def test_event(req: Request):
            """연결 테스트용 엔드포인트"""
            try:
                body = await req.json()
                test_type = body.get("type", "test")
                test_msg = body.get("message", "연결 테스트")
                
                success = handler.handle_event(f"test.{test_type}", {"text": test_msg})
                
                return {
                    "ok": success,
                    "message": "Test event processed",
                    "stats": handler.get_stats()
                }
            except Exception as e:
                return JSONResponse({"error": str(e)}, status_code=500)

        @app.post("/v1/chat/completions")
        async def completions(req: Request):
            """기존 LLM 프록시 기능 유지"""
            body = await req.json()
            messages = body.get("messages") or []

            if contains_image(messages):
                created = int(time.time())
                stream = bool(body.get("stream"))
                out = orch.chat_vision(messages)
                model_name = (orch.cfg.get("llm_vision") or orch.cfg.get("llm_chat") or orch.cfg.get("llm") or {}).get("model")
                if stream:
                    async def vision_stream():
                        chunk = {
                            "id": f"chatcmpl-{created}",
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_name,
                            "choices": [{"index": 0, "delta": {"role": "assistant", "content": out}, "finish_reason": None}]
                        }
                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                        yield "data: [DONE]\n\n"
                    return StreamingResponse(vision_stream(), media_type="text/event-stream")
                return JSONResponse({
                    "id": f"chatcmpl-{created}",
                    "object": "chat.completion",
                    "created": created,
                    "model": model_name,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": out}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                })

            user_text = ""
            for m in reversed(messages):
                if (m or {}).get("role") == "user":
                    user_text = (m or {}).get("content") or ""
                    break
            want_mode = body.get("luna_mode") or "auto"
            mode = "tools"
            if want_mode == "chat":
                mode = "chat"
            elif want_mode == "tools":
                mode = "tools"
            else:
                mdl = (body.get("model") or "") + ""
                chat_model = (orch.cfg.get("llm_chat") or orch.cfg.get(
                    "llm") or {}).get("model", "")
                if mdl and chat_model and mdl == chat_model:
                    mode = "chat"

            created = int(time.time())
            stream = bool(body.get("stream"))

            if mode == "chat":
                if stream:
                    async def chat_stream():
                        out = orch.chat(user_text)
                        chunk = {
                            "id": f"chatcmpl-{created}",
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": (orch.cfg.get("llm_chat") or orch.cfg.get("llm") or {}).get("model"),
                            "choices": [{"index": 0, "delta": {"role": "assistant", "content": out}, "finish_reason": None}]
                        }
                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                        yield "data: [DONE]\n\n"
                    return StreamingResponse(chat_stream(), media_type="text/event-stream")

                out = orch.chat(user_text)
                return JSONResponse({
                    "id": f"chatcmpl-{created}",
                    "object": "chat.completion",
                    "created": created,
                    "model": (orch.cfg.get("llm_chat") or orch.cfg.get("llm") or {}).get("model"),
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": out}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                })

            # tools mode
            parsed = orch.orchestrate(user_text)
            say = strip_think_and_fences((parsed.get("say") or ""))
            tc_openai = openai_function_calls(parsed.get("tool_calls") or [])

            if stream:
                async def tools_stream():
                    delta = {
                        "id": f"chatcmpl-{created}",
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": (orch.cfg.get("llm_tools") or orch.cfg.get("llm") or {}).get("model"),
                        "choices": [{"index": 0, "delta": {"role": "assistant", "content": say, "tool_calls": tc_openai}, "finish_reason": None}]
                    }
                    yield f"data: {json.dumps(delta, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                return StreamingResponse(tools_stream(), media_type="text/event-stream")

            return JSONResponse({
                "id": f"chatcmpl-{created}",
                "object": "chat.completion",
                "created": created,
                "model": (orch.cfg.get("llm_tools") or orch.cfg.get("llm") or {}).get("model"),
                "choices": [{"index": 0, "message": {"role": "assistant", "content": say, "tool_calls": tc_openai}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            })

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        config = uvicorn.Config(self.app, host=self.host,
                                port=self.port, log_level="warning")
        self._server = uvicorn.Server(config)

        def runner():
            asyncio.set_event_loop(asyncio.new_event_loop())
            self._server.run()
        self._thread = threading.Thread(target=runner, daemon=True)
        self._thread.start()
        logger.info(
            f"[proxy] started at http://{self.host}:{self.port}")
        
        # 연결 테스트
        if not self.connection_test_done:
            threading.Thread(target=self._test_connection_async, daemon=True).start()

    def _test_connection_async(self):
        """비동기 연결 테스트"""
        time.sleep(1)  # 서버 시작 대기
        try:
            import requests
            # 헬스 체크
            r = requests.get(f"http://{self.host}:{self.port}/health", timeout=2.0)
            if r.status_code == 200:
                if hasattr(self.event_handler, 'window'):
                    self.event_handler.window.appended.emit("🟢 연결테스트", "프록시 서버 정상 작동")
                
                # 테스트 이벤트
                test_r = requests.post(f"http://{self.host}:{self.port}/test", 
                    json={"type": "startup", "message": "오버레이 시작됨"}, timeout=2.0)
                if test_r.status_code == 200:
                    logger.info("[proxy] Connection test passed")
                else:
                    logger.warning("[proxy] Test event failed")
            else:
                logger.warning(f"[proxy] Health check failed: {r.status_code}")
        except Exception as e:
            logger.error(f"[proxy] Connection test failed: {e}")
        self.connection_test_done = True

    def stop(self):
        if self._server:
            self._server.should_exit = True

# ---------------- Overlay Window ----------------
# Defaults for UI
class _CfgDefaults:
    FONT_FAMILY = "Malgun Gothic"  # Windows 기본 한글 폰트
    FONT_SIZE = 11
    WIDTH = 640
    HEIGHT = 500
    OPACITY = 0.92
    ALWAYS_ON_TOP = True
    SHOW_ON_START = True


class OverlayWindow(QtWidgets.QWidget):
    appended = QtCore.pyqtSignal(str, str)  # (who, msg)

    def __init__(self, cfg: Dict[str, Any], tray: "Tray"):
        super().__init__()
        self.cfg = cfg
        self.tray = tray
        self.mode = "tools"
        self.transcript = []
        self.pending_vision_images: List[Dict[str, Any]] = []

        # Pin 모델 관리
        self.pinned_model = None  # None, "4b", "14b", "20b"
        self.pin_active = False

        QtWidgets.QApplication.setQuitOnLastWindowClosed(False)
        if not QtWidgets.QSystemTrayIcon.isSystemTrayAvailable():
            print("[WARN] System tray not available. Overlay will run without tray.")

        flags = QtCore.Qt.FramelessWindowHint | QtCore.Qt.Tool
        if (cfg.get("ui") or {}).get("always_on_top", _CfgDefaults.ALWAYS_ON_TOP):
            flags |= QtCore.Qt.WindowStaysOnTopHint
        self.setWindowFlags(flags)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        self.setWindowTitle("Luna Overlay")

        ui = self._ui_cfg()
        self.resize(ui["width"], ui["height"])

        font = QtGui.QFont(ui["font_family"], ui["font_size"])

        self.container = QtWidgets.QFrame(self)
        self.container.setStyleSheet("background-color: rgba(10,10,15,220); border-radius: 14px;")
        self.container.setGeometry(0, 0, self.width(), self.height())

        self.title = QtWidgets.QLabel(self.container)
        self.title.setText("Luna Overlay v9 – Mode: Tools(4B)")
        self.title.setStyleSheet("color:#9CC4FF; font-weight:bold;")
        self._update_title()  # 초기 제목 설정

        self.output = QtWidgets.QTextEdit(self.container)
        self.output.setReadOnly(True)
        self.output.setFont(font)
        self.output.setStyleSheet("color:#DCE4F7; background: transparent; border:none;")

        self.input = QtWidgets.QLineEdit(self.container)
        self.input.setFont(font)
        self.input.setStyleSheet("color:#E7F1FF; background: rgba(255,255,255,28); padding:6px 10px; border-radius:10px;")
        self.input.returnPressed.connect(self.on_send)

        self.btn_image = QtWidgets.QPushButton("🖼️", self.container)
        self.btn_image.setFixedWidth(40)
        self.btn_image.clicked.connect(self.on_image)

        layout = QtWidgets.QVBoxLayout(self.container)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.addWidget(self.title, 0)
        layout.addWidget(self.output, 1)
        input_row = QtWidgets.QHBoxLayout()
        input_row.addWidget(self.input, 1)
        input_row.addWidget(self.btn_image, 0)
        layout.addLayout(input_row, 0)

        self._drag_pos = None
        self.setWindowOpacity(float(ui["opacity"]))

        self.orch = Orchestrator(cfg, window=self)
        
        # 이벤트 핸들러 초기화
        self.event_handler = EventHandler(self)
        debug_cfg = cfg.get("debug", {})
        if debug_cfg.get("log_events", False):
            self.event_handler.set_debug_mode(True)

        # health watcher (optional)
        try:
            agent = (cfg.get("agent") or {}) if isinstance(cfg, dict) else {}
            enable = bool(agent.get("enable_health_watch", False))
            if enable:
                url = agent.get("health_url", "http://127.0.0.1:8765/health")
                interval = float(agent.get("health_interval_seconds", 5))
                fail_quit = int(agent.get("health_fail_quit_count", 6))
                self.health = HealthWatcher(url, interval, fail_quit)
                self.health.dead.connect(self._on_dead)
                self.health.okping.connect(lambda: None)
                self.health.start()
                self._append("overlay", f"Agent health watching: {url} (every {interval}s)")
            else:
                self.health = None
                self._append("overlay", "Agent health watch: disabled")
        except Exception as e:
            self._append("error", f"health watcher init failed: {e}")

        # 견고한 프록시 서버
        px = cfg.get("proxy", {})
        try:
            if px.get("enable", True):
                self.proxy = ProxyServer(self.orch, px.get("host", "127.0.0.1"), 
                                        int(px.get("port", 8350)), self.event_handler)
                self.proxy.start()
                
                host, port = px.get('host','127.0.0.1'), int(px.get('port',8350))
                self._append("overlay", f"✅ 프록시: http://{host}:{port}/v1/chat/completions")
                self._append("overlay", f"✅ 이벤트 수신: http://{host}:{port}/event")
                self._append("overlay", f"✅ 토스트: http://{host}:{port}/overlay/event")
                self._append("overlay", f"✅ 테스트: http://{host}:{port}/test")
            else:
                self.proxy = None
                self._append("overlay", "❌ 프록시 비활성화")
        except Exception as e:
            self._append("error", f"proxy init failed: {e}")
            self.proxy = None

        # connect thread-safe appends
        try:
            self.appended.connect(self._append)
        except Exception:
            pass

        self._append("overlay", "Ready v9 - Enhanced Event Processing + Model Pinning")
        threading.Thread(target=self._ping_llm_both, daemon=True).start()

        QtWidgets.QShortcut(QtGui.QKeySequence("Esc"), self, activated=self.hide)
        # tray.toggle는 나중에 tray가 self.window로 연결된 후에도 안전
        QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+`"), self, activated=self.tray.toggle)

        if (cfg.get("ui") or {}).get("show_on_start", _CfgDefaults.SHOW_ON_START):
            QtCore.QTimer.singleShot(100, self._ensure_show)

        self._append("overlay", "명령어는 /help 를 입력하세요.")
        self._append("overlay", "📌 모델 고정: /pin4b, /pin14b, /pin20b, /pinoff")

    def _ui_cfg(self):
        ui = (self.cfg.get('ui') or {}) if isinstance(self.cfg, dict) else {}
        return {
            'font_family': ui.get('font_family', _CfgDefaults.FONT_FAMILY),
            'font_size': int(ui.get('font_size', _CfgDefaults.FONT_SIZE)),
            'width': int(ui.get('width', _CfgDefaults.WIDTH)),
            'height': int(ui.get('height', _CfgDefaults.HEIGHT)),
            'opacity': float(ui.get('opacity', _CfgDefaults.OPACITY)),
            'always_on_top': bool(ui.get('always_on_top', _CfgDefaults.ALWAYS_ON_TOP)),
            'show_on_start': bool(ui.get('show_on_start', _CfgDefaults.SHOW_ON_START)),
        }

    def _ensure_show(self):
        if not self.isVisible():
            pos = QtGui.QCursor.pos()
            self.move(pos.x()-self.width()//2, pos.y()-self.height()//2)
            self.show()
            self.raise_()
            self.activateWindow()

    def _append(self, who: str, msg: str):
        safe = str(msg).replace("<", "&lt;").replace(">", "&gt;")
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.output.append(f"<span style='color:#666'>[{timestamp}]</span> <b style='color:#7FB2FF'>[{who}]</b> {safe}")
        self.transcript.append(f"[{timestamp}] [{who}] {str(msg)}")

    # drag move
    def mousePressEvent(self, event: QtGui.QMouseEvent):
        if event.button() == QtCore.Qt.LeftButton:
            self._drag_pos = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent):
        if self._drag_pos is not None and (event.buttons() & QtCore.Qt.LeftButton):
            self.move(event.globalPos() - self._drag_pos)
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent):
        self._drag_pos = None
        event.accept()
        super().mouseReleaseEvent(event)

    def closeEvent(self, event: QtGui.QCloseEvent):
        try:
            if hasattr(self, "health") and self.health:
                self.health.stop()
        except Exception:
            pass
        try:
            if hasattr(self, "proxy") and self.proxy:
                self.proxy.stop()
        except Exception:
            pass
        super().closeEvent(event)

    @QtCore.pyqtSlot()
    def _on_dead(self, reason: str):
        try:
            self.tray.showMessage("Luna Overlay", "Agent down. Quitting.",
                                  QtWidgets.QSystemTrayIcon.Information, 2000)
        except Exception:
            pass
        QtCore.QTimer.singleShot(500, self.close)

    def _update_title(self):
        """제목 업데이트 (모드 및 pin 상태 반영)"""
        if self.mode == 'chat':
            mode_text = 'Chat(14B)'
        elif self.mode == 'vision':
            mode_text = 'Vision(12B)'
        else:
            mode_text = 'Tools(4B)'

        pin_text = f" | Pin: {self.pinned_model.upper()}" if self.pin_active and self.pinned_model else ""
        self.title.setText(f"Luna Overlay v9 – Mode: {mode_text}{pin_text}")

    def set_mode(self, mode: str):
        self.mode = mode
        self._update_title()
        if mode == 'vision':
            desc = '비전 모드(12B)'
        elif mode == 'chat':
            desc = '대화모드(14B)'
        else:
            desc = '툴 모드(4B)'
        self._append("overlay", f"모드 전환: {desc}")

    def _get_effective_llm_config(self, preferred_type: str = "tools") -> Dict[str, Any]:
        """Pin 상태를 고려한 LLM 설정 반환"""
        cfg = self.orch.cfg
        
        # Pin이 활성화된 경우 Pin된 모델 사용
        if self.pin_active and self.pinned_model:
            if self.pinned_model == "4b":
                return cfg.get("llm_tools") or cfg.get("llm_tools_small") or cfg.get("llm", {})
            elif self.pinned_model == "14b":
                return cfg.get("llm_chat") or cfg.get("llm_tools_chat") or cfg.get("llm", {})
            elif self.pinned_model == "20b":
                return cfg.get("llm_summary") or cfg.get("llm_chat") or cfg.get("llm", {})

        # Pin이 없는 경우 기본 동작
        if preferred_type == "chat":
            return cfg.get("llm_chat") or cfg.get("llm", {})
        if preferred_type == "vision":
            return cfg.get("llm_vision") or cfg.get("llm_chat") or cfg.get("llm", {})
        return cfg.get("llm_tools") or cfg.get("llm", {})

    def _set_pin_model(self, model: str):
        """모델 Pin 설정"""
        self.pinned_model = model
        self.pin_active = True
        self._update_title()
        self._append("overlay", f"📌 모델 고정: {model.upper()}")

    def _clear_pin(self):
        """Pin 해제"""
        old_model = self.pinned_model
        self.pinned_model = None
        self.pin_active = False
        self._update_title()
        self._append("overlay", f"📌 해제: {old_model.upper() if old_model else 'None'}")

    def _status_line(self) -> str:
        cfg = self.orch.cfg
        
        # Pin 상태 고려한 실제 사용 모델
        if self.pin_active and self.pinned_model:
            effective_cfg = self._get_effective_llm_config("tools")
            m_tools = effective_cfg.get('model', 'unknown')
            effective_cfg = self._get_effective_llm_config("chat")
            m_chat = effective_cfg.get('model', 'unknown')
            pin_info = f" 📌{self.pinned_model.upper()}"
        else:
            m_tools = (cfg.get('llm_tools') or cfg.get('llm') or {}).get('model')
            m_chat = (cfg.get('llm_chat') or cfg.get('llm') or {}).get('model')
            pin_info = ""
        m_vision = (cfg.get('llm_vision') or cfg.get('llm_chat') or cfg.get('llm') or {}).get('model')

        px = cfg.get("proxy", {})
        purl = f"http://{px.get('host','127.0.0.1')}:{int(px.get('port',8350))}/v1/chat/completions"

        # 이벤트 통계 추가
        stats = self.event_handler.get_stats()

        if self.mode == 'chat':
            mode_disp = '대화(14B)'
        elif self.mode == 'vision':
            mode_disp = '비전(12B)'
        else:
            mode_disp = '툴(4B)'

        return (f"모드: {mode_disp}{pin_info} | "
                f"tools={m_tools} | chat={m_chat} | vision={m_vision} | proxy={purl}\n"
                f"이벤트: 총 {stats['total_received']}개, 에러 {stats['errors']}개 "
                f"({stats['error_rate']:.1%})")

    def _ping_llm_both(self):
        def ping_one(label, llm_cfg):
            try:
                if not llm_cfg:
                    return
                url = llm_cfg["endpoint"].rstrip("/") + "/models"
                r = httpx.get(url, timeout=3.0)
                self.appended.emit("overlay", f"{label} ping: {r.status_code} ({url})")
            except Exception as e:
                self.appended.emit("error", f"{label} ping fail: {e}")
        try:
            ping_one("Tools LLM", self.orch.cfg.get("llm_tools") or self.orch.cfg.get("llm"))
            ping_one("Chat  LLM", self.orch.cfg.get("llm_chat") or self.orch.cfg.get("llm"))
        except Exception as e:
            self.appended.emit("error", f"ping error: {e}")

    def _show_plugins(self):
        te = (self.orch.cfg.get('tool_endpoints') or {})
        msg = ["플러그인 엔드포인트:"]
        for k, v in te.items():
            msg.append(f"  - {k}: {v.get('event_url','(none)')}")
        self._append('overlay', "\n".join(msg))

    def _run_tool_calls_async(self, calls):
        import threading
        def _w():
            try:
                import asyncio
                asyncio.run(self.orch.run_tool_calls(calls))
                self.appended.emit('overlay', 'tool calls sent')
            except Exception as e:
                self.appended.emit('error', f'tool calls failed: {e}')
        threading.Thread(target=_w, daemon=True).start()

    def _remove_think_tags(self):
        raw = self.output.toPlainText()
        cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r"</?think>", "", cleaned, flags=re.IGNORECASE)
        self.output.setPlainText(cleaned)
        self._append("overlay", "<Think> 태그 제거됨")

    def _help(self):
        self._append("overlay",
                     "명령어:\n"
                     "  /대화모드 | /chat        → 대화모드(14B)\n"
                     "  /대화종료 | /end         → 툴 모드(4B)\n"
                     "  /vision                → 비전모드(12B)\n"
                     "  /상태 | /status          → 현재 모드/모델/프록시/이벤트 통계 표시\n"
                     "  /리셋 | /reset           → 히스토리 초기화 + tool_memory 재적용\n"
                     "  /memory reload           → tool_memory.txt 다시 읽기\n"
                     "  /model tools <name>      → 4B 모델 변경\n"
                     "  /model chat <name>       → 14B 모델 변경\n"
                     "  /model vision <name>     → 비전 모델 변경\n"
                     "  /opacity <0~1>           → 투명도 조절\n"
                     "  /size WxH                → 크기 변경 ex) /size 680x520\n"
                     "  /pos X Y                 → 위치 이동\n"
                     "  /top on|off              → 항상 위 토글\n"
                     "  /save                    → 대화 로그 저장\n"
                     "  /ping                    → 두 LLM 핑\n"
                     "  /proxy url               → 프록시 주소 표시\n"
                     "  /stats                   → 이벤트 처리 통계 표시\n"
                     "  /test <메시지>           → 테스트 이벤트 전송\n"
                     "  /debug on|off            → 디버그 모드 토글\n"
                     "  /ocr on|off             → OCR 시작/정지 (직접)\n"
                     "  /stt on|off             → STT 시작/정지 (직접)\n"
                     "  /web <query>           → 웹 검색 이벤트 전송\n"
                     "  /plugins                → 플러그인 엔드포인트 표시\n"
                     "  /event ocr <url>       → OCR 이벤트 URL 변경\n"
                     "  /event stt <url>       → STT 이벤트 URL 변경\n"
                     "  /20B                   → 대화모델을 GPT-OSS-20B로 설정\n"
                     "  /14B                   → 대화모델을 GPT-OSS-14B로 설정\n"
                     "  /clearthink            → 출력의 <Think> 태그 제거\n"
                     )

    def _try_slash(self, text: str) -> bool:
        if not text.startswith("/"):
            return False
        toks = text.strip().split()
        cmd = toks[0].lower()
        if cmd in ("/help", "/도움말"):
            self._help(); return True
        if cmd in ("/대화모드", "/chat"):
            self.set_mode("chat"); return True
        if cmd in ("/대화종료", "/end"):
            self.set_mode("tools"); return True
        if cmd == "/vision":
            self.set_mode("vision"); return True
        if cmd in ("/상태", "/status"):
            self._append("overlay", self._status_line()); return True
        if cmd == "/stats":
            stats = self.event_handler.get_stats()
            msg = f"이벤트 통계:\n"
            msg += f"  전체 수신: {stats['total_received']}개\n"
            msg += f"  에러: {stats['errors']}개 ({stats['error_rate']:.1%})\n"
            if stats['by_type']:
                msg += "  타입별:\n"
                for event_type, count in sorted(stats['by_type'].items()):
                    msg += f"    {event_type}: {count}개\n"
            if stats['last_event_time'] > 0:
                ago = int(time.time() - stats['last_event_time'])
                msg += f"  마지막 이벤트: {ago}초 전"
            self._append("overlay", msg); return True
        if cmd == "/test":
            test_msg = " ".join(toks[1:]) or "테스트 메시지"
            success = self.event_handler.handle_event("test.manual", {"text": test_msg})
            self._append("overlay", f"테스트 이벤트 {'성공' if success else '실패'}"); return True
        if cmd == "/debug" and len(toks) == 2:
            enable = toks[1].lower() in ("on", "true", "1")
            self.event_handler.set_debug_mode(enable)
            self._append("overlay", f"디버그 모드 {'활성화' if enable else '비활성화'}"); return True
        
        # Pin 관련 명령어들
        if cmd == "/pin4b":
            self._set_pin_model("4b"); return True
        if cmd == "/pin14b":
            self._set_pin_model("14b"); return True
        if cmd == "/pin20b":
            self._set_pin_model("20b"); return True
        if cmd == "/pinoff":
            self._clear_pin(); return True
        if cmd in ("/리셋", "/reset"):
            self.orch.reload_memory(); self._append("overlay", "히스토리 초기화 + tool_memory 재적용"); return True
        if cmd == "/memory" and len(toks) >= 2 and toks[1].lower() == "reload":
            self.orch.reload_memory(); self._append("overlay", "tool_memory.txt reloaded"); return True
        if cmd == "/model" and len(toks) >= 3:
            which, name = toks[1].lower(), " ".join(toks[2:])
            if which == "tools":
                self.orch.cfg.setdefault("llm_tools", {})["model"] = name; self._append("overlay", f"tools 모델 변경: {name}"); return True
            if which == "chat":
                self.orch.cfg.setdefault("llm_chat", {})["model"] = name; self._append("overlay", f"chat 모델 변경: {name}"); return True
            if which == "vision":
                self.orch.cfg.setdefault("llm_vision", {})["model"] = name; self._append("overlay", f"vision 모델 변경: {name}"); return True
        if cmd == "/opacity" and len(toks) == 2:
            try:
                val = float(toks[1]); self.setWindowOpacity(max(0.1, min(1.0, val))); self._append("overlay", f"opacity={self.windowOpacity():.2f}"); return True
            except: pass
        if cmd == "/size" and len(toks) == 2 and "x" in toks[1]:
            try:
                w, h = map(int, toks[1].lower().split("x")); self.resize(w, h); self.container.setGeometry(0, 0, w, h); self._append("overlay", f"size={w}x{h}"); return True
            except: pass
        if cmd == "/pos" and len(toks) == 3:
            try:
                x, y = int(toks[1]), int(toks[2]); self.move(x, y); self._append("overlay", f"pos={x},{y}"); return True
            except: pass
        if cmd == "/top" and len(toks) == 2:
            val = toks[1].lower() in ("on","true","1"); flags = self.windowFlags()
            if val: flags |= QtCore.Qt.WindowStaysOnTopHint
            else:   flags &= ~QtCore.Qt.WindowStaysOnTopHint
            self.setWindowFlags(flags); self.show(); self._append("overlay", f"always_on_top={'on' if val else 'off'}"); return True
        if cmd == "/save":
            now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(APP_DIR, f"overlay_log_{now}.txt")
            try:
                open(path, "w", encoding="utf-8").write("\n".join(self.transcript)); self._append("overlay", f"저장됨: {path}")
            except Exception as e: self._append("error", f"저장 실패: {e}")
            return True
        if cmd == "/clearthink":
            self._remove_think_tags(); return True
        if cmd == "/ocr" and len(toks) == 2 and toks[1].lower() in ("on","off"):
            name = 'ocr.start' if toks[1].lower()=='on' else 'ocr.stop'
            self._run_tool_calls_async([{"name": name, "args": {}}])
            return True
        if cmd == "/stt" and len(toks) == 2 and toks[1].lower() in ("on","off"):
            name = 'stt.start' if toks[1].lower()=='on' else 'stt.stop'
            self._run_tool_calls_async([{"name": name, "args": {}}])
            return True
        if cmd == "/web" and len(toks) >= 2:
            q = " ".join(toks[1:]).strip()
            self._run_tool_calls_async([{"name": "web.search", "args": {"q": q, "k": 5}}])
            return True
        if cmd == "/plugins":
            self._show_plugins(); return True
        if cmd == "/event" and len(toks) == 3 and toks[1].lower() in ("ocr","stt"):
            which = toks[1].lower(); url = toks[2]
            self.orch.cfg.setdefault('tool_endpoints', {}).setdefault(which, {})['event_url'] = url
            # also reflect into cfg['tools'] mapping
            tmap = self.orch.cfg.setdefault('tools', {})
            if which == 'ocr':
                tmap['ocr.start'] = url; tmap['ocr.stop'] = url
            elif which == 'stt':
                tmap['stt.start'] = url; tmap['stt.stop'] = url
            self._append('overlay', f"{which.upper()} event URL set → {url}"); return True
        if cmd == "/20b":
            self.orch.cfg.setdefault('llm_chat', {})['model'] = 'GPT-OSS-20B'
            self._append('overlay', "대화모델 변경: GPT-OSS-20B"); return True
        if cmd == "/14b":
            self.orch.cfg.setdefault('llm_chat', {})['model'] = 'GPT-OSS-14B'
            self._append('overlay', "대화모델 변경: GPT-OSS-14B"); return True
        if cmd == "/ping":
            threading.Thread(target=self._ping_llm_both, daemon=True).start(); return True
        if cmd == "/proxy" and len(toks) >= 2 and toks[1].lower() == "url":
            px = self.cfg.get("proxy", {}); self._append("overlay", f"http://{px.get('host','127.0.0.1')}:{int(px.get('port',8350))}/v1/chat/completions"); return True
        self._append("error", "알 수 없는 명령어. /help 를 확인하세요."); return True

    @QtCore.pyqtSlot()
    def on_send(self):
        text = self.input.text().strip()
        if not text and not self.pending_vision_images and self.mode != "vision":
            return
        self.input.clear()
        if text and self._try_slash(text):
            return

        if self.mode == "vision":
            if text:
                self._append("you", text)
            parts = list(self.pending_vision_images)
            if text:
                parts.append({"type": "text", "text": text})
            if not parts:
                return
            messages = [{"role": "user", "content": parts}]
            self.pending_vision_images = []

            def worker(messages=messages):
                try:
                    out = self.orch.chat_vision(messages)
                    self.appended.emit("vision", strip_think_and_fences(out))
                except Exception as e:
                    self.appended.emit("error", f"vision failed: {e}")
            threading.Thread(target=worker, daemon=True).start()
            return

        if text:
            self._append("you", text)

            def worker(msg=text, mode=self.mode):
                try:
                    t0 = time.time()
                    if mode == "chat":
                        self.appended.emit("overlay", "↗ LLM(chat) 요청")
                        out = self.orch.chat(msg)
                        out = strip_think_and_fences(out)
                        dt = int((time.time()-t0)*1000)
                        self.appended.emit("overlay", f"↙ LLM(chat) 응답 ({dt}ms)")
                        self.appended.emit("llm-14B", out or "(empty)")
                        # also try tools (optional)
                        self.appended.emit("overlay", "↗ LLM(tools) 요청")
                        parsed = self.orch.orchestrate(msg)
                        say = strip_think_and_fences(parsed.get("say") or "")
                        tcs = parsed.get("tool_calls") or []
                        if not say and tcs:
                            say = "(tools) " + ", ".join((c or {}).get("name","?") for c in tcs if isinstance(c, dict))
                        dt2 = int((time.time()-t0)*1000)
                        self.appended.emit("overlay", f"↙ LLM(tools) 응답 ({dt2}ms)")
                        self.appended.emit("llm-4B", say or "(no text; tools only)")
                        if bool((self.cfg.get("debug") or {}).get("overlay_echo_raw", False)) and getattr(self.orch, "last_tools_raw", ""):
                            raw = self.orch.last_tools_raw
                            if len(raw) > 4000: raw = raw[:4000] + " ... (truncated)"
                            self.appended.emit("debug", f"raw: {raw}")
                    else:
                        self.appended.emit("overlay", "↗ LLM(tools) 요청")
                        parsed = self.orch.orchestrate(msg)
                        say = strip_think_and_fences(parsed.get("say") or "")
                        tcs = parsed.get("tool_calls") or []
                        if not say and tcs:
                            say = "(tools) " + ", ".join((c or {}).get("name","?") for c in tcs if isinstance(c, dict))
                        dt = int((time.time()-t0)*1000)
                        self.appended.emit("overlay", f"↙ LLM(tools) 응답 ({dt}ms)")
                        self.appended.emit("llm-4B", say or "(no text; tools only)")
                        if bool((self.cfg.get("debug") or {}).get("overlay_echo_raw", False)) and getattr(self.orch, "last_tools_raw", ""):
                            raw = self.orch.last_tools_raw
                            if len(raw) > 4000: raw = raw[:4000] + " ... (truncated)"
                            self.appended.emit("debug", f"raw: {raw}")
                except Exception as e:
                    self.appended.emit("error", str(e))
                    logger.exception("overlay worker error")
            threading.Thread(target=worker, daemon=True).start()

    def on_image(self):
        """이미지 파일을 선택해 비전 모델로 전송"""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if not path:
            return
        try:
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("ascii")
            mime = "image/png"
            if path.lower().endswith(".jpg") or path.lower().endswith(".jpeg"):
                mime = "image/jpeg"
            elif path.lower().endswith(".bmp"):
                mime = "image/bmp"
            self.pending_vision_images.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"}
            })
            self.set_mode("vision")
            self._append("you", f"[이미지] {os.path.basename(path)}")
        except Exception as e:
            self._append("error", f"이미지 로드 실패: {e}")

# ---------------- Tray ----------------


class Tray(QtWidgets.QSystemTrayIcon):
    def __init__(self, app: QtWidgets.QApplication, window: QtWidgets.QWidget):
        pix = QtGui.QPixmap(32, 32)
        pix.fill(QtGui.QColor("#3A78FF"))
        painter = QtGui.QPainter(pix)
        painter.setPen(QtGui.QPen(QtGui.QColor("#FFFFFF")))
        painter.drawText(pix.rect(), QtCore.Qt.AlignCenter, "L")
        painter.end()
        icon = QtGui.QIcon(pix)
        super().__init__(icon, app)
        self.app = app
        self.window = window
        menu = QtWidgets.QMenu()
        act = menu.addAction("Show/Hide (Ctrl+`)")
        act.triggered.connect(self.toggle)
        menu.addSeparator()
        q = menu.addAction("Quit")
        q.triggered.connect(self.app.quit)
        self.setContextMenu(menu)
        self.activated.connect(self._on_activated)
        self.setToolTip("Luna Overlay v9")
        self.show()

    def toggle(self):
        if self.window.isVisible():
            self.window.hide()
        else:
            pos = QtGui.QCursor.pos()
            self.window.move(pos.x()-self.window.width()//2,
                             pos.y()-self.window.height()//2)
            self.window.show()
            self.window.raise_()
            self.window.activateWindow()

    def _on_activated(self, reason):
        if reason == QtWidgets.QSystemTrayIcon.Trigger:
            self.toggle()

# ---------------- Main ----------------


def main():
    # DPI/픽셀 스케일링 개선
    QtWidgets.QApplication.setAttribute(
        QtCore.Qt.AA_EnableHighDpiScaling, True)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

    cfg = load_cfg()
    app = QtWidgets.QApplication(sys.argv)
    dummy = QtWidgets.QWidget()
    tray = Tray(app, dummy)
    win = OverlayWindow(cfg, tray)
    tray.window = win

    if (cfg.get("ui") or {}).get("show_on_start", True):
        win.show()
        win.raise_()
        win.activateWindow()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()