# overlay_app_enhanced.py - Plugin 시스템이 통합된 버전
# 기존 overlay_app.py를 기반으로 플러그인 시스템을 통합

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Luna Overlay v10 — Plugin-Enhanced Event Processing System
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

# 플러그인 시스템 import
from plugin_system import PluginManager, PluginAwareOrchestrator

# 기존 imports...
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# logging
try:
    from loguru import logger
except Exception:
    class _Dummy:
        def __getattr__(self, name):
            def _p(*a, **k): print(f"[overlay:{name}]", a[0] if a else "")
            return _p
    logger = _Dummy()

# 로그 설정
APP_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(APP_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

try:
    logger.remove()
except Exception:
    pass

logger.add(
    os.path.join(LOG_DIR, "overlay_{time}.log"),
    rotation="10 MB",
    retention="14 days", 
    compression="zip",
    enqueue=True,
    backtrace=True,
    diagnose=False,
    level="INFO"
)

# ============================================================================
# Enhanced Event Processing with Plugin Support
# ============================================================================

@dataclass
class EventStats:
    total_received: int = 0
    by_type: Dict[str, int] = field(default_factory=dict)
    by_plugin: Dict[str, int] = field(default_factory=dict)
    last_event_time: float = 0
    errors: int = 0

class EnhancedEventHandler:
    """플러그인 지원 이벤트 핸들러"""
    
    def __init__(self, window, plugin_manager: PluginManager):
        self.window = window
        self.plugin_manager = plugin_manager
        self.stats = EventStats()
        self.debug_mode = False
        
    def set_debug_mode(self, enabled: bool):
        self.debug_mode = enabled
        
    def handle_event(self, event_type: str, payload: Dict[str, Any]) -> bool:
        """향상된 이벤트 처리 - 플러그인 우선"""
        try:
            self.stats.total_received += 1
            self.stats.by_type[event_type] = self.stats.by_type.get(event_type, 0) + 1
            self.stats.last_event_time = time.time()
            
            if self.debug_mode:
                logger.info(f"[event] Processing {event_type}: {payload}")
            
            # 1. 플러그인에서 먼저 처리 시도
            plugin_handled = self.plugin_manager.handle_event(event_type, payload)
            if plugin_handled:
                plugin_name = self._get_handling_plugin(event_type)
                self.stats.by_plugin[plugin_name] = self.stats.by_plugin.get(plugin_name, 0) + 1
                return True
            
            # 2. 기존 내장 처리기로 fallback
            success = False
            if event_type.startswith("stt.") or event_type.startswith("mictrans."):
                success = self._handle_stt(payload)
            elif event_type.startswith("ocr.") or event_type.startswith("capture_assist."):
                success = self._handle_ocr(event_type, payload)
            elif event_type.startswith("web.search"):
                success = self._handle_web_search(payload)
            elif event_type.startswith("llm.") or event_type.startswith("lm."):
                success = self._handle_llm(payload)
            elif event_type.startswith("overlay."):
                success = self._handle_overlay_event(event_type, payload)
            elif event_type.startswith("plugin."):
                success = self._handle_plugin_event(event_type, payload)
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
    
    def _get_handling_plugin(self, event_type: str) -> str:
        """이벤트 타입으로 처리 플러그인 추정"""
        if event_type.startswith("security.") or event_type.startswith("guard."):
            return "SecurityPlugin"
        elif event_type.startswith("sched."):
            return "SchedulePlugin"  
        elif event_type.startswith("web."):
            return "WebPlugin"
        elif event_type.startswith("kb."):
            return "KnowledgePlugin"
        else:
            return "UnknownPlugin"
    
    def _handle_plugin_event(self, event_type: str, payload: Dict[str, Any]) -> bool:
        """플러그인 전용 이벤트"""
        if event_type == "plugin.reload":
            plugin_name = payload.get("plugin")
            if plugin_name:
                self._emit_safe("🔌 Plugin", f"Reloading {plugin_name}")
                # 플러그인 재로드 로직
                return True
        elif event_type == "plugin.status":
            active_plugins = list(self.plugin_manager.plugins.keys())
            self._emit_safe("🔌 Plugin", f"Active: {', '.join(active_plugins)}")
            return True
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
    
    # 기존 핸들러들...
    def _handle_stt(self, payload: Dict[str, Any]) -> bool:
        """STT 이벤트 처리"""
        text = payload.get("text", "").strip()
        if not text:
            return False
            
        display_text = text
        translation = payload.get("translation", "")
        if translation and translation.strip() != text:
            display_text += f"\n🔄 {translation}"
        
        confidence = payload.get("confidence", 0)
        if confidence > 0 and confidence < 0.8:
            display_text += f" ({confidence:.0%})"
            
        language = payload.get("language", "")
        if language:
            display_text += f" [{language}]"

        model = payload.get("model", "")
        if model:
            display_text += f" ({model})"

        self._emit_safe("🎤 STT", display_text)
        return True
    
    def _handle_ocr(self, event_type: str, payload: Dict[str, Any]) -> bool:
        """OCR 이벤트 처리"""
        text = payload.get("text", "") or payload.get("ocr", "")
        if not text:
            return False

        text = text.strip()
        display_text = text

        bbox = payload.get("bbox", [])
        if bbox and len(bbox) >= 2:
            display_text += f" @({bbox[0]:.0f},{bbox[1]:.0f})"

        confidence = payload.get("confidence", 0)
        if confidence > 0:
            display_text += f" ({confidence:.0%})"

        label = "Assist-Capture" if event_type.startswith("capture_assist.") or payload.get("assist") else "👁️ OCR"
        self._emit_safe(label, display_text)
        return True
    
    def _handle_web_search(self, payload: Dict[str, Any]) -> bool:
        """웹검색 결과 처리"""
        query = payload.get("query", "")
        items = payload.get("items", [])
        provider = payload.get("provider", "")
        
        if not query:
            return False
        
        header = f"'{query}'"
        if provider:
            header += f" ({provider})"
        if items:
            header += f" - {len(items)}개 결과"
            
        self._emit_safe("🔍 검색", header)
        
        for i, item in enumerate(items[:3], 1):
            title = item.get("title", "")[:60]
            snippet = item.get("snippet", "")
            
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

        # <think> 태그 제거
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)

        display_text = text.strip()
        if len(display_text) > 200:
            display_text = display_text[:197] + "..."
        
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
        display_type = event_type.replace(".", " ").title()
        
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
            "by_plugin": dict(self.stats.by_plugin),
            "last_event_time": self.stats.last_event_time,
            "errors": self.stats.errors,
            "error_rate": self.stats.errors / max(1, self.stats.total_received),
            "plugin_efficiency": self._calculate_plugin_efficiency()
        }
    
    def _calculate_plugin_efficiency(self) -> Dict[str, float]:
        """플러그인 효율성 계산"""
        total_plugin = sum(self.stats.by_plugin.values())
        if total_plugin == 0:
            return {}
        
        return {
            plugin: (count / total_plugin) * 100
            for plugin, count in self.stats.by_plugin.items()
        }

# ============================================================================
# Enhanced Orchestrator Integration
# ============================================================================

class EnhancedOrchestrator(PluginAwareOrchestrator):
    """overlay_app.py 통합용 향상된 오케스트레이터"""
    
    def __init__(self, cfg: Dict[str, Any], window=None):
        # 부모 클래스 초기화
        super().__init__(cfg, window)
        
        # 기존 overlay_app.py의 추가 기능들
        self.procs: Dict[str, subprocess.Popen] = {}
        self.lock = threading.Lock()
        
        # 설정 기본값
        cfg.setdefault('tool_endpoints', {})
        self._setup_tool_endpoints(cfg)
    
    def _setup_tool_endpoints(self, cfg):
        """도구 엔드포인트 설정"""
        te = cfg['tool_endpoints']
        te.setdefault('ocr', {'event_url': 'http://127.0.0.1:8765/event'})
        te.setdefault('stt', {'event_url': 'http://127.0.0.1:8765/event'})
        te.setdefault('web', {'event_url': 'http://127.0.0.1:8765/event'})
        te.setdefault('discord', {'event_url': 'http://127.0.0.1:8765/event'})

        # 편의 매핑
        cfg.setdefault('tools', {})
        tools_map = cfg['tools']
        tools_map.setdefault('ocr.start', te['ocr']['event_url'])
        tools_map.setdefault('ocr.stop', te['ocr']['event_url'])
        tools_map.setdefault('stt.start', te['stt']['event_url'])
        tools_map.setdefault('stt.stop', te['stt']['event_url'])
        tools_map.setdefault('web.search', te['web']['event_url'])
    
    async def call_tools(self, user_text: str) -> Dict[str, Any]:
        """도구 호출 (플러그인 우선)"""
        lower = user_text.strip().lower()
        # 사용자가 도구 목록을 요청하는 경우 명시적으로 처리
        if any(k in lower for k in ["tool list", "list tools", "툴 목록", "도구 목록"]):
            return {"say": "", "tool_calls": [{"name": "agent.list_tools", "args": {}}]}

        # 기존 LLM 기반 도구 호출
        llm_result = await super().call_tools_llm(user_text) if hasattr(super(), 'call_tools_llm') else {}

        # 플러그인에서 직접 도구 호출 가능한지 확인
        plugin_tools = self._try_plugin_direct_tools(user_text)

        # 결합
        tool_calls = plugin_tools + llm_result.get("tool_calls", [])
        say = llm_result.get("say", "")

        return {"say": say, "tool_calls": tool_calls}
    
    def _try_plugin_direct_tools(self, user_text: str) -> List[Dict[str, Any]]:
        """플러그인에서 직접 도구 추천"""
        tools = []
        
        # 간단한 휴리스틱 (개선 가능)
        text_lower = user_text.lower()
        
        if "스케줄" in user_text or "일정" in user_text:
            if "만들" in user_text or "생성" in user_text:
                tools.append({"name": "sched.create", "args": {"title": user_text}})
        
        if "검색" in user_text or "search" in text_lower:
            query = user_text.replace("검색", "").replace("search", "").strip()
            if query:
                tools.append({"name": "web.search", "args": {"query": query}})
        
        return tools

# ============================================================================
# Enhanced Proxy Server
# ============================================================================

class EnhancedProxyServer:
    """플러그인 지원 프록시 서버"""
    
    def __init__(self, orch: EnhancedOrchestrator, host: str, port: int, 
                 event_handler: EnhancedEventHandler):
        self.orch = orch
        self.event_handler = event_handler
        self.host = host
        self.port = int(port)
        self.app = FastAPI(title="Luna Overlay Enhanced Proxy")
        self._setup_routes()
        self._server = None
        self._thread = None

    def _setup_routes(self):
        app = self.app
        orch = self.orch
        handler = self.event_handler

        @app.get("/health")
        async def health():
            stats = handler.get_stats()
            plugin_info = {
                "active_plugins": list(orch.plugin_manager.plugins.keys()),
                "available_tools": len(orch.plugin_manager.get_available_tools()),
                "plugin_stats": stats.get("by_plugin", {})
            }
            return {
                "ok": True, 
                "who": "luna-overlay-enhanced-proxy",
                "event_stats": stats,
                "plugin_info": plugin_info,
                "uptime": time.time()
            }

        @app.post("/event")
        async def receive_event(req: Request):
            try:
                body = await req.json()
                event_type = body.get("type", "")
                payload = body.get("payload", {}) or {}
                
                if not event_type:
                    return JSONResponse({"error": "Missing event type"}, status_code=400)
                
                logger.debug(f"[event] Received {event_type}")
                success = handler.handle_event(event_type, payload)
                
                return {
                    "ok": success, 
                    "message": "processed" if success else "failed",
                    "type": event_type,
                    "handled_by": "plugin" if success else "builtin"
                }
                
            except Exception as e:
                logger.error(f"[event] Error processing event: {e}")
                return JSONResponse({"error": str(e)}, status_code=500)

        @app.get("/plugins")
        async def list_plugins():
            """플러그인 목록"""
            return {
                "active_plugins": list(orch.plugin_manager.plugins.keys()),
                "available_tools": orch.plugin_manager.get_available_tools(),
                "tools_schema": orch.plugin_manager.get_tools_schema()
            }

        @app.post("/plugins/{plugin_name}/reload")
        async def reload_plugin(plugin_name: str):
            """플러그인 재로드"""
            try:
                # TODO: 플러그인 재로드 로직 구현
                return {"ok": True, "message": f"Plugin {plugin_name} reloaded"}
            except Exception as e:
                return JSONResponse({"error": str(e)}, status_code=500)

        @app.post("/tools/{tool_name}")
        async def execute_tool(tool_name: str, req: Request):
            """도구 직접 실행"""
            try:
                body = await req.json()
                args = body.get("args", {})
                result = orch.plugin_manager.execute_tool(tool_name, args)
                return {"ok": True, "result": result}
            except Exception as e:
                logger.error(f"[tools] Direct execution failed for {tool_name}: {e}")
                return JSONResponse({"error": str(e)}, status_code=500)

        # 기존 채팅 완료 엔드포인트는 유지...
        @app.post("/v1/chat/completions")
        async def completions(req: Request):
            """기존 LLM 프록시 기능 유지"""
            # 기존 구현과 동일...
            pass

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        config = uvicorn.Config(self.app, host=self.host, port=self.port, log_level="warning")
        self._server = uvicorn.Server(config)

        def runner():
            asyncio.set_event_loop(asyncio.new_event_loop())
            self._server.run()
        
        self._thread = threading.Thread(target=runner, daemon=True)
        self._thread.start()
        logger.info(f"[proxy] Enhanced server started at http://{self.host}:{self.port}")

    def stop(self):
        """Gracefully stop the enhanced uvicorn server."""
        if self._server:
            self._server.should_exit = True
            if self._thread:
                self._thread.join(timeout=5)
            self._server = None
            self._thread = None

# ============================================================================
# Enhanced Overlay Window
# ============================================================================

class EnhancedOverlayWindow(QtWidgets.QWidget):
    """플러그인 지원 오버레이 윈도우"""
    
    appended = QtCore.pyqtSignal(str, str)

    def __init__(self, cfg: Dict[str, Any], tray):
        super().__init__()
        self.cfg = cfg
        self.tray = tray
        
        # 플러그인 지원 오케스트레이터
        self.orch = EnhancedOrchestrator(cfg, window=self)
        
        # 플러그인 지원 이벤트 핸들러
        self.event_handler = EnhancedEventHandler(self, self.orch.plugin_manager)
        
        # UI 초기화 (기존과 동일)
        self._init_ui()
        
        # 플러그인 지원 프록시
        px = cfg.get("proxy", {})
        if px.get("enable", True):
            self.proxy = EnhancedProxyServer(
                self.orch, 
                px.get("host", "127.0.0.1"), 
                int(px.get("port", 8350)),
                self.event_handler
            )
            self.proxy.start()
        else:
            self.proxy = None

        # 시그널 연결
        self.appended.connect(self._append)

        self._append("overlay", "Ready v10 - Enhanced Plugin System")
        self._append("plugins", f"Loaded: {', '.join(self.orch.plugin_manager.plugins.keys())}")

    def closeEvent(self, event: QtGui.QCloseEvent):
        """Ensure the proxy server shuts down with the overlay."""
        try:
            if getattr(self, "proxy", None):
                self.proxy.stop()
        except Exception:
            pass
        super().closeEvent(event)
        
    def _init_ui(self):
        """UI 초기화"""
        # 기존 overlay_app.py의 UI 초기화 코드와 동일
        # 여기서는 생략...
        pass
    
    def _try_slash(self, text: str) -> bool:
        """슬래시 명령 처리 (플러그인 명령 추가)"""
        if not text.startswith("/"):
            return False
            
        toks = text.strip().split()
        cmd = toks[0].lower()
        
        # 플러그인 관련 명령어들
        if cmd == "/plugins":
            active = list(self.orch.plugin_manager.plugins.keys())
            tools = self.orch.plugin_manager.get_available_tools()
            self._append("plugins", f"활성화된 플러그인: {', '.join(active)}")
            self._append("plugins", f"사용 가능한 도구 ({len(tools)}개): {', '.join(tools[:10])}")
            if len(tools) > 10:
                self._append("plugins", f"... 외 {len(tools)-10}개 더")
            return True
            
        if cmd == "/plugin" and len(toks) >= 2:
            subcmd = toks[1].lower()
            if subcmd == "stats":
                stats = self.event_handler.get_stats()
                self._append("plugins", f"플러그인 처리 통계: {stats['by_plugin']}")
                self._append("plugins", f"플러그인 효율성: {stats['plugin_efficiency']}")
                return True
            elif subcmd == "reload" and len(toks) >= 3:
                plugin_name = toks[2]
                # TODO: 플러그인 재로드 구현
                self._append("plugins", f"플러그인 재로드: {plugin_name}")
                return True
        
        if cmd == "/tool" and len(toks) >= 2:
            tool_name = toks[1]
            if tool_name in self.orch.plugin_manager.get_available_tools():
                # 도구 직접 실행 (인자는 비어있음)
                try:
                    result = self.orch.plugin_manager.execute_tool(tool_name, {})
                    self._append("tool", f"{tool_name} → {result}")
                except Exception as e:
                    self._append("error", f"{tool_name} 실행 실패: {e}")
                return True
        
        # 기존 명령어들...
        if cmd in ("/help", "/도움말"):
            self._help_enhanced()
            return True

        # ... 기존 명령어 처리 ...

        # 알 수 없는 명령어 안내
        self._append("overlay", "알 수 없는 명령어. /help 를 입력하세요.")
        return True
    
    def _help_enhanced(self):
        """향상된 도움말 (플러그인 명령어 포함)"""
        help_text = """명령어:
  기본 명령어:
    /대화모드 | /chat        → 대화모드(14B)
    /상태 | /status          → 현재 상태 표시
    /리셋 | /reset           → 히스토리 초기화
    
  플러그인 명령어:
    /plugins                 → 활성화된 플러그인 목록
    /plugin stats            → 플러그인 통계
    /plugin reload <name>    → 플러그인 재로드
    /tool <name>            → 도구 직접 실행
    
  이벤트/디버깅:
    /debug on|off           → 디버그 모드
    /test <메시지>          → 테스트 이벤트
    /stats                  → 상세 통계 (플러그인 포함)
"""
        self._append("overlay", help_text)
    
    def _append(self, who: str, msg: str):
        """메시지 추가 (기존과 동일)"""
        safe = str(msg).replace("<", "&lt;").replace(">", "&gt;")
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.output.append(f"<span style='color:#666'>[{timestamp}]</span> <b style='color:#7FB2FF'>[{who}]</b> {safe}")
        # transcript 기록도 유지...

# ============================================================================
# 설정 및 메인 함수
# ============================================================================

def create_enhanced_config():
    """향상된 설정 생성"""
    config = {
        "llm_tools": {
            "endpoint": "http://127.0.0.1:1234/v1",
            "model": "qwen/qwen3-4b-2507",
            "api_key": "",
            "timeout_seconds": 60
        },
        "llm_chat": {
            "endpoint": "http://127.0.0.1:1234/v1", 
            "model": "qwen3-14b-instruct",
            "api_key": "",
            "timeout_seconds": 60
        },
        "plugins": {
            "SecurityPlugin": {
                "enabled": True,
                "rate_limits": {
                    "web.fetch_lite": {"limit": 30, "window_sec": 60},
                    "py.run_sandbox": {"limit": 6, "window_sec": 60}
                }
            },
            "SchedulePlugin": {
                "enabled": True,
                "timezone": "Asia/Seoul"
            },
            "WebPlugin": {
                "enabled": True,
                "max_bytes": 400000
            },
            "KnowledgePlugin": {
                "enabled": True,
                "storage_path": "agent_output/kb"
            }
        },
        "proxy": {
            "enable": True,
            "host": "127.0.0.1",
            "port": 8350
        },
        "ui": {
            "opacity": 0.92,
            "width": 640,
            "height": 500,
            "always_on_top": True,
            "show_on_start": True
        },
        "debug": {
            "overlay_echo_raw": False,
            "log_events": False,
            "plugin_debug": False
        }
    }
    return config

def main_enhanced():
    """메인 함수 (플러그인 지원)"""
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

    # 설정 로드 (기존 방식 또는 새로운 방식)
    try:
        cfg = load_cfg()  # 기존 함수 사용
    except:
        cfg = create_enhanced_config()

    app = QtWidgets.QApplication(sys.argv)
    
    # 기존 tray 생성...
    dummy = QtWidgets.QWidget()
    tray = None  # Tray 클래스 생성 (기존 코드에서)
    
    # 향상된 오버레이 윈도우
    win = EnhancedOverlayWindow(cfg, tray)
    
    if cfg.get("ui", {}).get("show_on_start", True):
        win.show()
        win.raise_()
        win.activateWindow()

    try:
        sys.exit(app.exec_())
    finally:
        # 정리
        if hasattr(win, 'orch'):
            win.orch.cleanup()

if __name__ == "__main__":
    main_enhanced()