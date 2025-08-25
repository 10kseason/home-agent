# -*- coding: utf-8 -*-
"""
Luna Overlay Plugin System
플러그인 기반 확장 시스템
"""
import os
import sys
import json
import yaml
import time
import threading
import subprocess
import platform
import traceback
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path

# 외부 tools 폴더에서 YAML 기반 플러그인을 가져온다
try:
    from tools.overlay_plugin import YamlToolsPlugin
except Exception:
    YamlToolsPlugin = None

# 로깅 설정
try:
    from loguru import logger
except ImportError:
    class _DummyLogger:
        def __getattr__(self, name):
            def _log(*args, **kwargs):
                print(f"[{name}]", args[0] if args else "")
            return _log
    logger = _DummyLogger()

# ============================================================================
# Plugin Interface
# ============================================================================

class BasePlugin(ABC):
    """플러그인 기본 인터페이스"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.name = self.__class__.__name__
        self.enabled = True
        self.handlers = {}
        self._setup()
    
    @abstractmethod
    def _setup(self):
        """플러그인 초기화"""
        pass
    
    def get_handlers(self) -> Dict[str, Callable]:
        """핸들러 딕셔너리 반환"""
        return self.handlers
    
    def get_tools_schema(self) -> List[Dict[str, Any]]:
        """OpenAI 함수 스키마 반환"""
        return []
    
    def on_event(self, event_type: str, payload: Dict[str, Any]) -> bool:
        """이벤트 처리"""
        return False
    
    def cleanup(self):
        """정리 작업"""
        pass

# ============================================================================
# Security Plugin
# ============================================================================

class SecurityPlugin(BasePlugin):
    """보안 관련 기능 플러그인"""
    
    def _setup(self):
        from tools.security.security_plugins import (
            guard_prompt_injection_check, secrets_scan, 
            RateLimiter, ApprovalGate
        )
        
        self.rl = RateLimiter()
        self.gate = ApprovalGate()
        
        # 기본 정책 설정
        self.rl.set_policy("web.fetch_lite", limit=30, window_sec=60)
        self.rl.set_policy("py.run_sandbox", limit=6, window_sec=60)
        
        self.handlers.update({
            "guard.prompt_injection_check": self._guard_check,
            "security.secrets_scan": self._secrets_scan,
            "rate.limit": self._rate_limit,
            "approval.gate": self._approval_gate,
        })
    
    def _guard_check(self, args: Dict[str, Any]):
        from tools.security.security_plugins import guard_prompt_injection_check
        return guard_prompt_injection_check(**args)
    
    def _secrets_scan(self, args: Dict[str, Any]):
        from tools.security.security_plugins import secrets_scan
        return secrets_scan(**args)
    
    def _rate_limit(self, args: Dict[str, Any]):
        key = args["key"]
        limit = args.get("limit")
        window = args.get("window_sec")
        get_only = bool(args.get("get_only"))
        
        if not get_only and (limit is not None or window is not None):
            cur = self.rl.get_policy(key)
            if cur:
                if limit is None: limit = cur.limit
                if window is None: window = cur.window_sec
            self.rl.set_policy(key, int(limit or 0), int(window or 0))
        
        pol = self.rl.get_policy(key)
        if not pol:
            return {"key": key, "limit": 0, "window_sec": 0}
        return {"key": key, "limit": pol.limit, "window_sec": pol.window_sec}
    
    def _approval_gate(self, args: Dict[str, Any]):
        return self.gate.ask(args["title"], args["detail"], 
                           int(args.get("timeout_sec", 45)))
    
    def get_tools_schema(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "guard.prompt_injection_check",
                    "description": "Check text for prompt injection attacks",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string", "description": "Text to check"},
                            "source": {"type": "string", "description": "Source of text"}
                        },
                        "required": ["text"]
                    }
                }
            },
            {
                "type": "function", 
                "function": {
                    "name": "security.secrets_scan",
                    "description": "Scan text for secrets/credentials",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string", "description": "Text to scan"},
                            "mask": {"type": "boolean", "description": "Mask found secrets"}
                        },
                        "required": ["text"]
                    }
                }
            }
        ]

# ============================================================================
# Schedule Plugin
# ============================================================================

class SchedulePlugin(BasePlugin):
    """스케줄링 관련 기능 플러그인"""
    
    def _setup(self):
        from tools.security.security_plugins import MiniScheduler
        
        self.scheduler = MiniScheduler(
            tz="Asia/Seoul", 
            on_fire=self._on_schedule_fire
        )
        
        self.handlers.update({
            "sched.create": self._sched_create,
            "sched.cancel": self._sched_cancel,
            "sched.list": self._sched_list,
            "sched.persist": self._sched_persist,
        })
    
    def _on_schedule_fire(self, task):
        logger.info(f"[Schedule] Fired: {task.title} - {task.payload}")
        # 이벤트를 overlay에 전송할 수 있음
        
    def _sched_create(self, args: Dict[str, Any]):
        return self.scheduler.create(**args)
    
    def _sched_cancel(self, args: Dict[str, Any]):
        return self.scheduler.cancel(args["id"])
    
    def _sched_list(self, args: Dict[str, Any]):
        return self.scheduler.list()
    
    def _sched_persist(self, args: Dict[str, Any]):
        """Windows Task Scheduler 통합"""
        title = args["title"]
        delete = bool(args.get("delete", False))
        
        if delete:
            cmd = f'schtasks /Delete /TN "{title}" /F'
            r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            return {"deleted": r.returncode == 0, "stdout": r.stdout, "stderr": r.stderr}
        
        time_s = args.get("time") or "12:00"
        every = (args.get("every") or "").upper()
        cmdline = args.get("cmd") or 'cmd /c echo hello-from-sched.persist'
        
        if every.startswith("MINUTE:"):
            n = every.split(":", 1)[1]
            sch = f'schtasks /Create /TN "{title}" /SC minute /MO {n} /TR {cmdline} /F'
        elif every == "HOURLY":
            sch = f'schtasks /Create /TN "{title}" /SC hourly /TR {cmdline} /F'
        else:
            sch = f'schtasks /Create /TN "{title}" /SC daily /ST {time_s} /TR {cmdline} /F'
        
        r = subprocess.run(sch, shell=True, capture_output=True, text=True)
        return {"created": r.returncode == 0, "stdout": r.stdout, "stderr": r.stderr}
    
    def get_tools_schema(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "sched.create",
                    "description": "Create a scheduled task",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string", "description": "Task title"},
                            "run_at": {"type": "string", "description": "ISO datetime to run"},
                            "interval_sec": {"type": "integer", "description": "Repeat interval in seconds"},
                            "payload": {"type": "object", "description": "Task payload"}
                        },
                        "required": ["title"]
                    }
                }
            }
        ]

# ============================================================================
# Web Plugin
# ============================================================================

class WebPlugin(BasePlugin):
    """웹 관련 기능 플러그인"""
    
    def _setup(self):
        import httpx
        self.client = httpx.AsyncClient(timeout=10.0)
        
        self.handlers.update({
            "web.fetch_lite": self._fetch_lite,
            "web.search": self._web_search,
        })
    
    def _fetch_lite(self, args: Dict[str, Any]):
        """경량 웹페이지 가져오기"""
        import urllib.parse
        import urllib.request
        import html
        import re
        
        url = args["url"]
        allowed = set(args.get("allowed_domains") or [])
        max_bytes = int(args.get("max_bytes", 400_000))
        
        host = urllib.parse.urlparse(url).hostname or ""
        if allowed and not any(host.endswith(d) for d in allowed):
            raise RuntimeError(f"Domain not allowed: {host}")
        
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = resp.read(max_bytes)
        
        text = data.decode("utf-8", errors="ignore")
        text = re.sub(r"(?is)<script.*?</script>|<style.*?</style>", " ", text)
        text = re.sub(r"(?s)<[^>]+>", " ", text)
        text = html.unescape(re.sub(r"\s+", " ", text)).strip()
        
        return {"url": url, "text": text[:200000], "bytes": len(data)}
    
    def _web_search(self, args: Dict[str, Any]):
        """웹 검색 (구현 예제)"""
        query = args.get("q", "")
        if not query:
            return {"error": "Query required"}
        
        # 실제 검색 API 호출 로직
        return {
            "query": query,
            "results": [
                {"title": "Example Result", "url": "https://example.com", "snippet": "Example snippet"}
            ]
        }
    
    def get_tools_schema(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "web.fetch_lite",
                    "description": "Fetch webpage content (lightweight)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string", "description": "URL to fetch"},
                            "max_bytes": {"type": "integer", "description": "Max bytes to read"}
                        },
                        "required": ["url"]
                    }
                }
            }
        ]
    
    def cleanup(self):
        if hasattr(self, 'client'):
            import asyncio
            try:
                asyncio.run(self.client.aclose())
            except:
                pass

# ============================================================================
# Knowledge Base Plugin
# ============================================================================

class KnowledgePlugin(BasePlugin):
    """지식베이스 관련 기능 플러그인"""
    
    def _setup(self):
        import uuid
        from pathlib import Path
        
        self.kb_dir = Path("agent_output/kb")
        self.kb_dir.mkdir(parents=True, exist_ok=True)
        
        self.handlers.update({
            "kb.ingest": self._kb_ingest,
            "kb.search": self._kb_search,
        })
    
    def _kb_ingest(self, args: Dict[str, Any]):
        """텍스트를 지식베이스에 저장"""
        import uuid
        
        text = args.get("text") or ""
        if not text:
            return {"ok": False, "error": "missing text"}
        
        rid = str(uuid.uuid4())[:8]
        path = self.kb_dir / f"{rid}.txt"
        path.write_text(text, encoding="utf-8")
        
        return {"ok": True, "id": rid, "path": str(path)}
    
    def _kb_search(self, args: Dict[str, Any]):
        """지식베이스 검색"""
        query = args.get("query") or ""
        top_k = int(args.get("top_k") or 5)
        
        if not query:
            return {"ok": False, "error": "missing query"}
        
        results = []
        for fp in self.kb_dir.glob("*.txt"):
            try:
                t = fp.read_text(encoding="utf-8", errors="ignore")
                score = sum(t.lower().count(q.strip().lower()) 
                           for q in query.split() if q.strip())
                if score > 0:
                    results.append({
                        "path": str(fp), 
                        "score": float(score), 
                        "preview": t[:160]
                    })
            except:
                pass
        
        results.sort(key=lambda x: x["score"], reverse=True)
        return {"ok": True, "items": results[:top_k]}
    
    def get_tools_schema(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "kb.ingest",
                    "description": "Store text in knowledge base",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string", "description": "Text to store"}
                        },
                        "required": ["text"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "kb.search", 
                    "description": "Search knowledge base",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                            "top_k": {"type": "integer", "description": "Number of results"}
                        },
                        "required": ["query"]
                    }
                }
            }
        ]

# ============================================================================
# Plugin Manager
# ============================================================================

class PluginManager:
    """플러그인 관리자"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.plugins: Dict[str, BasePlugin] = {}
        self.handlers: Dict[str, Callable] = {}
        self.tools_schema: List[Dict[str, Any]] = []
        
        # 기본 플러그인들 등록
        self._register_default_plugins()
    
    def _register_default_plugins(self):
        """기본 플러그인들 등록"""
        default_plugins = []
        if YamlToolsPlugin:
            default_plugins.append(YamlToolsPlugin)
        default_plugins.extend([
            SecurityPlugin,
            SchedulePlugin,
            WebPlugin,
            KnowledgePlugin,
        ])
        
        for plugin_cls in default_plugins:
            try:
                plugin_name = plugin_cls.__name__
                plugin_config = self.config.get('plugins', {}).get(plugin_name, {})
                
                if plugin_config.get('enabled', True):
                    self.register_plugin(plugin_cls, plugin_config)
                    logger.info(f"[Plugin] Loaded: {plugin_name}")
                else:
                    logger.info(f"[Plugin] Disabled: {plugin_name}")
                    
            except Exception as e:
                logger.error(f"[Plugin] Failed to load {plugin_cls.__name__}: {e}")
    
    def register_plugin(self, plugin_cls, config: Dict[str, Any] = None):
        """플러그인 등록"""
        plugin = plugin_cls(config)
        plugin_name = plugin.name
        
        self.plugins[plugin_name] = plugin
        
        # 핸들러 등록
        for name, handler in plugin.get_handlers().items():
            self.handlers[name] = handler
        
        # 스키마 등록
        schema = plugin.get_tools_schema()
        if schema:
            self.tools_schema.extend(schema)
    
    def unregister_plugin(self, plugin_name: str):
        """플러그인 해제"""
        if plugin_name in self.plugins:
            plugin = self.plugins[plugin_name]
            
            # 핸들러 제거
            for name in plugin.get_handlers().keys():
                self.handlers.pop(name, None)
            
            # 정리
            plugin.cleanup()
            del self.plugins[plugin_name]
            
            logger.info(f"[Plugin] Unregistered: {plugin_name}")
    
    def execute_tool(self, name: str, args: Dict[str, Any]) -> Any:
        """도구 실행"""
        handler = self.handlers.get(name)
        if not handler:
            raise ValueError(f"Tool not found: {name}")
        
        try:
            return handler(args)
        except Exception as e:
            logger.error(f"[Plugin] Tool '{name}' failed: {e}")
            raise
    
    def handle_event(self, event_type: str, payload: Dict[str, Any]) -> bool:
        """이벤트를 모든 플러그인에 전파"""
        handled = False
        for plugin in self.plugins.values():
            try:
                if plugin.on_event(event_type, payload):
                    handled = True
            except Exception as e:
                logger.error(f"[Plugin] Event handling failed in {plugin.name}: {e}")
        return handled
    
    def get_available_tools(self) -> List[str]:
        """사용 가능한 도구 목록"""
        return list(self.handlers.keys())
    
    def get_tools_schema(self) -> List[Dict[str, Any]]:
        """OpenAI 함수 스키마 목록"""
        return self.tools_schema
    
    def cleanup(self):
        """모든 플러그인 정리"""
        for plugin in self.plugins.values():
            try:
                plugin.cleanup()
            except Exception as e:
                logger.error(f"[Plugin] Cleanup failed for {plugin.name}: {e}")
        self.plugins.clear()
        self.handlers.clear()

# ============================================================================
# Enhanced Orchestrator with Plugin Support
# ============================================================================

class PluginAwareOrchestrator:
    """플러그인 지원 오케스트레이터"""
    
    def __init__(self, config: Dict[str, Any], window=None):
        self.config = config
        self.window = window
        self.plugin_manager = PluginManager(config)
        
        # 기존 Orchestrator 기능들...
        self.tool_memory = self._load_tool_memory()
        self.history_tools = [{"role": "system", "content": self._system_prompt_tools()}]
        self.history_chat = [{"role": "system", "content": self._system_prompt_chat()}]
        
        logger.info(f"[Orchestrator] Loaded with {len(self.plugin_manager.get_available_tools())} tools")
    
    def _load_tool_memory(self) -> str:
        """도구 메모리 로드"""
        try:
            with open("tool_memory.txt", "r", encoding="utf-8") as f:
                return f.read()
        except:
            return "You are a tool orchestrator. Use available plugins to help users."
    
    def _system_prompt_tools(self) -> str:
        """도구용 시스템 프롬프트"""
        available_tools = self.plugin_manager.get_available_tools()
        tools_list = "\n".join(f"- {tool}" for tool in available_tools)
        
        return f"""You are a Tool Orchestrator with plugin support. 
Available tools:
{tools_list}

Respond with JSON: {{"say": "...", "tool_calls": [{{"name": "...", "args": {{}}}}]}}
"""
    
    def _system_prompt_chat(self) -> str:
        """채팅용 시스템 프롬프트"""
        return "You are a friendly assistant. Keep replies concise and helpful."
    
    def execute_tool_calls(self, tool_calls: List[Dict[str, Any]]):
        """도구 호출 실행 (플러그인 매니저 사용)"""
        results = []
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
                
            name = call.get("name", "")
            args = call.get("args", {}) or {}
            
            try:
                result = self.plugin_manager.execute_tool(name, args)
                results.append({"name": name, "result": result})
                logger.info(f"[Orchestrator] Tool '{name}' executed successfully")
            except Exception as e:
                error_msg = str(e)
                results.append({"name": name, "error": error_msg})
                logger.error(f"[Orchestrator] Tool '{name}' failed: {error_msg}")
        
        return results
    
    def handle_event(self, event_type: str, payload: Dict[str, Any]) -> bool:
        """이벤트 처리 (플러그인에 위임)"""
        return self.plugin_manager.handle_event(event_type, payload)
    
    def get_available_tools(self) -> List[str]:
        """사용 가능한 도구 목록"""
        return self.plugin_manager.get_available_tools()
    
    def get_tools_schema(self) -> List[Dict[str, Any]]:
        """도구 스키마 목록"""  
        return self.plugin_manager.get_tools_schema()
    
    def cleanup(self):
        """정리"""
        self.plugin_manager.cleanup()

# ============================================================================
# Usage Example
# ============================================================================

def create_enhanced_overlay_config():
    """향상된 오버레이 설정 예제"""
    return {
        "llm_tools": {
            "endpoint": "http://127.0.0.1:1234/v1",
            "model": "qwen3-4b-instruct",
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
                "max_bytes": 400000,
                "allowed_domains": []
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
            "always_on_top": True
        }
    }

if __name__ == "__main__":
    # 플러그인 시스템 테스트
    config = create_enhanced_overlay_config()
    orchestrator = PluginAwareOrchestrator(config)
    
    print("Available tools:")
    for tool in orchestrator.get_available_tools():
        print(f"  - {tool}")
    
    print("\nTools schema:")
    for schema in orchestrator.get_tools_schema():
        print(f"  - {schema['function']['name']}: {schema['function']['description']}")
    
    # 정리
    orchestrator.cleanup()
