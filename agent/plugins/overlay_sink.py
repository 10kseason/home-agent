"""
Enhanced Overlay Sink Plugin v4: 개선된 이벤트 전달 및 표시
- OCR, STT, LLM, 웹검색 결과를 Overlay에 효과적으로 전달
- 재연결 로직 및 에러 핸들링 개선
- 이벤트 필터링 및 포맷팅 최적화
"""
from typing import List, Dict, Any, Optional
import os, asyncio, time, json
from datetime import datetime

try:
    import httpx
except Exception:
    httpx = None
import requests
from loguru import logger
from . import BasePlugin

# 환경 변수 설정
OVERLAY_HOST = os.environ.get("OVERLAY_HOST", "127.0.0.1")
OVERLAY_PORT = int(os.environ.get("OVERLAY_PORT", "8350"))
OVERLAY_BASE = f"http://{OVERLAY_HOST}:{OVERLAY_PORT}"
OVERLAY_EVENT_URL = os.environ.get("OVERLAY_EVENT_URL", f"{OVERLAY_BASE}/event")
OVERLAY_TOAST_URL = os.environ.get("OVERLAY_TOAST_URL", f"{OVERLAY_BASE}/overlay/event")

# 설정
MAX_TEXT_LENGTH = int(os.environ.get("OVERLAY_MAX_TEXT", "300"))
RETRY_ATTEMPTS = int(os.environ.get("OVERLAY_RETRY", "2"))
TIMEOUT_SECONDS = float(os.environ.get("OVERLAY_TIMEOUT", "3.0"))

class EnhancedOverlaySink(BasePlugin):
    name = "enhanced_overlay_sink"
    handles: List[str] = [
        "stt.",
        "mictrans.",
        "ocr.",
        "capture_assist.",
        "llm.",
        "lm.",
        "web.search",
        "overlay.",
    ]

    def __init__(self):
        self.last_connection_check = 0
        self.connection_ok = True
        self.event_count = 0

    async def _test_connection(self) -> bool:
        """Overlay 연결 상태 확인"""
        now = time.time()
        if now - self.last_connection_check < 30:  # 30초마다 체크
            return self.connection_ok
        
        try:
            health_url = f"{OVERLAY_BASE}/health"
            if httpx:
                async with httpx.AsyncClient(timeout=2.0) as client:
                    r = await client.get(health_url)
                    self.connection_ok = (r.status_code == 200)
            else:
                r = requests.get(health_url, timeout=2.0)
                self.connection_ok = (r.status_code == 200)
            
            self.last_connection_check = now
            if not self.connection_ok:
                logger.warning(f"[overlay_sink] Connection failed to {health_url}")
            return self.connection_ok
            
        except Exception as e:
            self.connection_ok = False
            self.last_connection_check = now
            logger.debug(f"[overlay_sink] Connection test failed: {e}")
            return False

    async def _post_with_retry(self, url: str, payload: Dict[str, Any]) -> bool:
        """재시도 로직이 있는 HTTP POST"""
        if not await self._test_connection():
            logger.debug(f"[overlay_sink] Skipping post - overlay not available")
            return False

        for attempt in range(RETRY_ATTEMPTS + 1):
            try:
                if httpx:
                    async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
                        r = await client.post(url, json=payload)
                        success = (200 <= r.status_code < 300)
                else:
                    def _sync_post():
                        resp = requests.post(url, json=payload, timeout=TIMEOUT_SECONDS)
                        return 200 <= resp.status_code < 300
                    success = await asyncio.to_thread(_sync_post)
                
                if success:
                    self.event_count += 1
                    if self.event_count % 50 == 0:  # 50개마다 로그
                        logger.info(f"[overlay_sink] Sent {self.event_count} events to overlay")
                    return True
                else:
                    logger.warning(f"[overlay_sink] HTTP error on attempt {attempt + 1}")
                    
            except Exception as e:
                if attempt == RETRY_ATTEMPTS:
                    logger.error(f"[overlay_sink] Failed after {RETRY_ATTEMPTS + 1} attempts: {e}")
                else:
                    logger.debug(f"[overlay_sink] Attempt {attempt + 1} failed: {e}")
                await asyncio.sleep(0.5)  # 재시도 전 대기
        
        return False

    def _truncate_text(self, text: str, max_len: int = MAX_TEXT_LENGTH) -> str:
        """텍스트 길이 제한 및 정리"""
        if not text:
            return ""
        
        text = str(text).strip()
        if len(text) <= max_len:
            return text
        
        # 문장 단위로 자르기 시도
        sentences = text[:max_len].split('. ')
        if len(sentences) > 1:
            return '. '.join(sentences[:-1]) + '.'
        
        return text[:max_len-3] + "..."

    def _format_stt_event(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """STT 이벤트 포맷팅"""
        text = payload.get("text", "")
        translation = payload.get("translation", "")
        confidence = payload.get("confidence", 0)
        
        # 메인 텍스트
        display_text = self._truncate_text(text)
        
        # 번역이 있고 원문과 다르면 추가
        if translation and translation.strip() != text.strip():
            trans_text = self._truncate_text(translation, max_len=100)
            display_text += f"\n🔄 {trans_text}"
        
        # 신뢰도가 낮으면 표시
        if confidence > 0 and confidence < 0.7:
            display_text += f" (신뢰도: {confidence:.0%})"
        
        return {
            "type": "stt.result",
            "payload": {
                "text": display_text,
                "original": text,
                "translation": translation,
                "confidence": confidence,
                "assist": payload.get("assist", False)
            }
        }

    def _format_ocr_event(self, event_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """OCR 이벤트 포맷팅"""
        text = payload.get("text", "") or payload.get("ocr", "")
        bbox = payload.get("bbox", [])
        confidence = payload.get("confidence", 0)

        display_text = self._truncate_text(text)

        # 좌표 정보가 있으면 추가
        if bbox and len(bbox) >= 4:
            display_text += f" 📍({bbox[0]:.0f},{bbox[1]:.0f})"

        evt_type = (
            "capture_assist.result" if event_type.startswith("capture_assist.") else "ocr.result"
        )

        return {
            "type": evt_type,
            "payload": {
                "text": display_text,
                "bbox": bbox,
                "confidence": confidence,
                "assist": payload.get("assist", False)
            }
        }

    def _format_llm_event(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """LLM 이벤트 포맷팅"""
        model = payload.get("model", "")
        text = payload.get("text", "") or payload.get("content", "")
        tokens = payload.get("tokens", 0)
        
        # 모델별 레이블
        model_label = "LLM"
        model_lower = model.lower()
        if "qwen3-8b" in model_lower or "8b" in model_lower:
            model_label = "LLM-8B"
        elif "hyperclovax" in model_lower or "1.5b" in model_lower:
            model_label = "LLM-1.5B"
        elif "4b" in model_lower:
            model_label = "LLM-4B"
        elif "14b" in model_lower:
            model_label = "LLM-14B"
        
        display_text = self._truncate_text(text, max_len=200)  # LLM은 좀 더 짧게
        
        # 토큰 수 표시
        if tokens > 0:
            display_text += f" ({tokens}tok)"
        
        return {
            "type": f"llm.{model_label.lower()}",
            "payload": {
                "text": display_text,
                "model": model,
                "model_label": model_label,
                "tokens": tokens
            }
        }

    def _format_search_event(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """웹검색 결과 포맷팅"""
        query = payload.get("query", "")
        items = payload.get("items", [])
        provider = payload.get("provider", "")
        
        if not query or not items:
            return None
        
        # 상위 3개 결과만 표시
        result_text = f"🔍 {query}"
        if provider:
            result_text += f" ({provider})"
        result_text += "\n"
        
        for i, item in enumerate(items[:3], 1):
            title = item.get("title", "")
            snippet = item.get("snippet", "")
            
            # 제목 길이 제한
            title = self._truncate_text(title, max_len=60)
            result_text += f"{i}. {title}"
            
            # 스니펫이 있으면 간단히 추가
            if snippet:
                snippet = self._truncate_text(snippet, max_len=80)
                result_text += f"\n   {snippet}"
            result_text += "\n"
        
        return {
            "type": "web.search.result",
            "payload": {
                "text": result_text.strip(),
                "query": query,
                "provider": provider,
                "count": len(items)
            }
        }

    async def handle(self, event) -> None:
        """이벤트 처리 메인 로직"""
        try:
            event_type = getattr(event, "type", "")
            payload = getattr(event, "payload", {}) or {}
            
            if not event_type:
                return
            
            # 이벤트 타입별 포맷팅
            formatted_event = None
            
            if event_type.startswith("stt.") or event_type.startswith("mictrans."):
                formatted_event = self._format_stt_event(payload)
            elif event_type.startswith("ocr.") or event_type.startswith("capture_assist."):
                formatted_event = self._format_ocr_event(event_type, payload)
            elif event_type.startswith("llm.") or event_type.startswith("lm."):
                formatted_event = self._format_llm_event(payload)
            elif event_type.startswith("web.search"):
                formatted_event = self._format_search_event(payload)
            elif event_type.startswith("overlay."):
                # 직접 전달 (토스트 등)
                formatted_event = {
                    "type": event_type,
                    "payload": payload
                }
            
            if not formatted_event:
                logger.debug(f"[overlay_sink] Unhandled event type: {event_type}")
                return
            
            # 공통 메타데이터 추가
            formatted_event.update({
                "priority": getattr(event, "priority", 5),
                "timestamp": int(time.time() * 1000),
                "source": getattr(event, "source", "agent")
            })
            
            # Overlay로 전송
            target_url = OVERLAY_TOAST_URL if event_type.startswith("overlay.") else OVERLAY_EVENT_URL
            success = await self._post_with_retry(target_url, formatted_event)
            
            if success:
                logger.debug(f"[overlay_sink] Sent {event_type} to overlay")
            else:
                logger.warning(f"[overlay_sink] Failed to send {event_type} to overlay")
                
        except Exception as e:
            logger.error(f"[overlay_sink] Handler error: {e}")

    async def on_shutdown(self):
        """종료 시 통계 출력"""
        logger.info(f"[overlay_sink] Sent total {self.event_count} events to overlay")