from __future__ import annotations

"""Voice command handler for assist mode.

Listens for ``cmd.detected`` events emitted by the STT assist script and
orchestrates follow-up actions such as triggering OCR capture, requesting
summaries or translations from a local LLM, and toggling focus mode. The
plugin keeps short-term state so "번역" after "요약" translates the summary
rather than the original OCR text and "다시" repeats the previous command.
"""

from . import BasePlugin
from agent.schemas import Event
import json
import hashlib
from typing import Dict, Optional, Tuple

import httpx
from loguru import logger


class AssistCommandPlugin(BasePlugin):
    name = "assist_command"
    handles = ["cmd.detected", "ocr.text"]

    def __init__(self, ctx):
        super().__init__(ctx)
        self.last_ocr: str = ""
        self.last_summary: str = ""
        self.last_translation: str = ""
        self.prev_cmd: Optional[str] = None
        self._cache: Dict[Tuple[str, str], str] = {}

    async def _toast(self, text: str) -> None:
        """Send a small overlay notification."""
        await self.ctx.bus.publish(
            Event(type="overlay.toast", payload={"title": "Assist", "text": text})
        )

    async def handle(self, event):
        if event.type == "ocr.text":
            # Track latest OCR output for summarize/translate commands
            self.last_ocr = event.payload.get("text", "")
            self.last_summary = ""
            self.last_translation = ""
            return

        cmd = event.payload.get("cmd")
        if not cmd:
            return
        await self.ctx.bus.publish(
            Event(type="overlay.toast", payload={"title": "📋 Cmd Detected", "text": cmd})
        )
        await self._execute(cmd)

    async def _execute(self, cmd: str) -> None:
        if cmd == "repeat":
            if self.prev_cmd and self.prev_cmd != "repeat":
                await self._execute(self.prev_cmd)
            return

        if cmd == "capture":
            await self._toast("📸 캡처")
            await self.ctx.bus.publish(Event(type="ocr_assist.start", payload={}))

        elif cmd == "summarize":
            if not self.last_ocr:
                return
            await self._toast("📝 요약")
            summary = await self._summarize(self.last_ocr)
            if summary:
                self.last_summary = summary
                self.last_translation = ""
                await self.ctx.bus.publish(
                    Event(
                        type="llm.summary",
                        payload={"text": summary, "model": self._summary_model()},
                    )
                )

        elif cmd == "translate":
            # If no recent summary exists, perform a combined summarize+translate
            if self.prev_cmd != "summarize" or not self.last_summary:
                if not self.last_ocr:
                    return
                await self._toast("🌐 번역")
                res = await self._summarize_translate(self.last_ocr)
                if res:
                    summary, translated = res
                    self.last_summary = summary
                    self.last_translation = translated
                    model = self._translate_model()
                    await self.ctx.bus.publish(
                        Event(
                            type="llm.summary",
                            payload={"text": summary, "model": model},
                        )
                    )
                    await self.ctx.bus.publish(
                        Event(
                            type="llm.translation",
                            payload={"text": translated, "model": model},
                        )
                    )
                self.prev_cmd = "summarize"
                return

            # Otherwise translate the cached summary
            text = self.last_summary
            if not text:
                return
            await self._toast("🌐 번역")
            translated = await self._translate(text)
            if translated:
                self.last_translation = translated
                await self.ctx.bus.publish(
                    Event(
                        type="llm.translation",
                        payload={"text": translated, "model": self._translate_model()},
                    )
                )
            self.prev_cmd = "summarize"
            return

        elif cmd == "focus":
            await self._toast("🎯 집중모드")
            await self.ctx.bus.publish(Event(type="FocusMode.toggle", payload={"on": True}))

        self.prev_cmd = cmd

    # ----- LLM helpers -----
    def _summary_model(self) -> str:
        return (self.ctx.config.get("llm_summary") or {}).get("model", "")

    def _translate_model(self) -> str:
        return (self.ctx.config.get("translate") or {}).get("model", "")

    async def _summarize(self, text: str) -> Optional[str]:
        cfg = self.ctx.config.get("llm_summary", {})
        endpoint = cfg.get("endpoint")
        model = cfg.get("model")
        api_key = cfg.get("api_key", "")
        if not endpoint or not model:
            logger.warning("[assist_command] llm_summary not configured")
            return None

        text = text[:1200]
        key = hashlib.sha1(text.encode("utf-8")).hexdigest()
        cached = self._cache.get(("SUMMARIZE", key))
        if cached:
            return cached

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "너는 자막/캡처 문서를 두 문장(200자 이내) 으로 핵심만 한국어 요약한다. 고유명사·숫자·단위는 그대로 유지한다. 불확실하면 포함하지 말고 확실한 사실만.",
                },
                {
                    "role": "user",
                    "content": f"[입력 텍스트 시작]\n{text}\n[끝]\n위 내용을 두 문장(200자 이내)으로 한국어 요약해줘.",
                },
            ],
            "temperature": cfg.get("temperature", 0.1),
            "max_tokens": cfg.get("max_new_tokens", 256),
            "stop": ["\n\n", "</end>"],
        }
        timeout = httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=30.0)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.post(f"{endpoint}/chat/completions", headers=headers, json=payload)
                r.raise_for_status()
                data = r.json()
                result = (data["choices"][0]["message"]["content"] or "").strip()
                self._cache[("SUMMARIZE", key)] = result
                return result
        except Exception as e:
            logger.error(f"[assist_command] summarize error: {e}")
            return None

    async def _translate(self, text: str) -> Optional[str]:
        cfg = self.ctx.config.get("translate", {})
        endpoint = cfg.get("endpoint")
        model = cfg.get("model")
        api_key = cfg.get("api_key", "")
        if not endpoint or not model:
            logger.warning("[assist_command] translate not configured")
            return None

        text = text[:1200]
        key = hashlib.sha1(text.encode("utf-8")).hexdigest()
        cached = self._cache.get(("TRANSLATE", key))
        if cached:
            return cached

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "너는 한국어 번역기다. 원문 의미·어조를 유지하고, 고유명사/제품명/수치는 그대로 둔다. 출력만 한국어 문장으로 작성한다. 불필요한 설명·서두 금지.",
                },
                {
                    "role": "user",
                    "content": f"[입력 텍스트 시작]\n{text}\n[끝]\n위 텍스트를 자연스러운 한국어로 번역해줘.",
                },
            ],
            "temperature": cfg.get("temperature", 0.2),
            "max_tokens": cfg.get("max_new_tokens", 256),
            "stop": ["\n\n", "</end>"],
        }
        timeout = httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=30.0)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.post(f"{endpoint}/chat/completions", headers=headers, json=payload)
                r.raise_for_status()
                data = r.json()
                result = (data["choices"][0]["message"]["content"] or "").strip()
                self._cache[("TRANSLATE", key)] = result
                return result
        except Exception as e:
            logger.error(f"[assist_command] translate error: {e}")
            return None

    async def _summarize_translate(self, text: str) -> Optional[tuple[str, str]]:
        """Request a summary and translation in a single LLM call."""
        cfg = self.ctx.config.get("translate", {})
        endpoint = cfg.get("endpoint")
        model = cfg.get("model")
        api_key = cfg.get("api_key", "")
        if not endpoint or not model:
            logger.warning("[assist_command] translate not configured")
            return None

        text = text[:1200]
        key = hashlib.sha1(text.encode("utf-8")).hexdigest()
        sum_cached = self._cache.get(("SUMMARIZE", key))
        tr_cached = self._cache.get(("TRANSLATE", key))
        if sum_cached and tr_cached:
            return sum_cached, tr_cached

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "너는 자막/캡처 문서의 요약과 한국어 번역을 제공한다. "
                        "요약은 두 문장(200자 이내)이며 고유명사·숫자·단위를 유지한다. "
                        "번역은 원문 의미·어조를 유지하고 고유명사/제품명/수치를 그대로 둔다. "
                        "JSON 형식으로 {'summary': '..', 'translation': '..'}만 응답한다."
                    ),
                },
                {"role": "user", "content": f"[입력 텍스트 시작]\n{text}\n[끝]"},
            ],
            "temperature": cfg.get("temperature", 0.2),
            "max_tokens": cfg.get("max_new_tokens", 512),
            "stop": ["\n\n", "</end>"],
        }
        timeout = httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=30.0)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.post(f"{endpoint}/chat/completions", headers=headers, json=payload)
                r.raise_for_status()
                data = r.json()
                content = data["choices"][0]["message"]["content"]
                parsed = json.loads(content)
                summary = (parsed.get("summary") or "").strip()
                translation = (parsed.get("translation") or "").strip()
                self._cache[("SUMMARIZE", key)] = summary
                self._cache[("TRANSLATE", key)] = translation
                return summary, translation
        except Exception as e:
            logger.error(f"[assist_command] summarize+translate error: {e}")
            return None

