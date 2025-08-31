from __future__ import annotations

"""Voice command handler for assist mode.

Listens for ``cmd.detected`` events emitted by the STT assist script and
orchestrates follow-up actions such as triggering OCR capture, requesting
summaries or translations from a local LLM, and toggling focus mode. The
plugin keeps short-term state so "번역" after "요약" translates the summary
rather than the original OCR text and "다시" repeats the previous command.
"""

from typing import Optional

import httpx
from loguru import logger

from . import BasePlugin
from agent.schemas import Event


class AssistCommandPlugin(BasePlugin):
    name = "assist_command"
    handles = ["cmd.detected", "ocr.text"]

    def __init__(self, ctx):
        super().__init__(ctx)
        self.last_ocr: str = ""
        self.last_summary: str = ""
        self.prev_cmd: Optional[str] = None

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
                await self.ctx.bus.publish(
                    Event(
                        type="llm.summary",
                        payload={"text": summary, "model": self._summary_model()},
                    )
                )

        elif cmd == "translate":
            text = self.last_summary if self.prev_cmd == "summarize" and self.last_summary else self.last_ocr
            if not text:
                return
            await self._toast("🌐 번역")
            translated = await self._translate(text)
            if translated:
                await self.ctx.bus.publish(
                    Event(
                        type="llm.translation",
                        payload={"text": translated, "model": self._translate_model()},
                    )
                )

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

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "Summarize the text in Korean."},
                {"role": "user", "content": text},
            ],
            "temperature": cfg.get("temperature", 0.1),
            "max_tokens": cfg.get("max_new_tokens", 1024),
        }
        timeout = httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=30.0)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.post(f"{endpoint}/chat/completions", headers=headers, json=payload)
                r.raise_for_status()
                data = r.json()
                return (data["choices"][0]["message"]["content"] or "").strip()
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

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        prompt = f"Translate to Korean. If already Korean, return the original. Text:\n{text}"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful translator."},
                {"role": "user", "content": prompt},
            ],
            "temperature": cfg.get("temperature", 0.2),
            "max_tokens": cfg.get("max_new_tokens", 1024),
        }
        timeout = httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=30.0)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.post(f"{endpoint}/chat/completions", headers=headers, json=payload)
                r.raise_for_status()
                data = r.json()
                return (data["choices"][0]["message"]["content"] or "").strip()
        except Exception as e:
            logger.error(f"[assist_command] translate error: {e}")
            return None

