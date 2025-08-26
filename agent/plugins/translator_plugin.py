from . import BasePlugin
from loguru import logger
from typing import Optional
import httpx, re, asyncio

THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

def strip_think(text: str) -> str:
    return THINK_RE.sub("", text)

class TranslatorPlugin(BasePlugin):
    name = "translator"
    handles = ["ocr.text", "stt.text", "discord.batch", "discord.text", "notif.batch"]

    async def _translate_once(self, endpoint, model, api_key, text: str, target_lang: str) -> Optional[str]:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        prompt = f"Translate to {target_lang}. If already in {target_lang}, return original. Text:\n{text}"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful translator."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        }

        timeout = httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=30.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(f"{endpoint}/chat/completions", headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
            return (data["choices"][0]["message"]["content"] or "").strip()

    async def _translate(self, text: str, target_lang: str = "ko") -> Optional[str]:
        cfg = self.ctx.config.get("translate", {})
        endpoint = cfg.get("endpoint")
        model = cfg.get("model")
        api_key = cfg.get("api_key", "")
        if not endpoint or not model:
            logger.warning("Translate endpoint/model not configured.")
            return None

        # 1) 생각 블럭 제거
        clean = strip_think(text).strip()
        if not clean:
            clean = text.strip()

        # 2) 1차 호출 + 느릴 때 1회 재시도(워밍업 고려)
        try:
            return await self._translate_once(endpoint, model, api_key, clean, target_lang)
        except (httpx.ReadTimeout, httpx.ConnectError) as e:
            logger.warning(f"Translate timeout/connect error, retrying once: {e}")
            await asyncio.sleep(1.0)
            try:
                return await self._translate_once(endpoint, model, api_key, clean[:4000], target_lang)
            except Exception as e2:
                logger.error(f"Translate failed after retry: {e2}")
                return None
        except Exception as e:
            logger.error(f"Translate error: {e}")
            return None

    async def _needs_translation(self, text: str, target_lang: str = "ko") -> bool:
        cfg = self.ctx.config.get("translate", {})
        endpoint = cfg.get("endpoint")
        model = cfg.get("model")
        api_key = cfg.get("api_key", "")
        if not endpoint or not model:
            return True
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        prompt = (
            f"Text:\n{text}\n\nDoes this require translation to {target_lang}?"
            " Answer yes or no."
        )
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You decide if translation is needed."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
        }
        timeout = httpx.Timeout(connect=10.0, read=30.0, write=30.0, pool=30.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(f"{endpoint}/chat/completions", headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
        ans = (data["choices"][0]["message"]["content"] or "").strip().lower()
        return ans.startswith("y")

    async def handle(self, event):
        payload = event.payload or {}
        existing = (payload.get("translation") or "").strip()
        text = payload.get("text") or payload.get("content") or ""
        if not text and not existing:
            return

        translated = existing
        if not translated:
            try:
                if not await self._needs_translation(text, target_lang="ko"):
                    return
            except Exception as e:
                logger.debug(f"[translator] needs_translation error: {e}")
            translated = await self._translate(text, target_lang="ko")
        if not translated:
            # 실패 시: 원문을 그대로 토스트/로그(앱은 멈추지 않게)
            fallback = (strip_think(text) or text)[:180]
            self.ctx.sinks.write_log(f"[{self.name}] translate failed; pass-through: {fallback}",
                                     self.ctx.config["sinks"].get("log_file"))
            if self.ctx.config["sinks"].get("toast", True):
                self.ctx.sinks.toast_notify("번역 실패 (원문 표시)", fallback)
            return

        msg = f"[{self.name}] {event.type} → 번역 완료: {translated[:180]}..."
        self.ctx.sinks.write_log(msg, self.ctx.config["sinks"].get("log_file"))
        if self.ctx.config["sinks"].get("toast", True):
            self.ctx.sinks.toast_notify("번역 완료", translated[:64])