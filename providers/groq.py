"""
Groq Vision provider — LLaMA 4 Scout multimodal.

Requires: GROQ_API_KEY in .env
Model configurable via GROQ_VISION_MODEL.
"""

from __future__ import annotations

import os
import time
import logging

from groq import Groq

from .base import VLMProvider, EXTRACTION_PROMPTS, SYSTEM_PROMPT

log = logging.getLogger(__name__)

_RETRY_DELAYS = [5, 15, 30]
_MAX_RETRIES  = len(_RETRY_DELAYS) + 1


class VLMGroq(VLMProvider):

    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key:
            raise ValueError("GROQ_API_KEY missing — set it in .env")
        self._model = os.getenv("GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
        self._groq  = Groq(api_key=api_key)
        log.info("[Groq] Model: %s", self._model)

    @property
    def name(self) -> str:
        return "groq"

    @property
    def call_delay(self) -> float:
        return 2.5

    def extract_structured(self, image_path: str, doc_type: str) -> dict:
        if doc_type not in EXTRACTION_PROMPTS:
            return {"erreur": f"Type inconnu: {doc_type}", "champs_manquants": []}

        data, mime = self._encode_image(image_path)
        prompt = EXTRACTION_PROMPTS[doc_type]

        try:
            raw = self._call_api(data, mime, prompt)
            result = self._parse_response(raw)

            # One retry if too many unknowns
            if self._count_unknowns(result) > 2:
                log.info("[Groq] Retrying — %d unknown fields", self._count_unknowns(result))
                time.sleep(2)
                raw2 = self._call_api(data, mime, prompt)
                result2 = self._parse_response(raw2)
                if self._count_unknowns(result2) < self._count_unknowns(result):
                    result = result2

            return result

        except Exception as e:
            log.error("[Groq] %s", e, exc_info=True)
            return {"erreur": str(e), "champs_manquants": []}

    def _call_api(self, data: str, mime: str, prompt: str) -> str:
        try:
            from groq import RateLimitError
        except ImportError:
            RateLimitError = None

        for attempt in range(_MAX_RETRIES):
            try:
                response = self._groq.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}},
                                {"type": "text", "text": prompt},
                            ],
                        },
                    ],
                    max_tokens=700,
                    temperature=0,
                )
                return response.choices[0].message.content.strip()

            except Exception as e:
                is_rate_limit = (
                    (RateLimitError and isinstance(e, RateLimitError))
                    or "429" in str(e)
                    or "rate limit" in str(e).lower()
                )
                if is_rate_limit and attempt < _MAX_RETRIES - 1:
                    wait = _RETRY_DELAYS[attempt]
                    log.warning("[Groq] 429 — waiting %ds (%d/%d)", wait, attempt + 1, _MAX_RETRIES - 1)
                    time.sleep(wait)
                else:
                    raise

        raise RuntimeError("[Groq] All retries exhausted")

    def _count_unknowns(self, result: dict) -> int:
        return sum(1 for k, v in result.items()
                   if k not in ("champs_manquants", "erreur", "brut") and v == "?")
