"""
Qwen2.5-VL via OpenRouter API.

Model: qwen/qwen2.5-vl-72b-instruct (configurable via OPENROUTER_QWEN_MODEL in .env)
Requires: OPENROUTER_API_KEY in .env
"""

from __future__ import annotations

import os
import time
import logging

from openai import OpenAI

from .base import VLMProvider, EXTRACTION_PROMPTS, SYSTEM_PROMPT

log = logging.getLogger(__name__)

_RETRY_DELAYS = [5, 15, 30]
_MAX_RETRIES  = len(_RETRY_DELAYS) + 1


class VLMQwen(VLMProvider):

    def __init__(self):
        api_key = os.getenv("OPENROUTER_API_KEY", "")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY missing — set it in .env")
        self._model = os.getenv("OPENROUTER_QWEN_MODEL", "qwen/qwen2.5-vl-72b-instruct")
        self._client = OpenAI(
            api_key=api_key,
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        )
        log.info("[Qwen] Using OpenRouter model: %s", self._model)

    @property
    def name(self) -> str:
        return "qwen"

    def extract_structured(self, image_path: str, doc_type: str) -> dict:
        if doc_type not in EXTRACTION_PROMPTS:
            return {"erreur": f"Type inconnu: {doc_type}", "champs_manquants": []}

        data, mt = self._encode_image(image_path)
        prompt = EXTRACTION_PROMPTS[doc_type]

        for attempt in range(_MAX_RETRIES):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": f"data:{mt};base64,{data}"}},
                                {"type": "text", "text": prompt},
                            ],
                        },
                    ],
                    max_tokens=700,
                    temperature=0.0,
                )
                raw = response.choices[0].message.content.strip()
                log.debug("[Qwen] Raw: %s", raw[:300])
                return self._parse_response(raw)

            except Exception as e:
                is_rate_limit = "429" in str(e) or "rate limit" in str(e).lower()
                if is_rate_limit and attempt < _MAX_RETRIES - 1:
                    wait = _RETRY_DELAYS[attempt]
                    log.warning("[Qwen] 429 — waiting %ds (%d/%d)", wait, attempt + 1, _MAX_RETRIES - 1)
                    time.sleep(wait)
                else:
                    log.error("[Qwen] %s", e)
                    return {"erreur": str(e), "champs_manquants": []}

        return {"erreur": "All retries exhausted", "champs_manquants": []}
