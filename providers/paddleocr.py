"""
PaddleOCR-VL-1.5 via AIStudio Sync API  +  Groq LLM for structured extraction.

Pipeline (2 stages):
    image  ──►  PaddleOCR API (AIStudio)  ──►  markdown text
                                                     │
                                                     ▼
                                          Groq LLM (qwen3-32b)
                                                     │
                                                     ▼
                                    structured JSON  (same schema as all providers)

Requires in .env:
    PADDLEOCR_TOKEN     — AIStudio access token
    PADDLEOCR_SYNC_URL  — sync API endpoint
    GROQ_API_KEY        — used for the LLM extraction step
"""

from __future__ import annotations

import os
import base64
import logging

import requests
from groq import Groq

from .base import VLMProvider, EXTRACTION_PROMPTS

log = logging.getLogger(__name__)

_LLM_MODEL = "qwen/qwen3-32b"


class VLMPaddleOCR(VLMProvider):
    """
    Two-stage provider:
      Stage 1 — PaddleOCR-VL-1.5 (AIStudio) : image → markdown
      Stage 2 — Groq LLM                     : markdown → structured JSON
    """

    def __init__(self):
        self._paddle_token = os.getenv("PADDLEOCR_TOKEN", "")
        self._paddle_url   = os.getenv("PADDLEOCR_SYNC_URL", "")
        groq_key           = os.getenv("GROQ_API_KEY", "")

        if not self._paddle_token:
            raise ValueError("PADDLEOCR_TOKEN missing — get it at https://aistudio.baidu.com/usercenter/token")
        if not self._paddle_url:
            raise ValueError("PADDLEOCR_SYNC_URL missing in .env")
        if not groq_key:
            raise ValueError("GROQ_API_KEY missing — required for the LLM extraction step")

        self._groq = Groq(api_key=groq_key)
        log.info("[PaddleOCR] OCR endpoint : %s", self._paddle_url)
        log.info("[PaddleOCR] LLM model    : %s (Groq)", _LLM_MODEL)

    @property
    def name(self) -> str:
        return "paddleocr"

    @property
    def call_delay(self) -> float:
        return 8.0

    # ── Public interface ───────────────────────────────────────────────────────

    def extract_structured(self, image_path: str, doc_type: str) -> dict:
        if doc_type not in EXTRACTION_PROMPTS:
            return {"erreur": f"Type inconnu: {doc_type}", "champs_manquants": []}

        if not os.path.exists(image_path):
            return {"erreur": f"Fichier introuvable: {image_path}", "champs_manquants": []}

        markdown = self._ocr_to_markdown(image_path)
        if markdown is None:
            return {"erreur": "PaddleOCR API returned no text", "champs_manquants": []}

        log.debug("[PaddleOCR] markdown preview: %s", markdown[:400])
        return self._extract_with_groq(markdown, doc_type)

    # ── Stage 1: PaddleOCR ────────────────────────────────────────────────────

    def _ocr_to_markdown(self, image_path: str) -> str | None:
        try:
            with open(image_path, "rb") as fh:
                file_b64 = base64.b64encode(fh.read()).decode("ascii")
        except OSError as e:
            log.error("[PaddleOCR] Cannot read file: %s", e)
            return None

        payload = {
            "file":                      file_b64,
            "fileType":                  0 if image_path.lower().endswith(".pdf") else 1,
            "useDocOrientationClassify": True,
            "useDocUnwarping":           True,
            "useLayoutDetection":        True,
            "useChartRecognition":       False,
            "prettifyMarkdown":          True,
            "visualize":                 False,
        }
        headers = {
            "Authorization": f"token {self._paddle_token}",
            "Content-Type":  "application/json",
        }

        for attempt in range(2):
            try:
                resp = requests.post(self._paddle_url, json=payload, headers=headers, timeout=180)
                resp.raise_for_status()
                break
            except requests.Timeout:
                if attempt == 0:
                    log.warning("[PaddleOCR] Timeout — retrying once")
                    continue
                log.error("[PaddleOCR] Timeout after retry")
                return None
            except requests.RequestException as e:
                log.error("[PaddleOCR] HTTP error: %s", e)
                return None

        pages = resp.json().get("result", {}).get("layoutParsingResults", [])
        if not pages:
            log.warning("[PaddleOCR] Empty layoutParsingResults")
            return None

        markdown = "\n\n".join(p.get("markdown", {}).get("text", "") for p in pages).strip()
        return markdown if markdown else None

    # ── Stage 2: Groq LLM ─────────────────────────────────────────────────────

    def _extract_with_groq(self, markdown: str, doc_type: str) -> dict:
        user_message = (
            "/no_think\n"
            f"{EXTRACTION_PROMPTS[doc_type]}\n\n"
            "──────────────────────────────────────────\n"
            "The text below was extracted by an OCR engine from the document image.\n"
            "Use it to fill every field as instructed above.\n"
            "──────────────────────────────────────────\n"
            f"OCR TEXT:\n{markdown}"
        )

        try:
            raw = self._call_groq_text(self._groq, _LLM_MODEL, user_message, max_tokens=2048)
        except Exception as e:
            log.error("[PaddleOCR] LLM error: %s", e)
            return {"erreur": f"Groq API error: {e}", "champs_manquants": []}

        log.debug("[PaddleOCR] LLM raw: %s", raw[:300])
        return self._parse_response(raw)
