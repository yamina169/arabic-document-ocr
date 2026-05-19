"""
Abstract base class + shared extraction prompts for all VLM providers.
Add a new model by subclassing VLMProvider and implementing name + extract_structured.

PROMPTS rewritten v2 — based on real CIN image analysis (6 cards inspected).
Key improvements:
  - Spatial anchors: exact pixel-region descriptions per field
  - Positional fallback: if label is unreadable, use position on card
  - Multi-line prenom: explicit join instruction
  - adresse: preserve print order, do NOT reorder RTL tokens
  - numero_registre: strict isolation from corner version numbers (02/03/07)
  - lieu_naissance: attempt hard even if faded — use card position
"""

from __future__ import annotations

import os
import re
import json
import time
import base64
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from difflib import SequenceMatcher

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Shared system prompt
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a precise data-extraction engine specialized in Arabic government "
    "documents. Output ONLY valid JSON — no markdown fences, no explanations, "
    "no comments. Every field value is a non-empty string or the sentinel '?'."
)

# ──────────────────────────────────────────────────────────────────────────────
# Extraction prompts — rewritten v2 from real card inspection
# ──────────────────────────────────────────────────────────────────────────────

EXTRACTION_PROMPTS: dict[str, str] = {

"CIN_FRONT": """
You are reading a Tunisian National Identity Card (بطاقة التعريف الوطنية).
The card template is FIXED and NEVER changes. Use field POSITION on the card
to extract data even when the Arabic label is faded, stamped over, or unclear.

══════════════════════════════════════════════════════
EXACT VISUAL LAYOUT — FRONT SIDE (memorise positions)
══════════════════════════════════════════════════════

  ┌──────────────────────────────────────────┐
  │  [FLAG]   بطاقة التعريف الوطنية  [LOGO] │  ← header — IGNORE
  │                                          │
  │         XXXXXXXX  ← 8-digit CIN number  │  ← zone A  (top-center)
  │                                          │
  │  [PHOTO] اللقب   FAMILY_NAME            │  ← zone B  (right side, row 1)
  │          الاسم   GIVEN_NAME line 1      │  ← zone C  (right side, row 2)
  │                  GIVEN_NAME line 2      │  ← zone C  (continuation)
  │                  GIVEN_NAME line 3      │  ← zone C  (continuation)
  │  تاريخ الولادة   DD MonthAR YYYY        │  ← zone D  (right side, row N-1)
  │  مكانها          CITY                   │  ← zone E  (right side, last row)
  └──────────────────────────────────────────┘

POSITIONAL RULE: The card always has exactly these 5 data rows on the right
column in THIS order: CIN → nom → prenom (1-3 lines) → date → city.
If a label is unreadable, STILL extract the value using its position.

══════════════════════════════════════════════════════
FIELD RULES
══════════════════════════════════════════════════════

1. numero_cin  [zone A — top-center]
   • 8 Latin digits, large font, printed alone in the center.
   • Copy EXACTLY, no spaces, no dashes.
   • These digits are NEVER Arabic — always Latin (0-9).
   • Example: "08660757"

2. nom  [zone B — label اللقب]
   • The Arabic word(s) on the SAME LINE as the label اللقب, to its RIGHT.
   • Typically 1 word. Occasionally 2 for compound names.
   • Copy in Arabic script exactly as printed.

3. prenom  [zone C — label الاسم, may span multiple lines]
   • Start extracting from the word(s) on the SAME LINE as الاسم.
   • CONTINUE onto the next line(s) UNTIL you reach the تاريخ الولادة line.
   • Join all lines into ONE string separated by a single space.
   • ALWAYS include relational particles: بن · بنت · حرم · أرملة
   • CRITICAL: a line starting with أرملة or حرم is STILL part of prenom —
     it is NOT a new field. Include it.
   • Example: "فلانة بنت فلان بن فلان أرملة فلان الفلاني"
   • The prenom ends BEFORE تاريخ الولادة — never include the date.

4. date_naissance  [zone D — label تاريخ الولادة]
   • Format printed on card: DD MonthArabic YYYY
   • Convert STRICTLY to ISO-8601: YYYY-MM-DD
   • Month mapping (Tunisian French-Arabic hybrid):
       جانفي→01  فيفري→02  مارس→03  أفريل→04  ماي→05  جوان→06
       جويلية→07  أوت→08  سبتمبر→09  أكتوبر→10  نوفمبر→11  ديسمبر→12
   • Example: "15 جانفي 1982" → "1982-01-15"

5. lieu_naissance  [zone E — label مكانها — LAST data row]
   • The city name on the LAST text line of the right column, after مكانها.
   • ATTEMPT extraction even if faded. Use card position: it is always the
     bottom-most Arabic text line before any blank space or card edge.
   • Common Tunisian cities for reference (do NOT invent):
     تونس · صفاقس · سوسة · قابس · المهدية  · القيروان · بنزرت
   • If truly unreadable after attempt: set "?" and add to champs_manquants.
   • Example: "المهدية"

══════════════════════════════════════════════════════
IGNORE COMPLETELY
══════════════════════════════════════════════════════
Portrait photo · Tunisian flag · Coat of arms · Header line
Watermark / background patterns · Any pink/red stamps overlaid

══════════════════════════════════════════════════════
OUTPUT — strict JSON only, no markdown, no commentary
══════════════════════════════════════════════════════
{
  "numero_cin":       "00000000",
  "nom":              "فلان",
  "prenom":           "فلانة بنت فلان حرم فلان الفلاني",
  "date_naissance":   "1970-01-01",
  "lieu_naissance":   "تونس",
  "champs_manquants": []
}

SENTINEL RULE: if a field is truly unreadable, use "?" and add its key to
champs_manquants. Never use null, never use empty string "".
Reply with ONLY the JSON object — nothing else.
""",

"CIN_BACK": """
You are reading the BACK of a Tunisian National Identity Card (بطاقة التعريف الوطنية).
The template is FIXED. Use field POSITION even when labels are faded.

══════════════════════════════════════════════════════
EXACT VISUAL LAYOUT — BACK SIDE (memorise positions)
══════════════════════════════════════════════════════

  ┌────────────────────────────────────────────────┐
  │  اسم ولقب الأم   MOTHER_FULL_NAME             │  ← row 1  (top)
  │  المهنة          PROFESSION                   │  ← row 2
  │  العنوان         ADDRESS line 1               │  ← row 3
  │                  ADDRESS line 2  (if present) │  ← row 3 continued
  │  تونس في         DD MonthAR YYYY   [STAMP]    │  ← row 4  (emission date)
  │  GS    Rh                                     │  ← IGNORE
  │                              XXXXXXXX         │  ← registro number (8 digits, bottom-right)
  │  [barcode]                                    │  ← IGNORE
  │  XX (corner)              XX (corner)         │  ← IGNORE — these are 2-digit version codes
  └────────────────────────────────────────────────┘

CRITICAL DISTINCTION for numero_registre:
  • The REAL registry number = 8 digits, printed ALONE near the fingerprint
    (e.g. "00000001", "00000002", "00000003")
  • The corner numbers (02, 03, 06, 07…) are 2-digit version codes — NEVER use these

══════════════════════════════════════════════════════
FIELD RULES
══════════════════════════════════════════════════════

1. nom_mere  [row 1 — label اسم ولقب الأم]
   • Full name after the label on the FIRST line of the card.
   • Include all name particles: بنت · بن · chains if present.
   • Copy in Arabic script exactly as printed.
   • WARNING: the label may appear cut or merged with the name — extract
     ONLY the name value, never include the label words اسم / ولقب / الأم.

2. profession  [row 2 — label المهنة]
   • Text after المهنة on the second line.
   • If the person has no job, the card prints "لاشيء" — output exactly "لاشيء".
     Accept both "لاشيء" and "لا شيء" (with/without space) as the same value.
   • Examples: "عامل يومي" / "لاشيء"

3. adresse  [row 3 — label العنوان — may span 2 lines]
   • Extract ALL address text starting from after the label العنوان.
   • If the address continues on the NEXT line (before the تونس في line),
     JOIN both lines with a single space.
   • CRITICAL: preserve the EXACT PRINT ORDER of words — do NOT reorder
     tokens because of RTL rendering. Copy left-to-right as they appear
     visually on each line, then join lines.
   • Numbers (21, 52, 1…) stay in their visual position — do NOT move them.
   • Example: "نهج فلان 10 حي فلان المدينة"
     NOT: "المدينة فلان حي 10 فلان نهج"

4. date_emission  [row 4 — label تونس في]
   • Text after "تونس في": DD MonthArabic YYYY
   • Convert to ISO-8601: YYYY-MM-DD
   • Month mapping:
       جانفي→01  فيفري→02  مارس→03  أفريل→04  ماي→05  جوان→06
       جويلية→07  أوت→08  سبتمبر→09  أكتوبر→10  نوفمبر→11  ديسمبر→12
   • Example: "15 جانفي 1990" → "1990-01-15"

5. numero_registre  [isolated 8-digit number near fingerprint]
   • Find the standalone 8-digit Latin number printed near or below
     the fingerprint area on the RIGHT side of the card.
   • It is ALWAYS 8 digits (e.g. "00000000").
   • NEVER use the 2-digit corner numbers (02, 03, 06, 07, etc.).
   • NEVER use the barcode digits.
   • If two numbers look like 8 digits, pick the one nearest to the fingerprint.

══════════════════════════════════════════════════════
IGNORE COMPLETELY
══════════════════════════════════════════════════════
Fingerprint image · Official stamp / circular seal · Barcode strip
"GS" and "Rh" text · 2-digit corner numbers (02, 03, 06, 07…)

══════════════════════════════════════════════════════
OUTPUT — strict JSON only, no markdown, no commentary
══════════════════════════════════════════════════════
{
  "nom_mere":         "فلانة بنت فلان",
  "profession":       "عامل يومي",
  "adresse":          "نهج فلان رقم 00 حي فلان تونس",
  "date_emission":    "1970-01-01",
  "numero_registre":  "00000000",
  "champs_manquants": []
}

SENTINEL RULE: if a field is truly unreadable, use "?" and add its key to
champs_manquants. Never use null, never use empty string "".
Reply with ONLY the JSON object — nothing else.
""",

}

SUPPORTED_DOC_TYPES: frozenset[str] = frozenset(EXTRACTION_PROMPTS)


# ──────────────────────────────────────────────────────────────────────────────
# Abstract base class
# ──────────────────────────────────────────────────────────────────────────────

class VLMProvider(ABC):
    """
    Abstract base for all VLM providers.
    Subclass, implement `name` and `extract_structured`, done.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier — e.g. 'groq', 'qwen'."""
        ...

    @abstractmethod
    def extract_structured(self, image_path: str, doc_type: str) -> dict:
        """
        Extract structured fields from image_path.

        Args:
            image_path: Absolute path to the image.
            doc_type:   One of SUPPORTED_DOC_TYPES.

        Returns:
            dict with document fields + 'champs_manquants' list.
            Contains 'erreur' key on failure — never raises.
        """
        ...

    @property
    def call_delay(self) -> float:
        """Seconds to wait between consecutive API calls (used by benchmark runner)."""
        return 0.0

    # ── Shared utilities ──────────────────────────────────────────────────────

    def _encode_image(self, image_path: str, max_px: int = 1024) -> tuple[str, str]:
        """
        Return (base64_data, mime_type).
        If the source file is > 500 KB, forces a resize to max_px and quality=85
        to stay under Groq's 413 limit. Otherwise resizes only when necessary and
        uses quality=92 to preserve OCR detail.
        Applies mild contrast boost to help OCR on faded fields.
        Requires: Pillow (pip install Pillow)
        """
        from PIL import Image, ImageEnhance
        import io

        ext = Path(image_path).suffix.lower()
        mime_map = {
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".jfif": "image/jpeg",
            ".png": "image/png",  ".bmp":  "image/bmp",
            ".tiff": "image/tiff", ".webp": "image/webp",
        }
        mime = mime_map.get(ext, "image/jpeg")

        file_too_large = os.path.getsize(image_path) > 500_000

        img = Image.open(image_path).convert("RGB")

        w, h = img.size
        if file_too_large or max(w, h) > max_px:
            scale = max_px / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        img = ImageEnhance.Contrast(img).enhance(1.2)

        buf = io.BytesIO()
        fmt = "JPEG" if mime == "image/jpeg" else "PNG"
        quality = 85 if file_too_large else 92
        img.save(buf, format=fmt, quality=quality)
        data = base64.b64encode(buf.getvalue()).decode()
        return data, mime

    def _call_groq_text(
        self,
        groq_client,
        model: str,
        user_message: str,
        max_tokens: int = 2048,  # vision calls (groq.py) use 700; text-only LLM calls use 2048
    ) -> str:
        """Call Groq text completion with automatic retry on 429."""
        retry_delays = [5, 15, 30]
        max_retries = len(retry_delays) + 1

        try:
            from groq import RateLimitError
        except ImportError:
            RateLimitError = None

        for attempt in range(max_retries):
            try:
                response = groq_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": user_message},
                    ],
                    temperature=0,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content.strip()

            except Exception as e:
                is_rate_limit = (
                    (RateLimitError and isinstance(e, RateLimitError))
                    or "429" in str(e)
                    or "rate limit" in str(e).lower()
                )
                if is_rate_limit and attempt < max_retries - 1:
                    m = re.search(r"try again in ([\d.]+)s", str(e), re.IGNORECASE)
                    wait = float(m.group(1)) + 1.0 if m else retry_delays[attempt]
                    log.warning("[Groq] 429 — waiting %.1fs (attempt %d/%d)",
                                wait, attempt + 1, max_retries - 1)
                    time.sleep(wait)
                else:
                    raise

        raise RuntimeError("All Groq retries exhausted")

    def _clean_json(self, text: str) -> str:
        text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<think>.*",             "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"```json\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"```\s*", "", text)
        return text.strip()

    def _parse_response(self, raw: str) -> dict:
        """
        Parse VLM text output → clean dict.
        Normalises missing_fields → champs_manquants and auto-detects '?' sentinels.
        Also normalises profession: 'لاشيء' ↔ 'لا شيء' → canonical 'لاشيء'.
        """
        cleaned = self._clean_json(raw)
        try:
            result: dict = json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                try:
                    result = json.loads(match.group())
                except Exception:
                    return {"erreur": "JSON non parseable", "brut": raw[:400], "champs_manquants": []}
            else:
                return {"erreur": "JSON non parseable", "brut": raw[:400], "champs_manquants": []}

        if "missing_fields" in result and "champs_manquants" not in result:
            result["champs_manquants"] = result.pop("missing_fields")
        result.setdefault("champs_manquants", [])

        if "profession" in result and result["profession"] == "لا شيء":
            result["profession"] = "لاشيء"

        manquants: list = result["champs_manquants"]
        for k, v in result.items():
            if k in ("champs_manquants", "missing_fields"):
                continue
            if v == "?" and k not in manquants:
                manquants.append(k)

        return result

    def _levenshtein_ratio(self, s1: str, s2: str) -> float:
        a, b = str(s1).lower().strip(), str(s2).lower().strip()
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        return SequenceMatcher(None, a, b).ratio()

    def _jaccard_score(self, s1: str, s2: str) -> float:
        """Token-level Jaccard — better for Arabic addresses where RTL may reorder words."""
        t1 = set(str(s1).strip().split())
        t2 = set(str(s2).strip().split())
        if not t1 and not t2:
            return 1.0
        if not t1 or not t2:
            return 0.0
        return len(t1 & t2) / len(t1 | t2)
