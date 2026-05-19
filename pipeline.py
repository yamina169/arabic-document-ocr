"""
Model-agnostic processing pipeline.

image file → VLM extraction → dict
"""

from __future__ import annotations

import logging
from pathlib import Path

from utils.barcode import read_barcode
from providers.base import VLMProvider, SUPPORTED_DOC_TYPES

log = logging.getLogger(__name__)


def run(image_path: Path, doc_type: str, model: VLMProvider) -> dict:
    """
    Run VLM extraction on image_path.

    Args:
        image_path: Path to the uploaded image file.
        doc_type:   "CIN_FRONT" or "CIN_BACK".
        model:      Any VLMProvider instance.

    Returns:
        Extracted dict with document fields + 'champs_manquants'.
        Contains 'erreur' key on failure — never raises.
    """
    doc_type = doc_type.upper()
    if doc_type not in SUPPORTED_DOC_TYPES:
        return {
            "erreur": f"Type inconnu: {doc_type}. Supportés: {sorted(SUPPORTED_DOC_TYPES)}",
            "champs_manquants": [],
        }

    log.info("[Pipeline] %s  doc_type=%s  model=%s", Path(image_path).name, doc_type, model.name)

    result = model.extract_structured(str(image_path), doc_type)

    if doc_type == "CIN_BACK":
        barcode = read_barcode(str(image_path))
        if barcode != "?":
            result["code_barres"] = barcode

    return result
