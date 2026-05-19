"""
Lecture code-barres verso CIN.
Optionnel: retourne "?" si absent ou pyzbar/cv2 non installé.
"""
import logging

log = logging.getLogger(__name__)


def read_barcode(image_path: str) -> str:
    try:
        import cv2
        from pyzbar.pyzbar import decode
        img = cv2.imread(str(image_path))
        if img is None:
            return "?"
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        codes = decode(gray)
        if codes:
            return codes[0].data.decode("utf-8", errors="replace")
        return "?"
    except Exception as e:
        log.debug("[Barcode] read failed: %s", e)
        return "?"
