"""
Benchmarking logic for Arabic Document OCR VLM providers.

Dataset structure:
    test_dataset/
    ├── MANIFEST.json                  ← describes all samples
    ├── cin_front_001.jpg
    ├── cin_front_001_expected.json
    ├── cin_back_001.jpg
    ├── cin_back_001_expected.json
    └── disability_001.jpg
    └── disability_001_expected.json

MANIFEST.json format:
    {
      "samples": [
        {"image": "cin_front_001.jpg", "ground_truth": "cin_front_001_expected.json", "doc_type": "CIN_FRONT"},
        ...
      ]
    }
"""

import json
import time
import logging
from pathlib import Path
from difflib import SequenceMatcher
from typing import Any

log = logging.getLogger(__name__)

_SKIP_FIELDS = {"champs_manquants", "extrait_le", "erreur", "brut"}


# ──────────────────────────────────────────────────────────────────────────────
# Dataset loading
# ──────────────────────────────────────────────────────────────────────────────

def _guess_doc_type(stem: str) -> str:
    s = stem.lower()
    if "front" in s or "recto" in s:
        return "CIN_FRONT"
    if "back" in s or "verso" in s:
        return "CIN_BACK"
    if "disability" in s or "handicap" in s:
        return "DISABILITY"
    return "CIN_FRONT"


def load_dataset(dataset_path: str) -> dict[str, dict]:
    """
    Load test samples from dataset_path.

    Returns:
        {image_abs_path: {"ground_truth": dict, "doc_type": str}}
    """
    base = Path(dataset_path)
    if not base.exists():
        raise FileNotFoundError(f"Dataset path not found: {base}")

    dataset: dict[str, dict] = {}
    manifest = base / "MANIFEST.json"

    if manifest.exists():
        with open(manifest, encoding="utf-8") as f:
            data = json.load(f)
        for entry in data.get("samples", []):
            img_path = base / entry["image"]
            gt_path = base / entry["ground_truth"]
            if not img_path.exists():
                log.warning("Image not found: %s", img_path)
                continue
            if not gt_path.exists():
                log.warning("Ground truth not found: %s", gt_path)
                continue
            with open(gt_path, encoding="utf-8") as f:
                gt = json.load(f)
            dataset[str(img_path.resolve())] = {
                "ground_truth": gt,
                "doc_type": entry.get("doc_type", _guess_doc_type(img_path.stem)),
            }
        log.info("Loaded %d samples from MANIFEST.json", len(dataset))
    else:
        for img_path in sorted(base.glob("*")):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".jfif"}:
                continue
            gt_path = base / (img_path.stem + "_expected.json")
            if not gt_path.exists():
                continue
            with open(gt_path, encoding="utf-8") as f:
                gt = json.load(f)
            dataset[str(img_path.resolve())] = {
                "ground_truth": gt,
                "doc_type": _guess_doc_type(img_path.stem),
            }
        log.info("Auto-discovered %d samples (no MANIFEST.json)", len(dataset))

    return dataset


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────

def levenshtein_ratio(s1: Any, s2: Any) -> float:
    a = str(s1).lower().strip()
    b = str(s2).lower().strip()
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def evaluate_field(extracted: Any, expected: Any) -> float:
    if expected in ("?", None, ""):
        return None  # type: ignore
    if extracted in ("?", None, ""):
        return 0.0
    return levenshtein_ratio(extracted, expected)


def evaluate_document(extracted_doc: dict, expected_doc: dict, doc_type: str, threshold: float = 0.85) -> dict:
    field_scores: dict[str, float] = {}

    for field, expected_val in expected_doc.items():
        if field in _SKIP_FIELDS:
            continue
        extracted_val = extracted_doc.get(field, "?")
        score = evaluate_field(extracted_val, expected_val)
        if score is None:
            continue
        field_scores[field] = score

    accuracy = sum(field_scores.values()) / len(field_scores) if field_scores else 0.0
    return {"field_scores": field_scores, "accuracy": accuracy, "doc_type": doc_type}


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark runner
# ──────────────────────────────────────────────────────────────────────────────

def run_benchmark(
    models: list,
    dataset: dict[str, dict],
    output_dir: str = "benchmark_output",
    threshold: float = 0.85,
) -> dict[str, dict]:
    from pathlib import Path as _P
    _P(output_dir).mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}

    for model in models:
        model_name = model.name
        log.info("[Benchmark] Running model: %s on %d samples", model_name, len(dataset))

        model_data: dict = {
            "model": model_name,
            "documents": [],
            "total_time": 0.0,
            "errors": 0,
        }

        call_delay = getattr(model, "call_delay", 0.0)
        for sample_idx, (img_path, meta) in enumerate(dataset.items()):
            if sample_idx > 0 and call_delay > 0:
                time.sleep(call_delay)

            doc_type = meta["doc_type"]
            ground_truth = meta["ground_truth"]
            img_name = Path(img_path).name

            t0 = time.perf_counter()
            try:
                extracted = model.extract_structured(img_path, doc_type)
                elapsed = time.perf_counter() - t0

                if "erreur" in extracted:
                    raise RuntimeError(extracted["erreur"])

                eval_result = evaluate_document(extracted, ground_truth, doc_type, threshold)
                model_data["documents"].append({
                    "image": img_name,
                    "doc_type": doc_type,
                    "elapsed": round(elapsed, 3),
                    "accuracy": round(eval_result["accuracy"], 4),
                    "field_scores": {k: round(v, 4) for k, v in eval_result["field_scores"].items()},
                    "extracted": extracted,
                    "expected": ground_truth,
                })
                model_data["total_time"] += elapsed
                log.info("  [%s] %s → accuracy=%.3f (%.2fs)", model_name, img_name, eval_result["accuracy"], elapsed)

            except Exception as e:
                elapsed = time.perf_counter() - t0
                model_data["errors"] += 1
                model_data["total_time"] += elapsed
                model_data["documents"].append({
                    "image": img_name,
                    "doc_type": doc_type,
                    "elapsed": round(elapsed, 3),
                    "accuracy": 0.0,
                    "error": str(e),
                })
                log.error("  [%s] %s → ERROR: %s", model_name, img_name, e)

        docs = model_data["documents"]
        accuracies = [d["accuracy"] for d in docs]
        model_data["field_accuracy"] = round(sum(accuracies) / len(accuracies), 4) if accuracies else 0.0
        model_data["doc_accuracy"] = round(sum(1 for a in accuracies if a >= threshold) / len(accuracies), 4) if accuracies else 0.0
        model_data["total_time"] = round(model_data["total_time"], 3)

        results[model_name] = model_data

    return results
