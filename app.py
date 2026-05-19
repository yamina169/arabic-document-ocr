"""
Arabic Document OCR API v3.0
Model-agnostic — one endpoint per model, per document type.

POST /process/<model>/<doc-type>          extract fields from image
GET  /result/<model>/<doc-type>/<stem>    retrieve one saved result
GET  /results/<model>/<doc-type>          list all results for model+type
GET  /health                              API + key status
GET  /                                    endpoint map

Supported models:    groq | qwen
Supported doc-types: cin-front | cin-back
Upload field:        'image' (multipart/form-data)
"""

from __future__ import annotations

import os
import json
import uuid
import logging
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename

import pipeline
from config import UPLOAD_DIR, OUTPUT_DIR, ALLOWED_EXTENSIONS, MAX_FILE_SIZE

load_dotenv()
log = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# ── Constants ─────────────────────────────────────────────────────────────────

_DOC_TYPE_MAP: dict[str, str] = {
    "cin-front": "CIN_FRONT",
    "cin-back":  "CIN_BACK",
}

SUPPORTED_MODELS: frozenset[str] = frozenset({"groq", "qwen"})

# Lazy-initialized model singletons — created on first request
_model_cache: dict[str, object] = {}


def _get_model(name: str):
    if name not in _model_cache:
        if name == "groq":
            from providers.groq import VLMGroq
            _model_cache["groq"] = VLMGroq()
        elif name == "qwen":
            from providers.qwen import VLMQwen
            _model_cache["qwen"] = VLMQwen()
        else:
            raise KeyError(f"Unknown model: {name}")
    return _model_cache[name]


# ── File helpers ──────────────────────────────────────────────────────────────

def _check_file(file) -> tuple[bool, str]:
    if not file or not file.filename:
        return False, "Aucun fichier fourni"
    if Path(file.filename).suffix.lower() not in ALLOWED_EXTENSIONS:
        return False, f"Format non supporté. Acceptés: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
    if len(file.read()) > MAX_FILE_SIZE:
        return False, "Fichier trop grand (max 20 MB)"
    file.seek(0)
    return True, ""


def _save_upload(file, prefix: str) -> Path:
    ext = Path(file.filename).suffix.lower()
    name = secure_filename(f"{prefix}_{uuid.uuid4().hex[:8]}{ext}")
    path = UPLOAD_DIR / name
    file.save(str(path))
    return path


def _result_path(model: str, doc_type: str, image_stem: str) -> Path:
    """output/<model>/<doc-type>/<image-stem>.json — unique key = image filename."""
    return OUTPUT_DIR / model / doc_type / f"{image_stem}.json"


def _save_result(data: dict, path: Path) -> None:
    """Overwrite result — same image processed again → same file, no duplicates."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data["extrait_le"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ── Process endpoint ──────────────────────────────────────────────────────────

@app.route("/process/<model>/<doc_type>", methods=["POST"])
def process(model: str, doc_type: str):
    """Extract fields from an uploaded document image."""
    if model not in SUPPORTED_MODELS:
        return jsonify({
            "erreur": f"Modèle inconnu: '{model}'. Disponibles: {sorted(SUPPORTED_MODELS)}"
        }), 404

    if doc_type not in _DOC_TYPE_MAP:
        return jsonify({
            "erreur": f"Type inconnu: '{doc_type}'. Disponibles: {sorted(_DOC_TYPE_MAP)}"
        }), 404

    if "image" not in request.files:
        return jsonify({"erreur": "Champ 'image' manquant dans la requête"}), 400

    ok, err = _check_file(request.files["image"])
    if not ok:
        return jsonify({"erreur": err}), 400

    try:
        image_file = request.files["image"]
        image_stem = Path(image_file.filename).stem        # unique key per image
        upload_path = _save_upload(image_file, f"{model}_{doc_type}")

        vlm = _get_model(model)
        result = pipeline.run(upload_path, _DOC_TYPE_MAP[doc_type], vlm)

        _save_result(result, _result_path(model, doc_type, image_stem))
        return jsonify(result), 200

    except KeyError as e:
        return jsonify({
            "erreur": f"Modèle '{model}' non disponible — vérifier les clés API dans .env",
            "detail": str(e),
        }), 503
    except Exception as e:
        log.exception("[API] Unhandled error")
        return jsonify({"erreur": str(e), "champs_manquants": []}), 500


# ── Result endpoints ──────────────────────────────────────────────────────────

@app.route("/result/<model>/<doc_type>/<image_stem>", methods=["GET"])
def get_result(model: str, doc_type: str, image_stem: str):
    """Retrieve the saved result for one specific image."""
    path = _result_path(model, doc_type, image_stem)
    if not path.exists():
        return jsonify({
            "erreur": (
                f"Aucun résultat pour '{image_stem}'. "
                f"Traiter d'abord via POST /process/{model}/{doc_type}"
            )
        }), 404
    with open(path, encoding="utf-8") as f:
        return jsonify(json.load(f)), 200


@app.route("/results/<model>/<doc_type>", methods=["GET"])
def list_results(model: str, doc_type: str):
    """List all saved results for a model + doc-type combination."""
    folder = OUTPUT_DIR / model / doc_type
    if not folder.exists():
        return jsonify([]), 200
    results = []
    for p in sorted(folder.glob("*.json")):
        with open(p, encoding="utf-8") as f:
            results.append(json.load(f))
    return jsonify(results), 200


# ── Health & root ─────────────────────────────────────────────────────────────

@app.route("/dashboard", methods=["GET"])
def dashboard():
    return render_template("dashboard.html")


@app.route("/api/results", methods=["GET"])
def api_results():
    path = OUTPUT_DIR.parent / "benchmark_output" / "all_results.json"
    if not path.exists():
        return jsonify({}), 200
    with open(path, encoding="utf-8") as f:
        return jsonify(json.load(f)), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "models": {
            "groq":   "✓" if os.getenv("GROQ_API_KEY")       else "⚠ GROQ_API_KEY manquant",
            "qwen":   "✓" if os.getenv("OPENROUTER_API_KEY") else "⚠ OPENROUTER_API_KEY manquant",
        },
        "output_dir": str(OUTPUT_DIR),
        "architecture": "Image → Preprocessing → VLM → JSON (1 appel par image)",
    }), 200


@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "api":       "Arabic Document OCR v3.0",
        "models":    sorted(SUPPORTED_MODELS),
        "doc_types": sorted(_DOC_TYPE_MAP),
        "endpoints": {
            "POST /process/<model>/<doc-type>":        "model: groq|qwen  doc-type: cin-front|cin-back",
            "GET  /result/<model>/<doc-type>/<stem>":  "retrieve result by original image filename (no extension)",
            "GET  /results/<model>/<doc-type>":        "list all results for model + doc-type",
            "GET  /health":                            "API + key status",
        },
        "upload_field": "image",
    }), 200


@app.errorhandler(404)
def not_found(e):
    return jsonify({"erreur": "Endpoint non trouvé"}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"erreur": "Erreur interne du serveur"}), 500


if __name__ == "__main__":
    port  = int(os.getenv("FLASK_PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "True").lower() == "true"
    print(f"\n  API:    http://localhost:{port}")
    print(f"  Health: http://localhost:{port}/health\n")
    app.run(debug=debug, port=port, host="0.0.0.0")
