"""
Batch test script — Arabic Document OCR API v3.0
Sends all CIN front/back images to the API and saves results.

Usage:
    python scripts/test_api.py [--model groq|qwen]
"""

import argparse
import json
import time
import requests
from datetime import datetime
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_DIR = Path(__file__).parent.parent / "test_dataset"
API_BASE    = "http://localhost:5000"
OUTPUT_DIR  = Path(__file__).parent.parent / "test_results"
DELAY       = 1.5   # seconds between requests (avoid rate limit)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def send_image(image_path: Path, endpoint: str) -> dict:
    with open(image_path, "rb") as f:
        files = {"image": (image_path.name, f, "image/jpeg")}
        resp = requests.post(endpoint, files=files, timeout=60)
    return {
        "status_code": resp.status_code,
        "response": resp.json() if "application/json" in resp.headers.get("Content-Type", "") else resp.text,
    }


def collect_images(folder: Path, pattern: str) -> list[Path]:
    return sorted(
        p for p in folder.iterdir()
        if p.is_file()
        and pattern in p.stem.lower()
        and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".jfif", ".bmp", ".tiff"}
    )


def main():
    parser = argparse.ArgumentParser(description="Batch test Arabic OCR API")
    parser.add_argument("--model", default="groq", choices=["groq", "qwen"],
                        help="VLM to use (default: groq)")
    args = parser.parse_args()

    model = args.model
    front_endpoint = f"{API_BASE}/process/{model}/cin-front"
    back_endpoint  = f"{API_BASE}/process/{model}/cin-back"

    print("=" * 62)
    print("  Arabic Document OCR — Batch Test")
    print(f"  Model   : {model}")
    print(f"  Dataset : {DATASET_DIR}")
    print(f"  API     : {API_BASE}")
    print(f"  Results : {OUTPUT_DIR}")
    print("=" * 62)

    try:
        health = requests.get(f"{API_BASE}/health", timeout=5).json()
        print(f"\n  API health: {health.get('status', '?')}  |  {model}: {health.get('models', {}).get(model, '?')}\n")
    except Exception as e:
        print(f"\n  API unreachable: {e}")
        print("  Start the server first: python app.py\n")
        return

    front_images = collect_images(DATASET_DIR, "cin_front")
    back_images  = collect_images(DATASET_DIR, "cin_back")
    print(f"  Found {len(front_images)} CIN FRONT images")
    print(f"  Found {len(back_images)}  CIN BACK  images\n")

    summary = []

    for label, images, endpoint, doc_type in [
        ("CIN FRONT", front_images, front_endpoint, "CIN_FRONT"),
        ("CIN BACK",  back_images,  back_endpoint,  "CIN_BACK"),
    ]:
        print(f"── {label} {'─' * (54 - len(label))}")
        for img_path in images:
            print(f"  → {img_path.name:<40}", end="", flush=True)
            try:
                result = send_image(img_path, endpoint)
                code = result["status_code"]
                ok = code == 200 and "erreur" not in result["response"]

                out_file = OUTPUT_DIR / f"{img_path.stem}_{model}_result.json"
                with open(out_file, "w", encoding="utf-8") as f:
                    json.dump({
                        "image":    img_path.name,
                        "model":    model,
                        "doc_type": doc_type,
                        "http":     code,
                        "result":   result["response"],
                    }, f, ensure_ascii=False, indent=2)

                mark = "OK" if ok else "ERR"
                print(f"[{mark}] HTTP {code}")
                summary.append({"image": img_path.name, "type": doc_type, "ok": ok, "status": code})

            except Exception as e:
                print(f"[ERR] {e}")
                summary.append({"image": img_path.name, "type": doc_type, "ok": False, "error": str(e)})

            time.sleep(DELAY)
        print()

    total   = len(summary)
    success = sum(1 for s in summary if s.get("ok"))
    failed  = total - success

    print("=" * 62)
    print(f"  TOTAL: {total}   OK: {success}   FAILED: {failed}")
    if failed:
        print("\n  Failed:")
        for s in summary:
            if not s.get("ok"):
                print(f"    - {s['image']}  (HTTP {s.get('status', 'ERR')})")
    print("=" * 62)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = OUTPUT_DIR / f"summary_{model}_{ts}.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump({"model": model, "total": total, "success": success,
                   "failed": failed, "details": summary}, f, ensure_ascii=False, indent=2)
    print(f"\n  Summary → {summary_file}\n")


if __name__ == "__main__":
    main()
