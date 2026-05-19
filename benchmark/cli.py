#!/usr/bin/env python3
"""
Benchmark CLI — test all VLM providers on an Arabic document dataset.

Usage:
    python -m benchmark.cli --models groq,qwen,paddleocr --dataset test_dataset/

"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
load_dotenv()

from benchmark.runner import load_dataset, run_benchmark
from benchmark.tracker import BenchmarkTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/benchmark.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ── Model registry ─────────────────────────────────────────────────────────────

_REGISTRY = {
    "groq":      ("providers.groq",      "VLMGroq"),
    "qwen":      ("providers.qwen",      "VLMQwen"),
    "paddleocr": ("providers.paddleocr", "VLMPaddleOCR"),
}


def _load_model(name: str):
    name = name.strip().lower()
    if name not in _REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {', '.join(_REGISTRY)}")
    module_path, class_name = _REGISTRY[name]
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)()


# ── Save all extracted results to a single JSON ────────────────────────────────

def _save_all_results(results: dict, output_dir: str) -> Path:
    out_path = Path(output_dir) / "all_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        try:
            with open(out_path, encoding="utf-8") as f:
                payload = json.load(f)
            if "models" not in payload:
                payload["models"] = {}
        except Exception:
            payload = {"models": {}}
    else:
        payload = {"models": {}}

    for model_name, data in results.items():
        payload["models"][model_name] = {
            "field_accuracy": data.get("field_accuracy", 0),
            "doc_accuracy":   data.get("doc_accuracy", 0),
            "total_time":     data.get("total_time", 0),
            "errors":         data.get("errors", 0),
            "documents":      data.get("documents", []),
        }

    payload["generated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    log.info("Results saved → %s  (models present: %s)",
             out_path, ", ".join(payload["models"]))
    return out_path, payload["models"]


# ── Parallel benchmark ─────────────────────────────────────────────────────────

def _run_parallel(models, dataset, output_dir, threshold) -> dict:
    results = {}
    with ThreadPoolExecutor(max_workers=len(models)) as pool:
        futures = {
            pool.submit(run_benchmark, [m], dataset, output_dir, threshold): m.name
            for m in models
        }
        for future in as_completed(futures):
            model_name = futures[future]
            try:
                partial = future.result()
                results.update(partial)
                log.info("[Parallel] %s finished", model_name)
            except Exception as e:
                log.error("[Parallel] %s failed: %s", model_name, e)
                results[model_name] = {
                    "model": model_name,
                    "field_accuracy": 0.0,
                    "doc_accuracy": 0.0,
                    "total_time": 0.0,
                    "errors": 1,
                    "documents": [],
                }
    return results


# ── Summary printer ────────────────────────────────────────────────────────────

def _print_summary(results: dict) -> None:
    print()
    print("=" * 70)
    print("  BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"  {'Model':<18} {'Field Acc':>10} {'Doc Acc':>10} {'Time(s)':>10} {'Errors':>8}")
    print("  " + "-" * 56)
    for model_name, data in results.items():
        print(
            f"  {model_name:<18} "
            f"{data.get('field_accuracy', 0):>10.4f} "
            f"{data.get('doc_accuracy', 0):>10.4f} "
            f"{data.get('total_time', 0):>10.2f} "
            f"{data.get('errors', 0):>8}"
        )
    if results:
        best = max(results, key=lambda m: results[m].get("field_accuracy", 0))
        print(f"\n  Best model: {best}  (field_accuracy={results[best].get('field_accuracy', 0):.4f})")
    print("=" * 70)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    Path("logs").mkdir(exist_ok=True)

    parser = argparse.ArgumentParser(
        description="Benchmark Arabic OCR VLM providers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--models", default="groq",
        help="Comma-separated list: groq,qwen,paddleocr  (default: groq)")
    parser.add_argument("--dataset", default="test_dataset/",
        help="Path to test dataset directory  (default: test_dataset/)")
    parser.add_argument("--output", default="benchmark_output/",
        help="Output directory for CSV/report/charts  (default: benchmark_output/)")
    parser.add_argument("--threshold", type=float, default=0.85,
        help="Fuzzy match threshold 0.0–1.0  (default: 0.85)")
    parser.add_argument("--parallel", action="store_true",
        help="Run models in parallel using threads")
    parser.add_argument("--wandb", action="store_true",
        help="Log results to Weights & Biases")
    parser.add_argument("--charts", action="store_true",
        help="Generate matplotlib comparison charts")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        log.error("Dataset path not found: %s", dataset_path)
        sys.exit(1)

    log.info("Loading dataset from %s", dataset_path)
    try:
        dataset = load_dataset(str(dataset_path))
    except Exception as e:
        log.error("Failed to load dataset: %s", e)
        sys.exit(1)

    if not dataset:
        log.error("No samples found. Add images + *_expected.json files, or create MANIFEST.json.")
        sys.exit(1)
    log.info("Dataset: %d samples", len(dataset))

    model_names = [n.strip() for n in args.models.split(",") if n.strip()]
    models = []
    for name in model_names:
        try:
            m = _load_model(name)
            models.append(m)
            log.info("Loaded model: %s", name)
        except Exception as e:
            log.error("Could not load model '%s': %s", name, e)

    if not models:
        log.error("No models loaded. Exiting.")
        sys.exit(1)

    log.info("Starting benchmark — %d model(s), %d sample(s), threshold=%.2f",
             len(models), len(dataset), args.threshold)

    if args.parallel and len(models) > 1:
        log.info("Running models in parallel")
        results = _run_parallel(models, dataset, args.output, args.threshold)
    else:
        results = run_benchmark(models, dataset, args.output, args.threshold)

    all_results_path, all_results = _save_all_results(results, args.output)

    tracker = BenchmarkTracker(output_dir=args.output, use_wandb=args.wandb)

    for model_name, data in results.items():
        tracker.log_benchmark_run(
            model_name=model_name,
            metrics={
                "field_accuracy": data.get("field_accuracy", 0),
                "doc_accuracy":   data.get("doc_accuracy", 0),
                "total_time":     data.get("total_time", 0),
                "errors":         data.get("errors", 0),
            },
            params={"threshold": args.threshold, "dataset": str(dataset_path)},
        )

    csv_path    = tracker.save_csv(all_results)
    report_path = tracker.generate_report(all_results)

    if args.charts:
        tracker.generate_charts(results)

    tracker.finish()

    _print_summary(all_results)
    print(f"\n  Results saved to:")
    print(f"    All extracted docs : {all_results_path}")
    print(f"    CSV                : {csv_path}")
    print(f"    Report             : {report_path}")
    if args.charts:
        print(f"    Chart              : {Path(args.output) / 'benchmark_chart.png'}")
    print()


if __name__ == "__main__":
    main()
