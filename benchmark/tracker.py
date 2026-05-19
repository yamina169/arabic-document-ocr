"""
Benchmark result logging and reporting.
Supports: local CSV + JSONL logs, W&B (optional), matplotlib charts (optional).

Usage:
    tracker = BenchmarkTracker(output_dir="benchmark_output", use_wandb=True)
    tracker.log_benchmark_run("groq", metrics={...})
    tracker.save_csv(results)
    tracker.generate_report(results)
    tracker.generate_charts(results)   # optional
    tracker.finish()
"""

import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)


class BenchmarkTracker:

    def __init__(
        self,
        output_dir: str = "benchmark_output",
        use_wandb: bool = False,
        project_name: str = "arabic-cin-ocr",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_wandb = use_wandb
        self.project_name = project_name
        self._wandb = None

        if use_wandb:
            try:
                import wandb
                wandb.init(project=project_name, reinit=True)
                self._wandb = wandb
                log.info("[Tracker] W&B initialized — project: %s", project_name)
            except ImportError:
                log.warning("[Tracker] wandb not installed — falling back to CSV/JSONL only")
                self.use_wandb = False
            except Exception as e:
                log.warning("[Tracker] W&B init failed: %s — falling back to CSV/JSONL only", e)
                self.use_wandb = False

    # ── Logging ───────────────────────────────────────────────────────────────

    def log_benchmark_run(self, model_name: str, metrics: dict, params: dict | None = None) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": model_name,
            "metrics": metrics,
            "params": params or {},
        }

        log_file = self.output_dir / "benchmark_log.jsonl"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        if self.use_wandb and self._wandb:
            self._wandb.log({"model": model_name, **metrics, **(params or {})})

    # ── CSV export ────────────────────────────────────────────────────────────

    def save_csv(self, results: dict, filename: str = "benchmark_results.csv") -> Path:
        csv_path = self.output_dir / filename
        rows = []

        for model_name, data in results.items():
            row = {
                "model": model_name,
                "field_accuracy": f"{data.get('field_accuracy', 0):.4f}",
                "doc_accuracy": f"{data.get('doc_accuracy', 0):.4f}",
                "total_time_s": f"{data.get('total_time', 0):.2f}",
                "docs_evaluated": len(data.get("documents", [])),
                "errors": data.get("errors", 0),
                "avg_time_per_doc_s": (
                    f"{data['total_time'] / len(data['documents']):.2f}"
                    if data.get("documents") else "N/A"
                ),
            }
            rows.append(row)

        if rows:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            log.info("[Tracker] CSV saved: %s", csv_path)

        self._save_detailed_csv(results)
        return csv_path

    def _save_detailed_csv(self, results: dict, filename: str = "benchmark_results_detailed.csv") -> None:
        csv_path = self.output_dir / filename
        rows = []
        for model_name, data in results.items():
            for doc in data.get("documents", []):
                row = {
                    "model": model_name,
                    "image": doc.get("image", ""),
                    "doc_type": doc.get("doc_type", ""),
                    "accuracy": f"{doc.get('accuracy', 0):.4f}",
                    "elapsed_s": f"{doc.get('elapsed', 0):.3f}",
                    "error": doc.get("error", ""),
                }
                for field, score in doc.get("field_scores", {}).items():
                    row[f"field_{field}"] = f"{score:.4f}"
                rows.append(row)

        if rows:
            all_keys = list(dict.fromkeys(k for r in rows for k in r))
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)

    # ── Text report ───────────────────────────────────────────────────────────

    def generate_report(self, results: dict, filename: str = "benchmark_report.txt") -> Path:
        report_path = self.output_dir / filename
        lines: list[str] = []
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        lines += [
            "=" * 70,
            "  BENCHMARK REPORT — Arabic Document OCR",
            f"  Generated: {now}",
            "=" * 70,
            "",
            "SUMMARY",
            "-" * 70,
            f"{'Model':<20} {'Field Acc':>10} {'Doc Acc':>10} {'Time(s)':>10} {'Errors':>8}",
            "-" * 62,
        ]

        for model_name, data in results.items():
            lines.append(
                f"{model_name:<20} "
                f"{data.get('field_accuracy', 0):>10.4f} "
                f"{data.get('doc_accuracy', 0):>10.4f} "
                f"{data.get('total_time', 0):>10.2f} "
                f"{data.get('errors', 0):>8}"
            )

        if results:
            best = max(results.items(), key=lambda kv: kv[1].get("field_accuracy", 0))
            lines += ["", f"Best model: {best[0]} (field_accuracy={best[1].get('field_accuracy', 0):.4f})", ""]

        lines += ["", "=" * 70, "DETAIL PER MODEL", "=" * 70]

        for model_name, data in results.items():
            lines += ["", f"--- {model_name} ---"]
            for doc in data.get("documents", []):
                img = doc.get("image", "?")
                acc = doc.get("accuracy", 0.0)
                elapsed = doc.get("elapsed", 0.0)
                err = doc.get("error", "")
                status = f"ERROR: {err}" if err else f"accuracy={acc:.4f}"
                lines.append(f"  {img:<45} {status}  ({elapsed:.2f}s)")

                for field, score in doc.get("field_scores", {}).items():
                    mark = "✓" if score >= 0.85 else "✗"
                    lines.append(f"    {mark} {field}: {score:.4f}")

        content = "\n".join(lines) + "\n"
        report_path.write_text(content, encoding="utf-8")
        log.info("[Tracker] Report saved: %s", report_path)
        return report_path

    # ── Charts (optional) ─────────────────────────────────────────────────────

    def generate_charts(self, results: dict) -> None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            log.warning("[Tracker] matplotlib not installed — skipping charts")
            return

        model_names = list(results.keys())
        field_accs = [results[m].get("field_accuracy", 0) for m in model_names]
        doc_accs = [results[m].get("doc_accuracy", 0) for m in model_names]

        x = range(len(model_names))
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].bar(x, field_accs, color="#4C72B0")
        axes[0].set_title("Field Accuracy by Model")
        axes[0].set_xticks(list(x))
        axes[0].set_xticklabels(model_names)
        axes[0].set_ylim(0, 1)
        axes[0].set_ylabel("Accuracy")
        for i, v in enumerate(field_accs):
            axes[0].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=10)

        axes[1].bar(x, doc_accs, color="#DD8452")
        axes[1].set_title("Document Accuracy by Model")
        axes[1].set_xticks(list(x))
        axes[1].set_xticklabels(model_names)
        axes[1].set_ylim(0, 1)
        axes[1].set_ylabel("Accuracy")
        for i, v in enumerate(doc_accs):
            axes[1].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=10)

        plt.tight_layout()
        chart_path = self.output_dir / "benchmark_chart.png"
        plt.savefig(chart_path, dpi=150, bbox_inches="tight")
        plt.close()
        log.info("[Tracker] Chart saved: %s", chart_path)

        if self.use_wandb and self._wandb:
            self._wandb.log({"benchmark_chart": self._wandb.Image(str(chart_path))})

    def finish(self) -> None:
        if self.use_wandb and self._wandb:
            self._wandb.finish()
