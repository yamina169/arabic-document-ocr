# Arabic Document OCR — VLM Benchmark

A Flask API and benchmarking framework for extracting structured data from **Tunisian National Identity Cards (CIN)** using Vision Language Models.

---

## Overview

This project compares multiple VLM providers on their ability to extract structured fields from Arabic CIN documents (front and back sides). Results are visualized in an interactive benchmark dashboard.

**Supported providers:**

| Provider | Model | Pipeline |
|----------|-------|----------|
| Groq | LLaMA 4 Scout 17B | Multimodal VLM |
| OpenRouter | Qwen2.5-VL 72B | Multimodal VLM |
| PaddleOCR | PaddleOCR-VL-1.5 + Qwen3-32B | OCR → LLM |

**Extracted fields:**

| CIN Front | CIN Back |
|-----------|----------|
| `numero_cin` | `nom_mere` |
| `nom` | `profession` |
| `prenom` | `adresse` |
| `date_naissance` | `date_emission` |
| `lieu_naissance` | `numero_registre` |

---

## Project Structure

```
arabic-document-ocr/
├── app.py                    # Flask API entry point
├── pipeline.py               # Orchestration layer
├── config.py                 # Centralized configuration
├── requirements.txt
│
├── providers/
│   ├── base.py               # Abstract base class + shared prompts
│   ├── groq.py               # Groq Vision provider
│   ├── qwen.py               # Qwen2.5-VL via OpenRouter
│   └── paddleocr.py          # PaddleOCR + Groq LLM (two-stage)
│
├── benchmark/
│   ├── cli.py                # Benchmark CLI entry point
│   ├── runner.py             # Evaluation logic (Levenshtein scoring)
│   └── tracker.py            # Results tracking + CSV/report export
│
├── utils/
│   └── barcode.py            # CIN barcode reader (optional)
│
├── templates/
│   └── dashboard.html        # Dashboard shell
├── static/
│   ├── dashboard.css
│   └── dashboard.js
│
├── test_dataset/             # Images + ground truth (gitignored)
│   └── MANIFEST.json
└── benchmark_output/         # Generated results
```

---

## Setup

### 1. Clone and create virtual environment

```bash
git clone <repo-url>
cd arabic-document-ocr
python -m venv venv
```

### 2. Activate virtual environment

```bash
# Windows
.\venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in your API keys:

```env
GROQ_API_KEY=gsk_...
OPENROUTER_API_KEY=sk-or-v1-...
PADDLEOCR_TOKEN=...
PADDLEOCR_SYNC_URL=https://...
```

---

## Running the API

```bash
python app.py
```

The server starts at `http://localhost:5000`.

**Endpoints:**

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/extract` | Extract fields from a CIN image |
| `GET` | `/health` | Health check |
| `GET` | `/dashboard` | Benchmark dashboard UI |
| `GET` | `/api/results` | Benchmark results JSON |

**Extract example:**

```bash
curl -X POST http://localhost:5000/extract \
  -F "image=@cin_front.jpg" \
  -F "model=groq" \
  -F "doc_type=CIN_FRONT"
```

**Response:**

```json
{
  "numero_cin": "00000000",
  "nom": "...",
  "prenom": "...",
  "date_naissance": "1990-01-15",
  "lieu_naissance": "تونس",
  "champs_manquants": []
}
```

---

## Running the Benchmark

### Prepare the dataset

Place images and their ground truth JSON files in `test_dataset/`:

```
test_dataset/
├── MANIFEST.json
├── cin_front_001.jpg
├── cin_front_001_expected.json
├── cin_back_001.jpg
└── cin_back_001_expected.json
```

**MANIFEST.json format:**

```json
{
  "samples": [
    {
      "image": "cin_front_001.jpg",
      "ground_truth": "cin_front_001_expected.json",
      "doc_type": "CIN_FRONT"
    }
  ]
}
```

### Run benchmark

```bash
# Single model
python -m benchmark.cli --models groq

# All models
python -m benchmark.cli --models groq,qwen,paddleocr

# All models in parallel (faster)
python -m benchmark.cli --models groq,qwen,paddleocr --parallel

# Custom threshold
python -m benchmark.cli --models groq --threshold 0.90
```

**CLI options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--models` | `groq` | Comma-separated model list |
| `--dataset` | `test_dataset/` | Dataset directory |
| `--output` | `benchmark_output/` | Output directory |
| `--threshold` | `0.85` | Similarity threshold |
| `--parallel` | off | Run models in parallel |

### View results

After the benchmark completes, open the dashboard:

```
http://localhost:5000/dashboard
```

---

## Metrics

| Metric | Definition |
|--------|-----------|
| **Field Accuracy** | Average Levenshtein similarity across all extracted fields |
| **Doc Accuracy** | % of documents where average field score ≥ threshold |

The **threshold** (default `0.85`) defines the minimum similarity for a field to be considered correct. It tolerates minor OCR noise such as diacritics and spacing differences without penalising accurate extractions.

---

## Adding a New Provider

1. Create `providers/your_provider.py`
2. Subclass `VLMProvider` from `providers/base.py`
3. Implement the `name` property and `extract_structured()` method
4. Register it in `benchmark/cli.py` under `_REGISTRY`

```python
from providers.base import VLMProvider, EXTRACTION_PROMPTS, SYSTEM_PROMPT

class VLMYours(VLMProvider):
    @property
    def name(self) -> str:
        return "yours"

    def extract_structured(self, image_path: str, doc_type: str) -> dict:
        prompt = EXTRACTION_PROMPTS[doc_type]
        # call your API ...
        return self._parse_response(raw_text)
```

---

## Requirements

- Python 3.10+
- API keys for the providers you want to use (see `.env.example`)
- `pyzbar` + `opencv-python` for barcode reading (optional)
