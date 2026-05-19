import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "output"       # API results: output/<model>/<doc-type>/<stem>.json
LOG_DIR    = BASE_DIR / "logs"

for _d in [UPLOAD_DIR, LOG_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Groq ──────────────────────────────────────────────────────────────────────
GROQ_API_KEY      = os.getenv("GROQ_API_KEY", "")
GROQ_VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")

# ── OpenRouter (Qwen) ─────────────────────────────────────────────────────────
OPENROUTER_API_KEY    = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL   = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_QWEN_MODEL = os.getenv("OPENROUTER_QWEN_MODEL", "qwen/qwen2.5-vl-72b-instruct")

# ── Image / upload settings ───────────────────────────────────────────────────
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".jfif", ".png", ".bmp", ".tiff"}
MAX_FILE_SIZE      = 20 * 1024 * 1024     # 20 MB
