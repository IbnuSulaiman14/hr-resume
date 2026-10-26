"""Konfigurasi: env/.env untuk lokal, st.secrets untuk Streamlit Cloud, plus konstanta proyek."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_CSV = DATA_DIR / "Resume.csv"
META_CSV = DATA_DIR / "resumes_meta.csv"

# Dimensi vektor tidak ditulis di sini: build_index membacanya dari model embedding (1.536 untuk
# openai/text-embedding-3-small). Mengganti model embedding berarti ingest ulang dengan --recreate.
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Chunking: seksi lebih dari CHUNK_TOKENS dipecah, dihitung dengan tiktoken
CHUNK_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50

HISTORY_TURNS = 6

# Nilai kolom Category di Resume.csv
CATEGORIES = [
    "ACCOUNTANT", "ADVOCATE", "AGRICULTURE", "APPAREL", "ARTS", "AUTOMOBILE",
    "AVIATION", "BANKING", "BPO", "BUSINESS-DEVELOPMENT", "CHEF", "CONSTRUCTION",
    "CONSULTANT", "DESIGNER", "DIGITAL-MEDIA", "ENGINEERING", "FINANCE", "FITNESS",
    "HEALTHCARE", "HR", "INFORMATION-TECHNOLOGY", "PUBLIC-RELATIONS", "SALES", "TEACHER",
]

# Nama seksi setelah normalisasi judul seksi di HTML
SECTIONS = ["summary", "skills", "experience", "education", "certifications", "other"]


def get_setting(name: str, default: str | None = None) -> str | None:
    """Ambil setelan dari environment dulu, lalu st.secrets (Streamlit Cloud)."""
    value = os.environ.get(name)
    if value:
        return value
    try:
        import streamlit as st

        return st.secrets.get(name, default)
    except Exception:
        return default


def require_setting(name: str) -> str:
    value = get_setting(name)
    if not value:
        raise RuntimeError(f"Setelan {name} belum diisi. Lihat .env.example.")
    return value
