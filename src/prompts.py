"""Pengambilan prompt: Langfuse (label production) dengan cadangan lokal di prompts/*.md."""
from src.config import PROMPTS_DIR


def load_local_prompt(name: str) -> str:
    return (PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8")


def get_prompt(name: str) -> str:
    """Tahap B3.1: ambil dari Langfuse; bila gagal, jatuh ke salinan lokal."""
    # TODO B3.1: coba Langfuse dulu, tangkap galat koneksi, lalu jatuh ke lokal.
    return load_local_prompt(name)
