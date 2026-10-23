"""Pengambilan prompt: Langfuse (label production) dengan cadangan lokal di prompts/*.md. Tahap B3.1.

Unggah atau perbarui prompt ke Langfuse: python -m src.prompts --upload
"""
import argparse
from dataclasses import dataclass
from typing import Any

from src.config import PROMPTS_DIR
from src.observability import get_langfuse

PROMPT_NAMES = ["supervisor-system", "matcher-system", "analyst-system"]
LABEL = "production"


@dataclass
class Prompt:
    name: str
    text: str
    source: str  # "langfuse v3" atau "lokal"
    client: Any = None  # objek prompt Langfuse, untuk menautkan trace ke versi prompt


def load_local_prompt(name: str) -> str:
    return (PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8").strip()


def get_prompt(name: str) -> Prompt:
    """Ambil prompt dari Langfuse; bila kunci kosong atau Langfuse tidak terjangkau, pakai salinan lokal."""
    local = load_local_prompt(name)
    client = get_langfuse()
    if client is None:
        return Prompt(name, local, "lokal")
    try:
        # fallback: SDK mengembalikan teks lokal bila fetch gagal, cache 5 menit supaya tidak fetch tiap pesan
        prompt = client.get_prompt(name, label=LABEL, fallback=local, cache_ttl_seconds=300, fetch_timeout_seconds=5)
    except Exception:
        return Prompt(name, local, "lokal")
    if getattr(prompt, "is_fallback", False):
        return Prompt(name, prompt.prompt, "lokal")
    return Prompt(name, prompt.prompt, f"langfuse v{prompt.version}", prompt)


def upload_prompts() -> None:
    """Buat versi baru di Langfuse hanya bila teks lokal berbeda dari versi production saat ini."""
    client = get_langfuse()
    if client is None:
        raise RuntimeError("Kunci Langfuse belum diisi.")
    for name in PROMPT_NAMES:
        local = load_local_prompt(name)
        try:
            current = client.get_prompt(name, label=LABEL, cache_ttl_seconds=0).prompt
        except Exception:
            current = None
        if current == local:
            print(f"{name}: sama dengan production, dilewati")
            continue
        created = client.create_prompt(name=name, prompt=local, labels=[LABEL], type="text")
        print(f"{name}: versi {created.version} dibuat dengan label {LABEL}")
    client.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--upload", action="store_true", help="unggah prompts/*.md ke Langfuse")
    args = parser.parse_args()
    if args.upload:
        upload_prompts()
    else:
        for prompt_name in PROMPT_NAMES:
            p = get_prompt(prompt_name)
            print(f"{prompt_name}: {p.source}, {len(p.text)} karakter")
