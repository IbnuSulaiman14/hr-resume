"""Prompt agent diambil dari Langfuse (label production). Tidak ada salinan lokal: prompt wajib dari Langfuse.

Lihat prompt yang sedang dipakai: python -m src.prompts [--show NAMA]
Mengubah prompt: edit di dashboard Langfuse (Prompts), simpan sebagai versi baru dengan label production.
"""
import argparse
import sys
from dataclasses import dataclass
from typing import Any

from langchain.agents.middleware import ModelRequest, dynamic_prompt

from src.observability import get_langfuse

PROMPT_NAMES = ["supervisor-system", "matcher-system", "analyst-system"]
LABEL = "production"
# SDK Langfuse menyimpan prompt di memori selama ini; setelah lewat, versi lama tetap dipakai
# sambil SDK mengambil versi baru di latar belakang, jadi pesan pengguna tidak menunggu jaringan.
CACHE_TTL_SECONDS = 300


class PromptUnavailableError(RuntimeError):
    """Prompt tidak bisa diambil dari Langfuse (kunci kosong, jaringan, atau prompt belum dibuat)."""


@dataclass
class Prompt:
    name: str
    text: str
    version: int
    client: Any  # objek prompt Langfuse, untuk menautkan trace ke versi prompt


def get_prompt(name: str) -> Prompt:
    """Ambil prompt `name` berlabel production dari Langfuse."""
    client = get_langfuse()
    if client is None:
        raise PromptUnavailableError("Kunci Langfuse belum diisi. Prompt agent wajib diambil dari Langfuse.")
    try:
        prompt = client.get_prompt(name, label=LABEL, cache_ttl_seconds=CACHE_TTL_SECONDS, fetch_timeout_seconds=10)
    except Exception as exc:
        raise PromptUnavailableError(f"Prompt '{name}' tidak bisa diambil dari Langfuse: {type(exc).__name__}") from exc
    return Prompt(name=name, text=prompt.prompt, version=prompt.version, client=prompt)


def prompt_metadata(prompt: Prompt) -> dict:
    """Metadata config yang membuat Langfuse menautkan panggilan LLM di dalam graf ke versi prompt ini."""
    return {"langfuse_prompt": prompt.client}


def langfuse_system_prompt(name: str):
    """Middleware create_agent yang mengisi system prompt dari Langfuse setiap kali model dipanggil.

    Dengan cara ini graf agent boleh di-cache (Streamlit), tetapi perubahan prompt di Langfuse tetap
    terpakai paling lambat CACHE_TTL_SECONDS kemudian tanpa restart aplikasi."""

    @dynamic_prompt
    def system_prompt_from_langfuse(request: ModelRequest) -> str:
        return get_prompt(name).text

    return system_prompt_from_langfuse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", choices=PROMPT_NAMES, help="cetak isi satu prompt")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if args.show:
        prompt = get_prompt(args.show)
        print(f"# {prompt.name} (Langfuse v{prompt.version}, label {LABEL})\n")
        print(prompt.text)
        return
    for name in PROMPT_NAMES:
        prompt = get_prompt(name)
        print(f"{name}: Langfuse v{prompt.version}, {len(prompt.text)} karakter")


if __name__ == "__main__":
    main()
