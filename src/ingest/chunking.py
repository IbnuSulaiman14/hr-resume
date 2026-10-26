"""Memecah CV menjadi chunk per seksi."""
from dataclasses import dataclass
from functools import lru_cache

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_OVERLAP_TOKENS, CHUNK_TOKENS
from src.ingest.parse_html import ParsedResume

# Tokenizer milik text-embedding-3-small, jadi hitungan token sama dengan yang dilihat model embedding
ENCODING = "cl100k_base"


@dataclass
class Chunk:
    resume_id: str
    category: str
    section: str
    chunk_idx: int
    text: str  # sudah diberi awalan "[CATEGORY] [section]"


@lru_cache(maxsize=4)
def _splitter(max_tokens: int, overlap_tokens: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=ENCODING, chunk_size=max_tokens, chunk_overlap=overlap_tokens
    )


@lru_cache(maxsize=1)
def _encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding(ENCODING)


def count_tokens(text: str) -> int:
    return len(_encoding().encode(text))


def chunk_prefix(category: str, section: str) -> str:
    # Awalan ikut di-embed supaya vektor membawa konteks kategori dan seksi
    return f"[{category}] [{section}]\n"


def build_chunks(
    parsed: ParsedResume,
    max_tokens: int = CHUNK_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
) -> list[Chunk]:
    """Satu chunk per seksi; seksi yang lebih panjang dari `max_tokens` dipecah dengan
    RecursiveCharacterTextSplitter.from_tiktoken_encoder (ukuran dan overlap dalam token,
    bukan karakter). chunk_idx berurutan sepanjang CV."""
    chunks: list[Chunk] = []
    for section, body in parsed.sections.items():
        body = body.strip()
        if not body:
            continue
        pieces = [body] if count_tokens(body) <= max_tokens else _splitter(max_tokens, overlap_tokens).split_text(body)
        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            chunks.append(
                Chunk(
                    resume_id=parsed.resume_id,
                    category=parsed.category,
                    section=section,
                    chunk_idx=len(chunks),
                    text=chunk_prefix(parsed.category, section) + piece,
                )
            )
    return chunks
