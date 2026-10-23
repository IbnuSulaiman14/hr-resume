"""Memecah CV menjadi chunk per seksi. Tahap B1.4."""
from dataclasses import dataclass

from src.config import CHUNK_OVERLAP_TOKENS, CHUNK_TOKENS
from src.ingest.parse_html import ParsedResume


@dataclass
class Chunk:
    resume_id: str
    category: str
    section: str
    chunk_idx: int
    text: str  # sudah diberi awalan "[CATEGORY] [section]"


def build_chunks(
    parsed: ParsedResume,
    max_tokens: int = CHUNK_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
) -> list[Chunk]:
    """Satu chunk per seksi; seksi yang lebih panjang dari `max_tokens` dipecah dengan
    RecursiveCharacterTextSplitter.from_tiktoken_encoder (ukuran dan overlap dalam token,
    bukan karakter). chunk_idx berurutan sepanjang CV."""
    raise NotImplementedError("B1.4")
