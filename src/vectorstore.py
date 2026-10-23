"""Klien Qdrant Cloud dan nama koleksi."""
from functools import lru_cache

from qdrant_client import QdrantClient

from src.config import get_setting, require_setting


def collection_name() -> str:
    return get_setting("QDRANT_COLLECTION") or "resumes"


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    # Timeout longgar: upsert batch ke cluster gratis kadang lebih dari 30 detik
    return QdrantClient(url=require_setting("QDRANT_URL"), api_key=require_setting("QDRANT_API_KEY"), timeout=90)
