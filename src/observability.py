"""Klien Langfuse dan callback tracing. Aplikasi tetap jalan bila kunci kosong atau Langfuse tidak terjangkau."""
from functools import lru_cache

from src.config import get_setting


@lru_cache(maxsize=1)
def get_langfuse():
    """Klien Langfuse, atau None bila kunci belum diisi."""
    public_key = get_setting("LANGFUSE_PUBLIC_KEY")
    secret_key = get_setting("LANGFUSE_SECRET_KEY")
    if not (public_key and secret_key):
        return None
    from langfuse import Langfuse

    return Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        host=get_setting("LANGFUSE_HOST") or "https://cloud.langfuse.com",
        timeout=10,
    )
