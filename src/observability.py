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


def get_langfuse_handler():
    """Callback LangChain yang merekam satu trace per permintaan, atau None bila Langfuse tidak dipakai."""
    client = get_langfuse()
    if client is None:
        return None
    try:
        from langfuse.langchain import CallbackHandler

        return CallbackHandler(public_key=get_setting("LANGFUSE_PUBLIC_KEY"))
    except Exception:
        return None


def flush() -> None:
    client = get_langfuse()
    if client is not None:
        try:
            client.flush()
        except Exception:
            pass
