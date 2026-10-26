"""Pembuat klien model chat dan embedding lewat OpenRouter (API kompatibel OpenAI)."""
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from src.config import DEFAULT_OPENROUTER_BASE_URL, get_setting, require_setting


def _base_url() -> str:
    return get_setting("OPENROUTER_BASE_URL") or DEFAULT_OPENROUTER_BASE_URL


def chat_model_name() -> str:
    return require_setting("CHAT_MODEL")


def get_chat_model(model: str | None = None, temperature: float = 0.0) -> ChatOpenAI:
    # temperature 0 supaya rute dan pemanggilan tool stabil antar percobaan.
    # Reasoning dimatikan lewat parameter OpenRouter: pada pengujian, DeepSeek tanpa reasoning tetap memanggil
    # tool dengan benar, tetapi sekitar 3 kali lebih cepat dan tidak memecah pencarian menjadi dua.
    return ChatOpenAI(
        model=model or chat_model_name(),
        api_key=require_setting("OPENROUTER_API_KEY"),
        base_url=_base_url(),
        temperature=temperature,
        timeout=60,
        max_retries=2,
        stream_usage=True,
        # provider sort throughput: OpenRouter memilih penyedia dengan keluaran token tercepat
        extra_body={"reasoning": {"enabled": False}, "provider": {"sort": "throughput"}},
    )


def get_embeddings() -> OpenAIEmbeddings:
    # check_embedding_ctx_length=False: tanpa ini langchain mengirim token ID (bukan teks),
    # yang hanya diterima endpoint OpenAI asli, bukan OpenRouter.
    return OpenAIEmbeddings(
        model=get_setting("EMBEDDING_MODEL") or "openai/text-embedding-3-small",
        api_key=require_setting("OPENROUTER_API_KEY"),
        base_url=_base_url(),
        check_embedding_ctx_length=False,
    )
