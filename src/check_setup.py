"""Uji koneksi (B0.4): python -m src.check_setup [--model ID_MODEL]

Memeriksa embedding OpenRouter (dan dimensinya), Qdrant, Langfuse, serta satu panggilan chat
dengan tool calling. Tidak pernah mencetak nilai kunci.
"""
import argparse
import time

from langchain_core.tools import tool

from src.config import get_setting


def check_embedding() -> None:
    from src.models import get_embeddings

    vector = get_embeddings().embed_query("Experienced line cook with catering background")
    print(f"OK embedding: dimensi {len(vector)}")


def check_qdrant() -> None:
    from src.vectorstore import get_qdrant_client

    names = [c.name for c in get_qdrant_client().get_collections().collections]
    print(f"OK qdrant: {len(names)} koleksi {names}")


def check_langfuse() -> None:
    from src.observability import get_langfuse

    client = get_langfuse()
    print(f"OK langfuse: auth_check={client.auth_check() if client else 'kunci tidak diisi'}")


@tool
def count_resumes(category: str) -> int:
    """Hitung jumlah CV pada satu kategori, misalnya CHEF."""
    return 42


def check_chat(model: str | None) -> None:
    from src.models import get_chat_model

    llm = get_chat_model(model).bind_tools([count_resumes])
    start = time.perf_counter()
    msg = llm.invoke("Berapa jumlah CV kategori CHEF? Gunakan tool.")
    elapsed = time.perf_counter() - start
    calls = [(c["name"], c["args"]) for c in msg.tool_calls]
    print(f"OK chat {model or get_setting('CHAT_MODEL')}: tool_calls={calls} usage={msg.usage_metadata} waktu={elapsed:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="ID model chat untuk diuji (bawaan: CHAT_MODEL)")
    parser.add_argument("--only", choices=["embedding", "qdrant", "langfuse", "chat"])
    args = parser.parse_args()

    checks = {
        "embedding": check_embedding,
        "qdrant": check_qdrant,
        "langfuse": check_langfuse,
        "chat": lambda: check_chat(args.model),
    }
    for name, fn in checks.items():
        if args.only and name != args.only:
            continue
        try:
            fn()
        except Exception as exc:  # tampilkan jenis galat tanpa membocorkan kunci
            print(f"GAGAL {name}: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
