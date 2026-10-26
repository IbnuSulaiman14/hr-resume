"""Reranker: urutkan ulang hasil pencarian vektor dengan model reranker lewat endpoint /rerank OpenRouter.

Pencarian vektor (bi-encoder) cepat tetapi kasar: query dan CV di-embed terpisah. Reranker (cross-encoder)
membaca query dan teks CV bersama-sama, jadi lebih teliti menilai relevansi. Pola: ambil kandidat lebih
banyak dari Qdrant, rerank, ambil teratas. Bila reranker gagal, urutan vektor tetap dipakai.

Setelan RERANK_MODEL: ID model reranker, atau "off" untuk mematikan.
"""
import time

import httpx

from src.config import DEFAULT_OPENROUTER_BASE_URL, get_setting, require_setting
from src.tools import context

# Satu-satunya model reranker OpenRouter yang lolos pengaturan Zero Data Retention akun ini
DEFAULT_RERANK_MODEL = "qwen/qwen3-reranker-8b"
DOCUMENT_CHARS = 2000
# Satu kali coba ulang: kegagalan sesaat penyedia sempat terlihat saat pengujian
ATTEMPTS = 2


def rerank_model() -> str | None:
    """ID model reranker, atau None bila dimatikan lewat setelan atau untuk satu permintaan lewat `ask(rerank=False)` (dipakai eval perbandingan)."""
    if not context.rerank_enabled():
        return None
    value = (get_setting("RERANK_MODEL") or DEFAULT_RERANK_MODEL).strip()
    return None if value.lower() in {"off", "none", "false", "0"} else value


def rerank_scores(query: str, documents: list[str]) -> list[float] | None:
    """Skor relevansi 0 sampai 1 untuk tiap dokumen, urut sesuai `documents`. None bila reranker mati atau gagal."""
    model = rerank_model()
    if not model or not documents:
        return None
    base_url = (get_setting("OPENROUTER_BASE_URL") or DEFAULT_OPENROUTER_BASE_URL).rstrip("/")
    body = None
    for attempt in range(ATTEMPTS):
        try:
            response = httpx.post(
                f"{base_url}/rerank",
                headers={"Authorization": f"Bearer {require_setting('OPENROUTER_API_KEY')}"},
                json={"model": model, "query": query, "documents": [d[:DOCUMENT_CHARS] for d in documents],
                      "top_n": len(documents)},
                timeout=20,
            )
            response.raise_for_status()
            body = response.json()
            break
        except Exception as exc:
            status = getattr(getattr(exc, "response", None), "status_code", "-")
            print(f"Reranker gagal (percobaan {attempt + 1}, {type(exc).__name__}, HTTP {status}).")
            if attempt + 1 < ATTEMPTS:
                time.sleep(1)
    if body is None:
        print("Reranker tidak tersedia; memakai urutan pencarian vektor.")
        return None
    usage = body.get("usage") or {}
    context.add_usage("rerank", tokens=usage.get("total_tokens", 0), cost_usd=usage.get("cost", 0.0))
    scores: list[float] = [0.0] * len(documents)
    for result in body.get("results", []):
        scores[result["index"]] = float(result["relevance_score"])
    return scores


def apply_rerank(items: list[dict], scores: list[float] | None, top_k: int) -> list[dict]:
    """Tempelkan `rerank_score` ke tiap item lalu urutkan dari yang paling relevan. Tanpa skor, urutan asli dipakai."""
    if scores is None:
        return items[:top_k]
    for item, score in zip(items, scores):
        item["rerank_score"] = round(score, 4)
    return sorted(items, key=lambda item: item["rerank_score"], reverse=True)[:top_k]
