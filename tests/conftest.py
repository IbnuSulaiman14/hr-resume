"""Pengaman bersama: tes tidak boleh memanggil OpenRouter, Qdrant, atau Langfuse sungguhan."""
import httpx
import pytest


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv("RERANK_MODEL", "off")

    def blocked(*args, **kwargs):
        raise RuntimeError("Tes mencoba memanggil jaringan; pakai tiruan (monkeypatch).")

    monkeypatch.setattr(httpx, "post", blocked)
    monkeypatch.setattr(httpx, "get", blocked)
