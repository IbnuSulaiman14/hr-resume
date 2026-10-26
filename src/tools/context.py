"""Penampung per permintaan untuk sumber RAG, log tool call, dan data grafik.

Sub-agent dipanggil sebagai tool di dalam supervisor, jadi artefak tool tidak naik sendiri ke UI.
Tool menulis ke sini; UI membacanya setelah agent selesai. Memakai contextvars supaya aman
terhadap permintaan bersamaan.
"""
from contextvars import ContextVar
from typing import Any

_sources: ContextVar[list[dict[str, Any]] | None] = ContextVar("sources", default=None)
_tool_calls: ContextVar[list[dict[str, Any]] | None] = ContextVar("tool_calls", default=None)
_usage: ContextVar[list[dict[str, Any]] | None] = ContextVar("usage", default=None)
_charts: ContextVar[list[dict[str, Any]] | None] = ContextVar("charts", default=None)
_rerank: ContextVar[bool] = ContextVar("rerank", default=True)


def start_run(rerank: bool = True) -> None:
    """Panggil di awal setiap permintaan chat. `rerank=False` mematikan reranker untuk permintaan ini (dipakai eval perbandingan)."""
    _sources.set([])
    _tool_calls.set([])
    _usage.set([])
    _charts.set([])
    _rerank.set(rerank)


def rerank_enabled() -> bool:
    return _rerank.get()


def add_usage(kind: str, tokens: int = 0, cost_usd: float = 0.0) -> None:
    """Catat pemakaian di luar LLM chat (mis. reranker) supaya ikut terhitung di token dan biaya per jawaban."""
    current = _usage.get()
    if current is not None:
        current.append({"kind": kind, "tokens": tokens, "cost_usd": cost_usd})


def get_usage() -> list[dict[str, Any]]:
    return list(_usage.get() or [])


def add_sources(items: list[dict[str, Any]]) -> None:
    """Catat sumber (resume_id, category, section, score, snippet) dari keluaran tool."""
    current = _sources.get()
    if current is not None:
        current.extend(items)


def add_tool_call(name: str, args: dict[str, Any], agent: str | None = None) -> None:
    """Catat satu pemanggilan tool. `agent` adalah agent pemanggilnya (supervisor, matcher, analyst)."""
    current = _tool_calls.get()
    if current is not None:
        current.append({"name": name, "args": args, "agent": agent})


def add_chart(spec: dict[str, Any]) -> None:
    """Catat spesifikasi grafik (judul dan data) dari keluaran tool; UI yang menggambarnya."""
    current = _charts.get()
    if current is not None:
        current.append(spec)


def get_charts() -> list[dict[str, Any]]:
    return list(_charts.get() or [])


def get_sources() -> list[dict[str, Any]]:
    return list(_sources.get() or [])


def get_tool_calls() -> list[dict[str, Any]]:
    return list(_tool_calls.get() or [])
