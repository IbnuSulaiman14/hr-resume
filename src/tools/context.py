"""Penampung per permintaan untuk sumber RAG dan log tool call.

Sub-agent dipanggil sebagai tool di dalam supervisor, jadi artefak tool tidak naik sendiri ke UI.
Tool menulis ke sini; UI membacanya setelah agent selesai. Memakai contextvars supaya aman
terhadap permintaan bersamaan.
"""
from contextvars import ContextVar
from typing import Any

_sources: ContextVar[list[dict[str, Any]] | None] = ContextVar("sources", default=None)
_tool_calls: ContextVar[list[dict[str, Any]] | None] = ContextVar("tool_calls", default=None)


def start_run() -> None:
    """Panggil di awal setiap permintaan chat."""
    _sources.set([])
    _tool_calls.set([])


def add_sources(items: list[dict[str, Any]]) -> None:
    """Catat sumber (resume_id, category, section, score, snippet) dari keluaran tool."""
    current = _sources.get()
    if current is not None:
        current.extend(items)


def add_tool_call(name: str, args: dict[str, Any]) -> None:
    current = _tool_calls.get()
    if current is not None:
        current.append({"name": name, "args": args})


def get_sources() -> list[dict[str, Any]]:
    return list(_sources.get() or [])


def get_tool_calls() -> list[dict[str, Any]]:
    return list(_tool_calls.get() or [])
