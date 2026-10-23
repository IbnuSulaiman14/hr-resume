"""Satu putaran chat: riwayat + pertanyaan -> supervisor -> jawaban, sumber, tool call, token.

Dipakai bersama oleh Streamlit (app.py), CLI (src/cli.py), dan evaluasi (eval/run_eval.py).
"""
import time
from dataclasses import dataclass, field
from functools import lru_cache

from langchain_core.messages import HumanMessage

from src.agents.supervisor import build_supervisor, prompt_metadata, trim_history
from src.config import HISTORY_TURNS
from src.models import chat_model_name
from src.observability import get_langfuse_handler
from src.tools import context
from src.usage import UsageTracker

# Batas langkah graf supervisor (tiap panggilan model atau tool dihitung satu langkah)
SUPERVISOR_RECURSION_LIMIT = 20


@dataclass
class ChatResult:
    answer: str
    sources: list[dict] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    seconds: float = 0.0
    trace_id: str | None = None


@lru_cache(maxsize=1)
def get_supervisor():
    return build_supervisor()


def dedupe_sources(sources: list[dict]) -> list[dict]:
    """Satu baris per (resume_id, seksi), ambil skor tertinggi, urut sesuai kemunculan pertama."""
    best: dict[tuple, dict] = {}
    for source in sources:
        key = (source.get("resume_id"), source.get("section"))
        current = best.get(key)
        if current is None or (source.get("score") or 0) > (current.get("score") or 0):
            best[key] = source  # mengganti nilai kunci lama tidak mengubah urutannya
    return list(best.values())


def ask(question: str, history: list[dict] | None = None, session_id: str | None = None, agent=None) -> ChatResult:
    """Jalankan satu pertanyaan. `history` berisi dict {"role", "content"} dari giliran sebelumnya.
    `agent` adalah pasangan (graf, prompt) dari build_supervisor; bawaannya disimpan sekali per proses."""
    context.start_run()
    graph, prompt = agent or get_supervisor()
    tracker = UsageTracker()
    handler = get_langfuse_handler()
    callbacks = [tracker] + ([handler] if handler else [])
    # Metadata diwarisi semua run turunan: Langfuse memakai session_id, tag, dan prompt untuk mengelompokkan trace
    metadata = {"langfuse_tags": ["asisten-rekruter"], **prompt_metadata(prompt)}
    if session_id:
        metadata["langfuse_session_id"] = session_id

    messages = trim_history(history or [], HISTORY_TURNS) + [HumanMessage(question)]
    started = time.perf_counter()
    result = graph.invoke(
        {"messages": messages},
        config={
            "callbacks": callbacks,
            "metadata": metadata,
            "run_name": "asisten-rekruter",
            "recursion_limit": SUPERVISOR_RECURSION_LIMIT,
        },
    )
    answer = result["messages"][-1].content
    return ChatResult(
        answer=answer if isinstance(answer, str) else str(answer),
        sources=dedupe_sources(context.get_sources()),
        tool_calls=context.get_tool_calls(),
        usage=tracker.as_dict(chat_model_name()),
        seconds=round(time.perf_counter() - started, 1),
        trace_id=getattr(handler, "last_trace_id", None) if handler else None,
    )
