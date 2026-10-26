"""Satu putaran chat: riwayat + pertanyaan -> supervisor -> jawaban, sumber, tool call, token.

Dipakai bersama oleh Streamlit (app.py), CLI (src/cli.py), dan evaluasi (eval/run_eval.py).
"""
import time
from dataclasses import dataclass, field
from functools import lru_cache

from langchain_core.messages import HumanMessage

from src.agents.analyst import PROMPT_NAME as ANALYST_PROMPT
from src.agents.matcher import PROMPT_NAME as MATCHER_PROMPT
from src.agents.supervisor import PROMPT_NAME as SUPERVISOR_PROMPT
from src.agents.supervisor import build_supervisor, trim_history
from src.config import HISTORY_TURNS
from src.models import chat_model_name
from src.observability import get_langfuse_handler
from src.prompts import get_prompt, prompt_metadata
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
    charts: list[dict] = field(default_factory=list)
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


def combine_usage(llm_usage: dict, extra: list[dict]) -> dict:
    """Gabungkan token LLM chat dengan pemakaian lain (reranker) yang dicatat tool ke context."""
    rerank = [e for e in extra if e["kind"] == "rerank"]
    llm_cost = llm_usage.get("cost_usd")
    extra_cost = sum(e["cost_usd"] or 0.0 for e in extra)
    return {
        **llm_usage,
        "rerank_calls": len(rerank),
        "rerank_tokens": sum(e["tokens"] or 0 for e in rerank),
        "cost_usd": None if llm_cost is None else llm_cost + extra_cost,
    }


def ask(
    question: str, history: list[dict] | None = None, session_id: str | None = None, agent=None, rerank: bool = True
) -> ChatResult:
    """Jalankan satu pertanyaan. `history` berisi dict {"role", "content"} dari giliran sebelumnya.
    `agent` adalah graf dari build_supervisor; bawaannya dibuat sekali per proses.
    `rerank=False` mematikan reranker untuk pertanyaan ini (UI selalu menyalakannya; eval memakainya untuk perbandingan)."""
    context.start_run(rerank=rerank)
    graph = agent or get_supervisor()
    # Ambil ketiga prompt di awal: bila Langfuse tidak terjangkau, galatnya jelas (PromptUnavailableError)
    # sebelum agent berjalan, dan versi supervisor dipakai untuk menautkan trace.
    prompt = get_prompt(SUPERVISOR_PROMPT)
    for name in (MATCHER_PROMPT, ANALYST_PROMPT):
        get_prompt(name)
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
        usage=combine_usage(tracker.as_dict(chat_model_name()), context.get_usage()),
        charts=context.get_charts(),
        seconds=round(time.perf_counter() - started, 1),
        trace_id=getattr(handler, "last_trace_id", None) if handler else None,
    )
