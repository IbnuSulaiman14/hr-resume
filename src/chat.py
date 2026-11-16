"""Satu putaran chat: riwayat + pertanyaan -> supervisor -> jawaban, sumber, tool call, token.

Dipakai bersama oleh Streamlit (app.py), API (server.py), CLI (src/cli.py), dan evaluasi (eval/run_eval.py).
ask() menunggu agent selesai; ask_stream() mengirim event selagi agent berjalan.
"""
import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from langchain_core.messages import HumanMessage, ToolMessage

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
# Panjang keluaran tool di event tool_end; keluaran utuh tool supervisor ada di ChatResult.tool_messages
TOOL_OUTPUT_PREVIEW_CHARS = 500


@dataclass
class ChatResult:
    answer: str
    sources: list[dict] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    charts: list[dict] = field(default_factory=list)
    seconds: float = 0.0
    trace_id: str | None = None
    # Keluaran tool supervisor (jawaban Matcher/Analyst) sesuai urutan pemanggilan
    tool_messages: list[str] = field(default_factory=list)


@dataclass
class _Run:
    """Bahan satu permintaan, sama untuk ask() dan ask_stream()."""
    graph: Any
    inputs: dict
    config: dict
    tracker: UsageTracker
    handler: Any
    started: float


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


def message_text(content: Any) -> str:
    """Isi pesan sebagai teks. Sebagian penyedia mengirim list blok, bukan string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return str(content)


def _prepare(question: str, history: list[dict] | None, session_id: str | None, agent) -> _Run:
    """Siapkan graf, pesan, dan config. Penampung per permintaan (context.start_run) diisi oleh pemanggil."""
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
    config = {
        "callbacks": callbacks,
        "metadata": metadata,
        "run_name": "asisten-rekruter",
        "recursion_limit": SUPERVISOR_RECURSION_LIMIT,
    }
    messages = trim_history(history or [], HISTORY_TURNS) + [HumanMessage(question)]
    return _Run(graph, {"messages": messages}, config, tracker, handler, started=time.perf_counter())


def _finish(run: _Run, messages: list) -> ChatResult:
    answer = messages[-1].content
    return ChatResult(
        answer=answer if isinstance(answer, str) else str(answer),
        sources=dedupe_sources(context.get_sources()),
        tool_calls=context.get_tool_calls(),
        usage=combine_usage(run.tracker.as_dict(chat_model_name()), context.get_usage()),
        charts=context.get_charts(),
        seconds=round(time.perf_counter() - run.started, 1),
        trace_id=getattr(run.handler, "last_trace_id", None) if run.handler else None,
        tool_messages=[message_text(m.content) for m in messages if isinstance(m, ToolMessage)],
    )


def ask(
    question: str, history: list[dict] | None = None, session_id: str | None = None, agent=None, rerank: bool = True
) -> ChatResult:
    """Jalankan satu pertanyaan. `history` berisi dict {"role", "content"} dari giliran sebelumnya.
    `agent` adalah graf dari build_supervisor; bawaannya dibuat sekali per proses.
    `rerank=False` mematikan reranker untuk pertanyaan ini (UI selalu menyalakannya; eval memakainya untuk perbandingan)."""
    context.start_run(rerank=rerank)
    run = _prepare(question, history, session_id, agent)
    result = run.graph.invoke(run.inputs, config=run.config)
    return _finish(run, result["messages"])


def _preview(output: Any) -> str:
    text = message_text(getattr(output, "content", output))
    return text if len(text) <= TOOL_OUTPUT_PREVIEW_CHARS else text[:TOOL_OUTPUT_PREVIEW_CHARS] + " ..."


async def ask_stream(
    question: str, history: list[dict] | None = None, session_id: str | None = None, agent=None, rerank: bool = True
) -> AsyncIterator[dict]:
    """Versi streaming ask(). Menghasilkan dict event selagi agent berjalan:

    - {"type": "tool_start", "agent", "tool", "input"} dan {"type": "tool_end", "agent", "tool", "output"}.
      `agent` adalah pemanggil tool (supervisor, matcher, atau analyst), jadi tool di dalam sub-agent ikut terlihat.
    - {"type": "token", "text"}: potongan teks dari model supervisor.
    - terakhir {"type": "done", "result": ChatResult}; jawaban utuh ada di result.answer.

    Harus dikonsumsi di satu task asyncio. Penampung per permintaan memakai contextvars, yang hilang bila
    tiap langkah generator berjalan di thread berbeda (misalnya generator sync di StreamingResponse)."""
    context.start_run(rerank=rerank)
    # Prompt diambil dari Langfuse lewat jaringan, jadi dijalankan di thread agar event loop tidak tertahan
    run = await asyncio.to_thread(_prepare, question, history, session_id, agent)
    sub_agents: dict[str, str] = {}  # run_id tool milik supervisor -> nama sub-agent yang dijalankannya
    messages = None
    async for event in run.graph.astream_events(run.inputs, config=run.config, version="v2"):
        kind = event["event"]
        parents = event.get("parent_ids") or []
        caller = next((sub_agents[p] for p in parents if p in sub_agents), "supervisor")
        if kind == "on_tool_start":
            if caller == "supervisor":
                sub_agents[event["run_id"]] = event["name"]
            yield {"type": "tool_start", "agent": caller, "tool": event["name"], "input": event["data"].get("input")}
        elif kind == "on_tool_end":
            output = _preview(event["data"].get("output"))
            yield {"type": "tool_end", "agent": caller, "tool": event["name"], "output": output}
        elif kind == "on_chat_model_stream" and caller == "supervisor":
            text = message_text(event["data"]["chunk"].content)
            if text:
                yield {"type": "token", "text": text}
        elif kind == "on_chain_end" and not parents:
            messages = event["data"]["output"]["messages"]
    if not messages:
        raise RuntimeError("Agent selesai tanpa pesan jawaban.")
    yield {"type": "done", "result": _finish(run, messages)}
