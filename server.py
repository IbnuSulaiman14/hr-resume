"""API Asisten Rekruter (FastAPI), siap dijalankan di Docker dan Google Cloud Run.

Endpoint:
- GET  /             cek server hidup
- POST /chat         jawaban utuh setelah agent selesai
- POST /chat/stream  Server-Sent Events: tool_start, tool_end, token, lalu done (isinya sama dengan /chat) atau error

Jalankan lokal: uvicorn server:app --reload --port 8080, lalu buka http://localhost:8080/docs
Bila environment API_KEY diisi, setiap request /chat wajib membawa header X-API-Key bernilai sama.
"""
import asyncio
import json
import logging
import os
import secrets
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from src.chat import ChatResult, ask, ask_stream
from src.observability import flush
from src.prompts import PromptUnavailableError

logger = logging.getLogger("uvicorn.error")

GENERIC_ERROR = "Agent gagal menjawab. Detailnya ada di log server."
ROLE_PREFIXES = {"user": "user", "human": "user", "assistant": "assistant", "ai": "assistant"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # Cloud Run mengirim SIGTERM sebelum mematikan instance: kirim sisa trace Langfuse
    flush()


app = FastAPI(
    title="Asisten Rekruter API",
    description="Chatbot RAG multi-agent untuk mencari, mencocokkan, dan menganalisis kandidat dari 2.484 CV.",
    version="1.0.0",
    lifespan=lifespan,
)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def check_api_key(key: str | None = Security(api_key_header)) -> None:
    expected = os.environ.get("API_KEY")
    if expected and not (key and secrets.compare_digest(key.encode(), expected.encode())):
        raise HTTPException(status_code=401, detail="Header X-API-Key tidak ada atau salah.")


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=20000)


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000, examples=["Cari 3 kandidat HR manager yang berpengalaman rekrutmen"])
    history: list[Message] | str = Field(
        default_factory=list,
        description="Giliran sebelumnya: list {role, content}, atau teks dengan baris 'user: ...' dan 'assistant: ...'",
    )
    session_id: str | None = Field(default=None, max_length=200, description="Mengelompokkan trace di Langfuse")


class ToolCall(BaseModel):
    name: str
    args: dict
    agent: str | None = None


class ChatResponse(BaseModel):
    response: str
    input_tokens: int
    output_tokens: int
    tool_messages: list[str] = Field(description="Keluaran tool supervisor (jawaban Matcher/Analyst)")
    tool_calls: list[ToolCall] = Field(description="Semua tool yang dipanggil, termasuk di dalam sub-agent")
    sources: list[dict] = Field(description="Sumber RAG: resume_id, category, section, score, snippet")
    charts: list[dict]
    llm_calls: int
    cost_usd: float | None
    seconds: float
    trace_id: str | None


def history_messages(history: list[Message] | str) -> list[dict]:
    """Riwayat sebagai list {role, content}. Teks per baris 'user: ...' juga diterima, seperti repo referensi."""
    if isinstance(history, list):
        return [m.model_dump() for m in history]
    messages: list[dict] = []
    for line in history.splitlines():
        prefix, sep, content = line.partition(":")
        role = ROLE_PREFIXES.get(prefix.strip().lower()) if sep else None
        if role:
            messages.append({"role": role, "content": content.strip()})
        elif messages and line.strip():
            messages[-1]["content"] += "\n" + line.strip()
    return messages


def to_response(result: ChatResult) -> ChatResponse:
    usage = result.usage
    return ChatResponse(
        response=result.answer,
        input_tokens=usage.get("input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
        tool_messages=result.tool_messages,
        tool_calls=result.tool_calls,
        sources=result.sources,
        charts=result.charts,
        llm_calls=usage.get("llm_calls", 0),
        cost_usd=usage.get("cost_usd"),
        seconds=result.seconds,
        trace_id=result.trace_id,
    )


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


@app.get("/")
def root() -> dict:
    return {"status": "ok", "message": "Asisten Rekruter API berjalan. Dokumentasi di /docs"}


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(check_api_key)])
def chat(request: ChatRequest) -> ChatResponse:
    """Jawaban utuh setelah agent selesai, beserta tool call, sumber, dan token."""
    try:
        result = ask(request.query, history_messages(request.history), session_id=request.session_id)
    except PromptUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("POST /chat gagal")
        raise HTTPException(status_code=500, detail=GENERIC_ERROR) from exc
    finally:
        # Cloud Run memperlambat CPU setelah respons terkirim, jadi trace Langfuse dikirim sebelum menjawab
        flush()
    return to_response(result)


@app.post(
    "/chat/stream",
    dependencies=[Depends(check_api_key)],
    response_class=StreamingResponse,
    responses={200: {"content": {"text/event-stream": {}}, "description": "Event: tool_start, tool_end, token, done, error"}},
)
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """Server-Sent Events selagi agent berjalan. Event done berisi data yang sama dengan POST /chat."""
    history = history_messages(request.history)

    async def events():
        # Generator ini harus async: penampung per permintaan (contextvars) hilang pada generator sync
        try:
            async for event in ask_stream(request.query, history, session_id=request.session_id):
                kind = event.pop("type")
                if kind == "done":
                    await asyncio.to_thread(flush)
                    yield sse("done", to_response(event["result"]).model_dump(mode="json"))
                else:
                    yield sse(kind, event)
        except PromptUnavailableError as exc:
            yield sse("error", {"detail": str(exc)})
        except Exception:
            logger.exception("POST /chat/stream gagal")
            await asyncio.to_thread(flush)
            yield sse("error", {"detail": GENERIC_ERROR})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
