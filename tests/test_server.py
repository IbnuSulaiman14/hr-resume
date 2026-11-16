"""API FastAPI (server.py) dan streaming ask_stream, offline dengan model palsu."""
import asyncio
import json

import pytest
from fastapi.testclient import TestClient
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk
from langchain_core.tools import tool

import server
import src.agents.supervisor as supervisor
import src.chat as chat
import src.prompts as prompts
from src.chat import ChatResult
from src.prompts import Prompt, PromptUnavailableError
from src.tools import context

ANSWER = "Kandidat terbaik adalah CV 123"
MATCHER_ANSWER = "Kandidat 123 cocok sebagai HR Manager"


class ScriptedModel(GenericFakeChatModel):
    """Model palsu yang bisa memanggil tool dan men-stream jawaban per kata."""

    def bind_tools(self, tools, **kwargs):
        return self

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        message = next(self.messages)
        if message.tool_calls:
            chunks = [AIMessageChunk(content="", tool_call_chunks=[
                {"name": c["name"], "args": json.dumps(c["args"]), "id": c["id"], "index": i}
                for i, c in enumerate(message.tool_calls)
            ])]
        else:
            words = message.content.split(" ")
            chunks = [AIMessageChunk(content=w if i == len(words) - 1 else w + " ") for i, w in enumerate(words)]
        if message.usage_metadata:
            chunks[-1].usage_metadata = message.usage_metadata
        for chunk in chunks:
            generation = ChatGenerationChunk(message=chunk)
            if run_manager:
                run_manager.on_llm_new_token(chunk.content, chunk=generation)
            yield generation


def fake_prompt(name):
    return Prompt(name=name, text=f"system {name}", version=1, client=None)


@pytest.fixture
def scripted_supervisor(monkeypatch):
    """Supervisor asli dari build_supervisor dengan model palsu: supervisor -> matcher -> search_resumes."""
    for module in (chat, supervisor, prompts):
        monkeypatch.setattr(module, "get_prompt", fake_prompt)
    monkeypatch.setattr(chat, "get_langfuse_handler", lambda: None)
    monkeypatch.setenv("CHAT_MODEL", "fake-model")

    @tool("search_resumes")
    def fake_search(query: str) -> str:
        """Cari CV."""
        context.add_tool_call("search_resumes", {"query": query}, agent="matcher")
        context.add_sources([{"resume_id": "123", "category": "HR", "section": "experience", "score": 0.9,
                              "snippet": "HR Manager"}])
        return '{"results": [{"resume_id": "123"}]}'

    matcher_model = ScriptedModel(messages=iter([
        AIMessage("", tool_calls=[{"name": "search_resumes", "args": {"query": "hr manager"}, "id": "s1"}]),
        AIMessage(MATCHER_ANSWER),
    ]))
    supervisor_model = ScriptedModel(messages=iter([
        AIMessage("", tool_calls=[{"name": "matcher", "args": {"task": "cari HR manager"}, "id": "m1"}]),
        AIMessage(ANSWER, usage_metadata={"input_tokens": 50, "output_tokens": 7, "total_tokens": 57}),
    ]))
    monkeypatch.setattr(supervisor, "get_chat_model", lambda: supervisor_model)
    monkeypatch.setattr(supervisor, "build_matcher", lambda: create_agent(
        model=matcher_model, tools=[fake_search], middleware=[prompts.langfuse_system_prompt("matcher-system")],
        name="matcher",
    ))
    monkeypatch.setattr(supervisor, "build_analyst", lambda: create_agent(
        model=ScriptedModel(messages=iter([])), tools=[], name="analyst",
    ))
    graph = supervisor.build_supervisor()
    monkeypatch.setattr(chat, "get_supervisor", lambda: graph)
    return graph


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(server, "flush", lambda: None)
    monkeypatch.delenv("API_KEY", raising=False)
    with TestClient(server.app) as test_client:
        yield test_client


def collect(events):
    async def run():
        return [event async for event in events]

    return asyncio.run(run())


def parse_sse(text):
    events = []
    for block in text.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((fields["event"], json.loads(fields["data"])))
    return events


RESULT = ChatResult(
    answer="CV 123 cocok.",
    sources=[{"resume_id": "123", "category": "HR", "section": "skills", "score": 0.8, "snippet": "x"}],
    tool_calls=[{"name": "matcher", "args": {"task": "cari"}, "agent": "supervisor"}],
    usage={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120, "llm_calls": 3, "cost_usd": 0.00001},
    seconds=1.5,
    trace_id="trace-1",
    tool_messages=["CV 123 cocok sebagai HR"],
)


def test_ask_stream_reports_sub_agent_tools_and_answer_tokens(scripted_supervisor):
    events = collect(chat.ask_stream("cari HR manager"))

    tools = [(e["type"], e["agent"], e["tool"]) for e in events if e["type"] in ("tool_start", "tool_end")]
    assert tools == [
        ("tool_start", "supervisor", "matcher"),
        ("tool_start", "matcher", "search_resumes"),
        ("tool_end", "matcher", "search_resumes"),
        ("tool_end", "supervisor", "matcher"),
    ]
    assert events[0]["input"] == {"task": "cari HR manager"}
    # Hanya teks supervisor yang di-stream; jawaban Matcher untuk supervisor tidak ikut
    assert "".join(e["text"] for e in events if e["type"] == "token") == ANSWER
    assert events[-1]["type"] == "done"
    result = events[-1]["result"]
    assert result.answer == ANSWER
    assert result.tool_messages == [MATCHER_ANSWER]
    assert [c["name"] for c in result.tool_calls] == ["matcher", "search_resumes"]
    assert result.sources[0]["resume_id"] == "123"
    assert result.usage["input_tokens"] == 50


def test_ask_gives_same_result_without_streaming(scripted_supervisor):
    result = chat.ask("cari HR manager")

    assert result.answer == ANSWER
    assert result.tool_messages == [MATCHER_ANSWER]
    assert [c["name"] for c in result.tool_calls] == ["matcher", "search_resumes"]
    assert result.sources[0]["resume_id"] == "123"
    assert result.usage["input_tokens"] == 50


def test_stream_endpoint_keeps_sources_and_tool_calls(scripted_supervisor, client):
    response = client.post("/chat/stream", json={"query": "cari HR manager"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(response.text)
    assert events[0][0] == "tool_start" and events[-1][0] == "done"
    done = events[-1][1]
    assert done["response"] == ANSWER
    assert done["tool_messages"] == [MATCHER_ANSWER]
    assert [c["name"] for c in done["tool_calls"]] == ["matcher", "search_resumes"]
    assert done["sources"][0]["resume_id"] == "123"
    assert done["input_tokens"] == 50


def test_root_is_health_check(client):
    assert client.get("/").json()["status"] == "ok"


def test_chat_returns_reference_fields_and_passes_history(client, monkeypatch):
    calls = []

    def fake_ask(question, history=None, session_id=None, **kwargs):
        calls.append((question, history, session_id))
        return RESULT

    monkeypatch.setattr(server, "ask", fake_ask)
    history = [{"role": "user", "content": "halo"}, {"role": "assistant", "content": "hai"}]
    response = client.post("/chat", json={"query": "cari", "history": history, "session_id": "s1"})

    assert response.status_code == 200
    body = response.json()
    assert body["response"] == "CV 123 cocok."
    assert (body["input_tokens"], body["output_tokens"], body["llm_calls"]) == (100, 20, 3)
    assert body["tool_messages"] == ["CV 123 cocok sebagai HR"]
    assert body["tool_calls"][0] == {"name": "matcher", "args": {"task": "cari"}, "agent": "supervisor"}
    assert body["trace_id"] == "trace-1"
    assert calls == [("cari", history, "s1")]


def test_history_text_format_is_parsed():
    text = "user: cari HR\nassistant: CV 123\nbaris lanjutan\nuser: yang kedua?"

    assert server.history_messages(text) == [
        {"role": "user", "content": "cari HR"},
        {"role": "assistant", "content": "CV 123\nbaris lanjutan"},
        {"role": "user", "content": "yang kedua?"},
    ]
    assert server.history_messages("") == []


def test_empty_query_is_rejected(client):
    assert client.post("/chat", json={"query": ""}).status_code == 422


def test_api_key_required_only_when_set(client, monkeypatch):
    monkeypatch.setattr(server, "ask", lambda *args, **kwargs: RESULT)
    assert client.post("/chat", json={"query": "cari"}).status_code == 200

    monkeypatch.setenv("API_KEY", "rahasia")
    assert client.post("/chat", json={"query": "cari"}).status_code == 401
    assert client.post("/chat", json={"query": "cari"}, headers={"X-API-Key": "salah"}).status_code == 401
    assert client.post("/chat", json={"query": "cari"}, headers={"X-API-Key": "rahasia"}).status_code == 200
    assert client.post("/chat/stream", json={"query": "cari"}).status_code == 401
    assert client.get("/").status_code == 200


def test_chat_error_hides_internal_detail(client, monkeypatch):
    def broken(*args, **kwargs):
        raise ValueError("detail-internal-rahasia")

    monkeypatch.setattr(server, "ask", broken)
    response = client.post("/chat", json={"query": "cari"})

    assert response.status_code == 500
    assert "detail-internal-rahasia" not in response.text


def test_chat_without_prompt_is_503(client, monkeypatch):
    def no_prompt(*args, **kwargs):
        raise PromptUnavailableError("Kunci Langfuse belum diisi.")

    monkeypatch.setattr(server, "ask", no_prompt)
    assert client.post("/chat", json={"query": "cari"}).status_code == 503


def test_stream_error_becomes_error_event(client, monkeypatch):
    async def broken(*args, **kwargs):
        yield {"type": "token", "text": "Sebagian"}
        raise ValueError("detail-internal-rahasia")

    monkeypatch.setattr(server, "ask_stream", broken)
    events = parse_sse(client.post("/chat/stream", json={"query": "cari"}).text)

    assert [name for name, _ in events] == ["token", "error"]
    assert "detail-internal-rahasia" not in json.dumps(events)
