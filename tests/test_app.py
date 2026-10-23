"""Alur dasar Streamlit dengan agent tiruan (tanpa jaringan)."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import src.agents.supervisor
import src.chat
from src.chat import ChatResult

APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture
def fake_agent(monkeypatch):
    calls = []

    def fake_ask(question, history=None, session_id=None, agent=None):
        calls.append({"question": question, "history": list(history or [])})
        return ChatResult(
            answer=f"Jawaban untuk: {question} (CV 123)",
            sources=[{"resume_id": "123", "category": "CHEF", "section": "experience", "score": 0.81, "snippet": "x"}],
            tool_calls=[{"name": "matcher", "args": {"task": question}, "agent": "supervisor"}],
            usage={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120, "llm_calls": 3, "cost_usd": 0.00001},
            seconds=1.2,
        )

    monkeypatch.setattr(src.chat, "ask", fake_ask)
    monkeypatch.setattr(src.agents.supervisor, "build_supervisor", lambda: object())
    return calls


def test_chat_shows_answer_details_and_sends_history(fake_agent):
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception

    at.chat_input[0].set_value("Cari chef katering").run()
    at.chat_input[0].set_value("Yang pertama pendidikannya apa?").run()

    assert not at.exception
    assert "Jawaban untuk: Cari chef katering (CV 123)" in [m.markdown[0].value for m in at.chat_message if m.markdown]
    labels = [e.label for e in at.expander]
    assert any(label.startswith("Tool calls (1)") for label in labels)
    assert any(label.startswith("Sumber RAG (1)") for label in labels)
    # Pertanyaan kedua membawa riwayat giliran pertama
    assert fake_agent[1]["history"][0] == {"role": "user", "content": "Cari chef katering"}
    assert len(fake_agent[1]["history"]) == 2


def test_example_button_and_reset(fake_agent):
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.sidebar.button[0].click().run()

    assert len(fake_agent) == 1
    assert len(at.session_state["messages"]) == 2

    reset = next(b for b in at.sidebar.button if b.label == "Reset percakapan")
    reset.click().run()
    assert at.session_state["messages"] == []


def test_agent_error_shows_friendly_message(monkeypatch):
    def broken_ask(*args, **kwargs):
        raise RuntimeError("koneksi putus")

    monkeypatch.setattr(src.chat, "ask", broken_ask)
    monkeypatch.setattr(src.agents.supervisor, "build_supervisor", lambda: object())
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.chat_input[0].set_value("halo").run()

    assert not at.exception
    assert at.error and "Maaf, terjadi galat" in at.error[0].value
