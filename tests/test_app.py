"""Alur dasar Streamlit dengan agent tiruan (tanpa jaringan)."""
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import src.agents.supervisor
import src.chat
import src.shortlist
from src.chat import ChatResult
from src.prompts import PromptUnavailableError

APP = str(Path(__file__).resolve().parent.parent / "app.py")
META = pd.DataFrame(
    [{"resume_id": "123", "category": "CHEF", "headline": "EXECUTIVE CHEF", "years_experience": 6.0,
      "education_level": "associate", "skills": "cooking; catering"}]
)


@pytest.fixture
def fake_agent(monkeypatch):
    calls = []

    def fake_ask(question, history=None, session_id=None, agent=None, rerank=True):
        calls.append({"question": question, "history": list(history or []), "rerank": rerank})
        return ChatResult(
            answer=f"Jawaban untuk: {question} (CV 123)",
            sources=[{"resume_id": "123", "category": "CHEF", "section": "experience", "score": 0.81,
                      "rerank_score": 0.97, "snippet": "x"}],
            tool_calls=[{"name": "matcher", "args": {"task": question}, "agent": "supervisor"}],
            usage={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120, "llm_calls": 3,
                   "rerank_calls": 1, "rerank_tokens": 300, "cost_usd": 0.00001},
            seconds=1.2,
        )

    monkeypatch.setattr(src.chat, "ask", fake_ask)
    monkeypatch.setattr(src.agents.supervisor, "build_supervisor", lambda: object())
    monkeypatch.setattr(src.shortlist, "load_meta", lambda: META)
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
    assert any("1 panggilan reranker" in c.value for c in at.caption)
    # Pertanyaan kedua membawa riwayat giliran pertama
    assert fake_agent[1]["history"][0] == {"role": "user", "content": "Cari chef katering"}
    assert len(fake_agent[1]["history"]) == 2


def test_menu_hides_print_and_record_but_keeps_theme_switcher(fake_agent):
    at = AppTest.from_file(APP, default_timeout=30).run()
    css = next(m.value for m in at.markdown if "stMainMenuItem-print" in m.value)

    assert "stMainMenuItem-recordScreencast" in css and "stMainMenuDivider" in css
    assert "display: none" in css
    assert "stThemeSwitcher" not in css and "stMainMenuItem-theme" not in css  # pemilih tema tidak boleh ikut hilang


def test_welcome_cards_shown_until_first_question(fake_agent):
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert any(s.value == "Tanya tentang kandidat" for s in at.title)
    assert len([b for b in at.button if b.key and b.key.startswith("welcome_")]) == 6
    # Sidebar tidak lagi punya daftar contoh pertanyaan; contoh hanya ada di kartu sambutan
    assert not any(e.label == "Contoh pertanyaan" for e in at.sidebar.expander)
    assert not any(b.key and b.key.startswith("example_") for b in at.button)

    at.chat_input[0].set_value("Cari chef").run()
    assert not any(s.value == "Tanya tentang kandidat" for s in at.title)
    assert not any(b.key and b.key.startswith("welcome_") for b in at.button)


def test_latest_answer_opens_sources_and_shows_tool_badges(fake_agent):
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.chat_input[0].set_value("Cari chef").run()
    at.chat_input[0].set_value("Yang kedua?").run()

    sources = [e for e in at.expander if e.label.startswith("Sumber RAG")]
    assert [e.proto.expanded for e in sources] == [False, True]  # hanya jawaban terbaru yang terbuka
    assert any(":violet-badge[matcher]" in m.value for m in at.markdown)


def test_chart_from_tool_is_drawn_with_data_table(monkeypatch):
    chart = {
        "title": "Jumlah CV per kategori", "orientation": "horizontal", "label_title": "Kategori",
        "value_title": "Jumlah CV", "data": [{"label": "CHEF", "value": 3}],
    }

    def ask_with_chart(question, history=None, session_id=None, agent=None, rerank=True):
        return ChatResult(answer="Ada 3 CV CHEF.", charts=[chart], seconds=0.5,
                          tool_calls=[{"name": "category_distribution", "args": {}, "agent": "analyst"}])

    monkeypatch.setattr(src.chat, "ask", ask_with_chart)
    monkeypatch.setattr(src.agents.supervisor, "build_supervisor", lambda: object())
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.chat_input[0].set_value("Sebaran kategori").run()

    assert not at.exception
    assert len(at.get("vega_lite_chart")) == 1
    assert any(e.label == "Data grafik: Jumlah CV per kategori" for e in at.expander)


def test_shortlist_add_download_and_clear(fake_agent):
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.chat_input[0].set_value("Cari chef katering").run()

    at.multiselect(key="pick_1").set_value(["123"]).run()
    next(b for b in at.button if b.label == "Tambah ke shortlist").click().run()

    assert at.session_state["shortlist"] == ["123"]
    assert any(s.value == "Shortlist (1)" for s in at.sidebar.subheader)
    # Reset percakapan tidak menghapus shortlist
    next(b for b in at.sidebar.button if b.label == "Reset percakapan").click().run()
    assert at.session_state["shortlist"] == ["123"]
    next(b for b in at.sidebar.button if b.label == "Kosongkan shortlist").click().run()
    assert at.session_state["shortlist"] == []


def test_example_button_and_reset(fake_agent):
    at = AppTest.from_file(APP, default_timeout=30).run()
    next(b for b in at.button if b.key == "welcome_0").click().run()

    assert len(fake_agent) == 1
    assert len(at.session_state["messages"]) == 2

    reset = next(b for b in at.sidebar.button if b.label == "Reset percakapan")
    reset.click().run()
    assert at.session_state["messages"] == []


@pytest.mark.parametrize(
    "exc, expected",
    [(RuntimeError("koneksi putus"), "Maaf, terjadi galat"), (PromptUnavailableError("x"), "tidak bisa diambil dari Langfuse")],
)
def test_agent_error_shows_friendly_message(monkeypatch, exc, expected):
    def broken_ask(*args, **kwargs):
        raise exc

    monkeypatch.setattr(src.chat, "ask", broken_ask)
    monkeypatch.setattr(src.agents.supervisor, "build_supervisor", lambda: object())
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.chat_input[0].set_value("halo").run()

    assert not at.exception
    assert at.error and expected in at.error[0].value
