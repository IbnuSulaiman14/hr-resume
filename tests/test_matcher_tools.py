import json

import pandas as pd
import pytest

from src.tools import context, matcher_tools


def _hit(resume_id, score, section="experience", category="CHEF", text="Led banquet kitchen"):
    return {"resume_id": resume_id, "score": score, "section": section, "category": category, "text": text}


def test_aggregate_one_entry_per_resume_sorted_by_score():
    hits = [
        _hit("A", 0.80, "experience"),
        _hit("A", 0.70, "skills"),
        _hit("A", 0.60, "summary"),
        _hit("B", 0.82, "skills"),
        _hit("C", 0.50),
    ]
    results = matcher_tools.aggregate_by_resume(hits, top_k=2)

    # A: 0.7*0.80 + 0.3*0.70 = 0.77; B: 0.7*0.82 + 0.3*0.82 = 0.82
    assert [r["resume_id"] for r in results] == ["B", "A"]
    assert results[1]["score"] == pytest.approx(0.77)
    assert results[1]["matched_sections"] == ["experience", "skills", "summary"]
    assert results[1]["best_section"] == "experience"


def test_aggregate_empty():
    assert matcher_tools.aggregate_by_resume([], top_k=5) == []


def test_snippet_cuts_on_word_boundary():
    assert matcher_tools._snippet("one two three four", limit=9) == "one two ..."
    assert matcher_tools._snippet("short") == "short"


@pytest.fixture
def fake_backend(monkeypatch):
    meta = pd.DataFrame(
        [{"resume_id": "7", "category": "CHEF", "headline": "CHEF", "years_experience": 6.0,
          "education_level": "associate", "job_titles": "Chef", "skills": "cooking; menu"}]
    )
    monkeypatch.setattr(matcher_tools, "load_meta", lambda: meta)
    monkeypatch.setattr(
        matcher_tools, "_query", lambda text, limit, category=None, section=None: [_hit("7", 0.9, text="Catering menus")]
    )
    monkeypatch.setattr(
        matcher_tools,
        "_resume_chunks",
        lambda rid: [
            {"resume_id": rid, "category": "CHEF", "section": "summary", "chunk_idx": 0, "text": "Chef summary"},
            {"resume_id": rid, "category": "CHEF", "section": "experience", "chunk_idx": 1, "text": "Banquets"},
        ]
        if rid == "7"
        else [],
    )


def test_search_tool_records_sources_and_call(fake_backend):
    context.start_run()
    output = json.loads(matcher_tools.search_resumes_tool.invoke({"query": "catering", "category": "CHEF"}))

    assert output[0]["resume_id"] == "7"
    assert context.get_tool_calls()[0] == {
        "name": "search_resumes", "args": {"query": "catering", "category": "CHEF", "top_k": 8}, "agent": "matcher"
    }
    assert context.get_sources()[0]["snippet"] == "Catering menus"


def test_get_resume_accepts_integer_id_and_orders_sections(fake_backend):
    context.start_run()
    output = json.loads(matcher_tools.get_resume_tool.invoke({"resume_id": 7}))

    assert list(output["sections"]) == ["summary", "experience"]
    assert output["years_experience_estimate"] == 6.0


def test_get_resume_unknown_id(fake_backend):
    output = json.loads(matcher_tools.get_resume_tool.invoke({"resume_id": "999"}))

    assert "error" in output


def test_compare_requires_two_ids(fake_backend):
    output = json.loads(matcher_tools.compare_candidates_tool.invoke({"resume_ids": ["7"]}))

    assert "error" in output
