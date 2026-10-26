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
          "education_level": "associate", "job_titles": "Chef", "skills": "cooking; menu",
          "timeline": json.dumps([{"title": "Chef", "start": "01/2015", "end": "Current", "months": 30},
                                  {"title": "Cook", "start": "01/2010", "end": "01/2013", "months": 36}]),
          "gaps": json.dumps([{"from": "2013-01", "to": "2015-01", "months": 24}])}]
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


def test_check_skills_list_text_and_missing():
    sections = {"experience": "Managed inventory control and weekly menu costing for a banquet kitchen."}
    result = matcher_tools.check_skills(
        ["Excel", "menu costing", "ServSafe", "inventory control"], ["microsoft excel", "cooking"], sections
    )

    found = {m["skill"]: m["found_in"] for m in result["matched"]}
    assert found == {"Excel": "skills", "menu costing": "experience", "inventory control": "experience"}
    assert result["missing"] == ["ServSafe"]
    assert result["coverage"] == 0.75
    assert "menu costing" in next(m["evidence"] for m in result["matched"] if m["skill"] == "menu costing")


def test_check_skills_needs_whole_phrase():
    # "sql" tidak boleh cocok dengan "mysqli", dan skill pendek di CV ("c") tidak boleh cocok di dalam kata lain
    result = matcher_tools.check_skills(["sql", "accounting"], ["c"], {"skills": "mysqli scripting"})

    assert result["matched"] == [] and result["missing"] == ["sql", "accounting"]


def test_skill_gap_tool_records_evidence_sources(fake_backend):
    context.start_run()
    output = json.loads(matcher_tools.skill_gap_tool.invoke({"resume_id": 7, "required_skills": ["cooking", "banquets", "sushi"]}))

    assert [m["skill"] for m in output["matched"]] == ["cooking", "banquets"]
    assert output["missing"] == ["sushi"]
    assert {s["section"] for s in context.get_sources()} == {"skills", "experience"}


def test_candidate_timeline_tool(fake_backend):
    context.start_run()
    output = json.loads(matcher_tools.candidate_timeline_tool.invoke({"resume_id": "7"}))

    assert [j["title"] for j in output["jobs_newest_first"]] == ["Chef", "Cook"]
    assert [j["years_estimate"] for j in output["jobs_newest_first"]] == [2.5, 3.0]
    assert output["gaps_6_months_or_more"][0]["months"] == 24
    assert context.get_sources()[0]["section"] == "experience"


def test_unknown_resume_for_new_tools(fake_backend):
    assert "error" in json.loads(matcher_tools.skill_gap_tool.invoke({"resume_id": "999", "required_skills": ["x"]}))
    assert "error" in json.loads(matcher_tools.candidate_timeline_tool.invoke({"resume_id": "999"}))


def test_search_uses_reranker_when_enabled(fake_backend, monkeypatch):
    monkeypatch.setattr(matcher_tools, "rerank_model", lambda: "model-tiruan")
    calls = {}

    def fake_query(text, limit, category=None, section=None):
        calls["limit"] = limit
        return [_hit("1", 0.9, text="Burger grill"), _hit("2", 0.8, text="Catering menus")]

    monkeypatch.setattr(matcher_tools, "_query", fake_query)
    monkeypatch.setattr(matcher_tools, "rerank_scores", lambda q, docs: [0.1, 0.95])

    results = matcher_tools.search_resumes("catering", top_k=1)

    assert calls["limit"] == 3  # top_k x RERANK_POOL_FACTOR
    assert [(r["resume_id"], r["rerank_score"]) for r in results] == [("2", 0.95)]


def test_rerank_switch_off_per_request(monkeypatch):
    monkeypatch.setenv("RERANK_MODEL", "qwen/qwen3-reranker-8b")
    from src import rerank

    context.start_run(rerank=False)
    assert rerank.rerank_model() is None
    context.start_run(rerank=True)
    assert rerank.rerank_model() == "qwen/qwen3-reranker-8b"
