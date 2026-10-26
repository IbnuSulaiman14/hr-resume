import json

import pandas as pd
import pytest

from src.tools import analyst_tools, context


@pytest.fixture
def meta():
    return pd.DataFrame(
        [
            {"resume_id": "1", "category": "CHEF", "headline": "EXECUTIVE CHEF", "years_experience": 12.0,
             "education_level": "associate", "job_titles": "Executive Chef; Line Cook", "skills": "cooking; menu"},
            {"resume_id": "2", "category": "CHEF", "headline": "SOUS CHEF", "years_experience": 4.5,
             "education_level": "high_school", "job_titles": "Sous Chef", "skills": "cooking; catering"},
            {"resume_id": "3", "category": "CHEF", "headline": "COOK", "years_experience": None,
             "education_level": "unknown", "job_titles": "Cook", "skills": "cooking"},
            {"resume_id": "4", "category": "HR", "headline": "HR MANAGER", "years_experience": 8.0,
             "education_level": "bachelor", "job_titles": "HR Manager", "skills": "payroll"},
        ]
    )


def test_category_distribution(meta):
    result = analyst_tools.category_distribution(meta)

    assert result == {"total_cv": 4, "per_category": {"CHEF": 3, "HR": 1}}


def test_filter_by_category_and_min_years_reports_missing_data(meta):
    result = analyst_tools.filter_candidates(category="CHEF", min_years=5, df=meta)

    assert result["total_match"] == 1
    assert [r["resume_id"] for r in result["rows"]] == ["1"]
    # CV 3 tidak punya data pengalaman, jadi tidak terhitung dan dilaporkan
    assert result["excluded_no_years_data"] == 1


def test_filter_title_and_education(meta):
    assert analyst_tools.filter_candidates(title_contains="manager", df=meta)["total_match"] == 1
    assert analyst_tools.filter_candidates(education_level="associate", df=meta)["total_match"] == 1


def test_filter_limit_only_limits_rows_not_total(meta):
    result = analyst_tools.filter_candidates(category="CHEF", limit=1, df=meta)

    assert result["total_match"] == 3
    assert result["shown"] == 1


def test_skill_frequency_counts_each_cv_once(meta):
    result = analyst_tools.skill_frequency("CHEF", top_n=2, df=meta)

    assert result["total_cv"] == 3
    assert result["top_skills"][0] == {"skill": "cooking", "cv_count": 3, "share": 1.0}
    assert len(result["top_skills"]) == 2


def test_filter_tool_records_call_and_sources(meta, monkeypatch):
    monkeypatch.setattr(analyst_tools, "load_meta", lambda: meta)
    context.start_run()

    output = json.loads(analyst_tools.filter_candidates_tool.invoke({"category": "HR"}))

    assert output["total_match"] == 1
    assert context.get_tool_calls()[0]["name"] == "filter_candidates"
    assert context.get_sources()[0]["resume_id"] == "4"


def test_years_distribution_trims_empty_ends_but_keeps_gaps():
    years = pd.Series([3.0, 4.9, 12.0, None])

    # 0-2 dan 15-20+ kosong di ujung dibuang; 5-10 kosong di tengah tetap ada
    assert analyst_tools.years_distribution(years) == [
        {"years": "2-5", "cv_count": 2}, {"years": "5-10", "cv_count": 0}, {"years": "10-15", "cv_count": 1},
    ]
    assert analyst_tools.years_distribution(pd.Series([None], dtype=float)) == []


def test_year_bins_include_left_edge():
    # "di atas 5 tahun" memakai min_years=5, jadi tepat 5 tahun harus masuk 5-10
    assert analyst_tools.years_distribution(pd.Series([5.0, 20.0])) == [
        {"years": "5-10", "cv_count": 1}, {"years": "10-15", "cv_count": 0},
        {"years": "15-20", "cv_count": 0}, {"years": "20+", "cv_count": 1},
    ]


def test_tools_record_charts_with_the_same_numbers(meta, monkeypatch):
    monkeypatch.setattr(analyst_tools, "load_meta", lambda: meta)
    context.start_run()

    distribution = json.loads(analyst_tools.category_distribution_tool.invoke({}))
    skills = json.loads(analyst_tools.skill_frequency_tool.invoke({"category": "CHEF"}))
    filtered = json.loads(analyst_tools.filter_candidates_tool.invoke({"category": "CHEF"}))

    assert distribution["ui_note"] == skills["ui_note"] == filtered["ui_note"] == analyst_tools.CHART_NOTE
    category, skill, years = context.get_charts()
    assert {d["label"]: d["value"] for d in category["data"]} == distribution["per_category"]
    assert [d["value"] for d in skill["data"]] == [s["cv_count"] for s in skills["top_skills"]]
    assert [d["value"] for d in years["data"]] == [b["cv_count"] for b in filtered["years_distribution"]]
    assert years["orientation"] == "vertical" and category["orientation"] == "horizontal"
    # 3 CV CHEF cocok, hanya 2 punya tahun: subjudul harus menjelaskan selisihnya
    assert years["subtitle"] == "3 CV cocok dengan filter; 1 di antaranya tanpa data pengalaman, tidak digambar"


def test_years_chart_with_year_filter_reports_excluded_cvs(meta, monkeypatch):
    monkeypatch.setattr(analyst_tools, "load_meta", lambda: meta)
    context.start_run()

    analyst_tools.filter_candidates_tool.invoke({"category": "CHEF", "min_years": 5})

    (years,) = context.get_charts()
    assert sum(d["value"] for d in years["data"]) == 1
    assert years["subtitle"] == "1 CV cocok dengan filter; 1 CV tanpa data pengalaman tidak ikut dihitung"


def test_no_chart_when_nothing_to_draw(meta, monkeypatch):
    monkeypatch.setattr(analyst_tools, "load_meta", lambda: meta)
    context.start_run()

    filtered = json.loads(analyst_tools.filter_candidates_tool.invoke({"category": "CHEF", "min_years": 50}))
    skills = json.loads(analyst_tools.skill_frequency_tool.invoke({"category": "ACCOUNTANT"}))

    assert context.get_charts() == []
    assert "ui_note" not in filtered and "ui_note" not in skills  # LLM tidak boleh menyebut grafik yang tidak ada


def test_tool_schema_restricts_category_values():
    schema = analyst_tools.skill_frequency_tool.args_schema.model_json_schema()
    category = json.dumps(schema["properties"]["category"])

    assert "INFORMATION-TECHNOLOGY" in category and "CHEF" in category
