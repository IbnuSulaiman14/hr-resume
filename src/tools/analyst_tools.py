"""Tool Analyst (Pandas atas data/resumes_meta.csv). Tahap B2.5 dan B2.6.

Angka dihitung di sini, bukan oleh LLM. Fungsi biasa (tanpa @tool) bisa diuji langsung;
pembungkus @tool untuk agent ada di bagian bawah.
"""
import json
from collections import Counter
from functools import lru_cache
from typing import Literal

import pandas as pd
from langchain_core.tools import tool

from src.config import CATEGORIES, META_CSV
from src.tools import context

EDUCATION_VALUES = ["high_school", "associate", "bachelor", "master", "doctorate", "unknown"]
LIST_SEPARATOR = "; "

CategoryName = Literal[tuple(CATEGORIES)]
EducationName = Literal[tuple(EDUCATION_VALUES)]


@lru_cache(maxsize=1)
def load_meta(path: str = str(META_CSV)) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"resume_id": str})
    for column in ["headline", "job_titles", "skills", "education_level"]:
        df[column] = df[column].fillna("")
    return df


def _split(value: str) -> list[str]:
    return [item for item in value.split(LIST_SEPARATOR) if item] if value else []


def category_distribution(df: pd.DataFrame | None = None) -> dict:
    """Jumlah CV per kategori, urut dari yang terbanyak."""
    df = load_meta() if df is None else df
    counts = df["category"].value_counts()
    return {"total_cv": int(len(df)), "per_category": {k: int(v) for k, v in counts.items()}}


def filter_candidates(
    category: str | None = None,
    min_years: float | None = None,
    max_years: float | None = None,
    education_level: str | None = None,
    title_contains: str | None = None,
    limit: int = 20,
    df: pd.DataFrame | None = None,
) -> dict:
    """Saring metadata CV. Kembalikan jumlah total yang cocok dan sebagian baris (maksimal `limit`)."""
    df = load_meta() if df is None else df
    mask = pd.Series(True, index=df.index)
    if category:
        mask &= df["category"] == category
    if education_level:
        mask &= df["education_level"] == education_level
    if title_contains:
        needle = title_contains.lower()
        mask &= df["job_titles"].str.lower().str.contains(needle, regex=False) | df["headline"].str.lower().str.contains(
            needle, regex=False
        )
    # Hitung CV tanpa data pengalaman sebelum filter tahun, supaya bisa disebut di jawaban
    without_years = int((mask & df["years_experience"].isna()).sum()) if (min_years or max_years) else 0
    if min_years is not None:
        mask &= df["years_experience"] >= min_years
    if max_years is not None:
        mask &= df["years_experience"] <= max_years

    matched = df[mask].sort_values("years_experience", ascending=False)
    rows = [
        {
            "resume_id": r.resume_id,
            "category": r.category,
            "headline": r.headline,
            "years_experience": None if pd.isna(r.years_experience) else float(r.years_experience),
            "education_level": r.education_level,
        }
        for r in matched.head(limit).itertuples()
    ]
    return {"total_match": int(len(matched)), "shown": len(rows), "excluded_no_years_data": without_years, "rows": rows}


def skill_frequency(category: str, top_n: int = 15, df: pd.DataFrame | None = None) -> dict:
    """Skill yang paling sering muncul di satu kategori. Satu CV dihitung sekali per skill."""
    df = load_meta() if df is None else df
    subset = df[df["category"] == category]
    counter: Counter[str] = Counter()
    for skills in subset["skills"]:
        counter.update(set(_split(skills)))
    total = len(subset)
    return {
        "category": category,
        "total_cv": total,
        "top_skills": [
            {"skill": skill, "cv_count": count, "share": round(count / total, 3) if total else 0.0}
            for skill, count in counter.most_common(top_n)
        ],
    }


def _metadata_sources(rows: list[dict]) -> list[dict]:
    return [
        {
            "resume_id": r["resume_id"],
            "category": r["category"],
            "section": "metadata",
            "score": None,
            "snippet": f"{r['headline']} | pengalaman ~{r['years_experience']} th | pendidikan {r['education_level']}",
        }
        for r in rows
    ]


def _dump(data) -> str:
    return json.dumps(data, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Pembungkus @tool untuk agent Analyst. Docstring di bawah dibaca LLM.


@tool("category_distribution")
def category_distribution_tool() -> str:
    """Hitung jumlah CV per kategori untuk seluruh dataset (24 kategori) beserta total CV.
    Pakai untuk pertanyaan "berapa CV di kategori X", "kategori terbanyak", atau sebaran kategori."""
    context.add_tool_call("category_distribution", {}, agent="analyst")
    return _dump(category_distribution())


@tool("filter_candidates")
def filter_candidates_tool(
    category: CategoryName | None = None,
    min_years: float | None = None,
    max_years: float | None = None,
    education_level: EducationName | None = None,
    title_contains: str | None = None,
    limit: int = 10,
) -> str:
    """Saring CV berdasarkan metadata terstruktur dan hitung jumlah yang cocok.

    Args:
        category: kategori CV (opsional).
        min_years: minimal perkiraan tahun pengalaman, inklusif. "Di atas 5 tahun" berarti min_years=5.
        max_years: maksimal perkiraan tahun pengalaman, inklusif.
        education_level: level pendidikan tertinggi hasil ekstraksi.
        title_contains: kata yang harus ada di jabatan atau judul CV, bahasa Inggris (mis. "manager").
        limit: jumlah baris contoh yang ditampilkan (bukan batas hitungan). Total selalu dihitung penuh.
    """
    args = {k: v for k, v in locals().items() if v is not None}
    context.add_tool_call("filter_candidates", args, agent="analyst")
    result = filter_candidates(category, min_years, max_years, education_level, title_contains, limit)
    context.add_sources(_metadata_sources(result["rows"]))
    return _dump(result)


@tool("skill_frequency")
def skill_frequency_tool(category: CategoryName, top_n: int = 15) -> str:
    """Daftar skill yang paling sering muncul di seksi Skills CV pada satu kategori, dengan jumlah CV
    yang menyebutnya dan proporsinya. Pakai untuk "skill apa yang paling umum di kategori X"."""
    context.add_tool_call("skill_frequency", {"category": category, "top_n": top_n}, agent="analyst")
    return _dump(skill_frequency(category, top_n))


ANALYST_TOOLS = [category_distribution_tool, filter_candidates_tool, skill_frequency_tool]
