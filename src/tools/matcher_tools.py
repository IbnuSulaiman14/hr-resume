"""Tool Matcher (RAG atas Qdrant). Tahap B2.2 sampai B2.6.

Fungsi biasa melakukan kerja (embed query, cari di Qdrant, susun hasil); pembungkus @tool di bagian
bawah mencatat tool call dan sumber ke context lalu mengembalikan JSON untuk LLM.
"""
import json
from collections import defaultdict
from functools import lru_cache
from typing import Literal

from langchain_core.tools import tool
from qdrant_client import models

from src.config import CATEGORIES, SECTIONS
from src.models import get_embeddings
from src.tools import context
from src.tools.analyst_tools import LIST_SEPARATOR, load_meta
from src.vectorstore import collection_name, get_qdrant_client

CategoryName = Literal[tuple(CATEGORIES)]
SectionName = Literal[tuple(SECTIONS)]

SNIPPET_CHARS = 350
RESUME_MAX_CHARS = 8000
# match_job mengambil lebih banyak chunk daripada CV yang diminta, lalu menggabungkannya per CV
MATCH_CANDIDATE_CHUNKS = 60
# Bobot skor agregat: chunk terbaik paling menentukan, rata-rata 3 chunk teratas memberi bonus untuk CV
# yang cocok di beberapa seksi sekaligus
BEST_WEIGHT = 0.7
TOP_N_FOR_MEAN = 3


@lru_cache(maxsize=1)
def _embeddings():
    return get_embeddings()


def _snippet(text: str, limit: int = SNIPPET_CHARS) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + " ..."


def _filter(category: str | None = None, section: str | None = None, resume_id: str | None = None):
    conditions = [
        models.FieldCondition(key=key, match=models.MatchValue(value=value))
        for key, value in [("category", category), ("section", section), ("resume_id", resume_id)]
        if value
    ]
    return models.Filter(must=conditions) if conditions else None


def _query(text: str, limit: int, category: str | None = None, section: str | None = None) -> list[dict]:
    vector = _embeddings().embed_query(text)
    response = get_qdrant_client().query_points(
        collection_name(), query=vector, query_filter=_filter(category, section), limit=limit, with_payload=True
    )
    return [{**point.payload, "score": round(point.score, 4)} for point in response.points]


def search_resumes(query: str, category: str | None = None, section: str | None = None, top_k: int = 8) -> list[dict]:
    """Cari chunk CV berdasarkan makna. category harus salah satu config.CATEGORIES; section salah satu config.SECTIONS."""
    hits = _query(query, top_k, category, section)
    return [
        {
            "resume_id": h["resume_id"],
            "category": h["category"],
            "section": h["section"],
            "score": h["score"],
            "snippet": _snippet(h["text"]),
        }
        for h in hits
    ]


def aggregate_by_resume(hits: list[dict], top_k: int) -> list[dict]:
    """Gabungkan skor chunk menjadi satu skor per CV supaya satu CV tidak muncul berulang.
    skor = 0.7 x chunk terbaik + 0.3 x rata-rata 3 chunk teratas CV itu."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for hit in hits:
        grouped[hit["resume_id"]].append(hit)

    results = []
    for resume_id, items in grouped.items():
        items.sort(key=lambda h: h["score"], reverse=True)
        top = items[:TOP_N_FOR_MEAN]
        mean_top = sum(h["score"] for h in top) / len(top)
        score = BEST_WEIGHT * items[0]["score"] + (1 - BEST_WEIGHT) * mean_top
        results.append(
            {
                "resume_id": resume_id,
                "category": items[0]["category"],
                "score": round(score, 4),
                "matched_sections": sorted({h["section"] for h in items}),
                "best_section": items[0]["section"],
                "snippet": _snippet(items[0]["text"]),
            }
        )
    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:top_k]


def _meta_row(resume_id: str) -> dict | None:
    df = load_meta()
    rows = df[df["resume_id"] == resume_id]
    if rows.empty:
        return None
    row = rows.iloc[0]
    years = row["years_experience"]
    return {
        "resume_id": resume_id,
        "category": row["category"],
        "headline": row["headline"],
        "years_experience_estimate": None if years != years else float(years),  # NaN != NaN
        "education_level_estimate": row["education_level"],
        "job_titles": [t for t in row["job_titles"].split(LIST_SEPARATOR) if t],
        "skills": [s for s in row["skills"].split(LIST_SEPARATOR) if s],
    }


def match_job(job_description: str, category: str | None = None, top_k: int = 5) -> list[dict]:
    """Cocokkan deskripsi lowongan dengan CV. Skor chunk digabung per CV, lalu ditambah metadata."""
    hits = _query(job_description, MATCH_CANDIDATE_CHUNKS, category)
    results = aggregate_by_resume(hits, top_k)
    for result in results:
        meta = _meta_row(result["resume_id"]) or {}
        result["years_experience_estimate"] = meta.get("years_experience_estimate")
        result["education_level_estimate"] = meta.get("education_level_estimate")
    return results


def _resume_chunks(resume_id: str) -> list[dict]:
    points, _ = get_qdrant_client().scroll(
        collection_name(), scroll_filter=_filter(resume_id=resume_id), limit=64, with_payload=True
    )
    return sorted((p.payload for p in points), key=lambda p: p["chunk_idx"])


def get_resume(resume_id: str) -> dict:
    """Ambil seluruh seksi satu CV (urut chunk_idx) beserta metadata."""
    meta = _meta_row(resume_id)
    chunks = _resume_chunks(resume_id)
    if meta is None or not chunks:
        return {"error": f"CV dengan ID {resume_id} tidak ditemukan."}
    sections: dict[str, str] = {}
    for chunk in chunks:
        sections[chunk["section"]] = (sections.get(chunk["section"], "") + "\n" + chunk["text"]).strip()
    remaining = RESUME_MAX_CHARS
    for name, text in sections.items():
        sections[name] = text[: max(remaining, 0)]
        remaining -= len(sections[name])
    meta["skills"] = meta["skills"][:30]
    return {**meta, "sections": {k: v for k, v in sections.items() if v}}


def compare_candidates(resume_ids: list[str]) -> list[dict]:
    """Kembalikan profil ringkas 2 sampai 3 CV untuk dibandingkan."""
    profiles = []
    for resume_id in resume_ids:
        meta = _meta_row(resume_id)
        if meta is None:
            profiles.append({"resume_id": resume_id, "error": "CV tidak ditemukan."})
            continue
        chunks = _resume_chunks(resume_id)
        by_section = {c["section"]: c["text"] for c in reversed(chunks)}  # chunk pertama tiap seksi
        meta["summary"] = _snippet(by_section.get("summary", ""), 500)
        meta["experience_snippet"] = _snippet(by_section.get("experience", ""), 500)
        meta["job_titles"] = meta["job_titles"][:6]
        meta["skills"] = meta["skills"][:15]
        profiles.append(meta)
    return profiles


def _dump(data) -> str:
    return json.dumps(data, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Pembungkus @tool untuk agent Matcher. Docstring di bawah dibaca LLM.


@tool("search_resumes")
def search_resumes_tool(
    query: str, category: CategoryName | None = None, section: SectionName | None = None, top_k: int = 8
) -> str:
    """Cari potongan CV yang maknanya paling mirip dengan query (pencarian semantik di Qdrant).
    Pakai untuk mencari kandidat berdasarkan keahlian, pengalaman, atau topik.

    Args:
        query: kalimat pencarian dalam bahasa Inggris, mis. "catering and menu development experience".
        category: batasi ke satu kategori CV. Kosongkan bila kategori tidak jelas.
        section: batasi ke satu seksi CV (summary, skills, experience, education, certifications, other).
        top_k: jumlah potongan yang dikembalikan (1 sampai 15).
    """
    top_k = max(1, min(top_k, 15))
    args = {k: v for k, v in {"query": query, "category": category, "section": section, "top_k": top_k}.items() if v}
    context.add_tool_call("search_resumes", args, agent="matcher")
    results = search_resumes(query, category, section, top_k)
    context.add_sources(results)
    return _dump(results)


@tool("match_job")
def match_job_tool(job_description: str, category: CategoryName | None = None, top_k: int = 5) -> str:
    """Cocokkan deskripsi lowongan dengan CV. Skor potongan CV digabung per CV, jadi tiap kandidat
    muncul sekali dengan skor agregat, seksi yang cocok, dan cuplikan terbaik.

    Args:
        job_description: deskripsi lowongan (tugas, skill, kualifikasi), sebaiknya dalam bahasa Inggris.
        category: batasi ke satu kategori CV bila lowongan jelas masuk kategori tertentu.
        top_k: jumlah kandidat yang dikembalikan (1 sampai 10).
    """
    top_k = max(1, min(top_k, 10))
    args = {k: v for k, v in {"job_description": job_description, "category": category, "top_k": top_k}.items() if v}
    context.add_tool_call("match_job", args, agent="matcher")
    results = match_job(job_description, category, top_k)
    context.add_sources(
        [
            {k: r[k] for k in ("resume_id", "category", "score", "snippet")} | {"section": r["best_section"]}
            for r in results
        ]
    )
    return _dump(results)


@tool("get_resume")
def get_resume_tool(resume_id: str | int) -> str:
    """Ambil isi lengkap satu CV per seksi beserta metadata (perkiraan tahun pengalaman, level pendidikan,
    jabatan, skill). Pakai untuk pertanyaan detail tentang satu kandidat, mis. pendidikan kandidat kedua.

    Args:
        resume_id: ID CV (angka) dari hasil pencarian atau percakapan sebelumnya.
    """
    resume_id = str(resume_id)
    context.add_tool_call("get_resume", {"resume_id": resume_id}, agent="matcher")
    result = get_resume(resume_id)
    if "error" not in result:
        context.add_sources(
            [
                {"resume_id": resume_id, "category": result["category"], "section": name, "score": None,
                 "snippet": _snippet(text, 200)}
                for name, text in result["sections"].items()
            ]
        )
    return _dump(result)


@tool("compare_candidates")
def compare_candidates_tool(resume_ids: list[str | int]) -> str:
    """Bandingkan 2 sampai 3 CV berdampingan: judul, perkiraan tahun pengalaman, pendidikan, jabatan,
    skill, ringkasan, dan cuplikan pengalaman.

    Args:
        resume_ids: daftar 2 sampai 3 ID CV.
    """
    ids = [str(i) for i in resume_ids][:3]
    context.add_tool_call("compare_candidates", {"resume_ids": ids}, agent="matcher")
    if len(ids) < 2:
        return _dump({"error": "Butuh minimal 2 ID CV untuk dibandingkan."})
    profiles = compare_candidates(ids)
    context.add_sources(
        [
            {"resume_id": p["resume_id"], "category": p["category"], "section": "summary", "score": None,
             "snippet": p["summary"] or p["headline"]}
            for p in profiles
            if "error" not in p
        ]
    )
    return _dump(profiles)


MATCHER_TOOLS = [search_resumes_tool, match_job_tool, get_resume_tool, compare_candidates_tool]
