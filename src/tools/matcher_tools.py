"""Tool Matcher (RAG atas Qdrant). Tahap B2.2 sampai B2.6.

Setiap fungsi nanti dibungkus dengan @tool dari langchain-core. Docstring dibaca LLM, jadi
jelaskan kapan tool dipakai dan arti tiap argumen. Setiap tool memanggil
context.add_tool_call(...) dan context.add_sources(...).
"""


def search_resumes(query: str, category: str | None = None, section: str | None = None, top_k: int = 8) -> list[dict]:
    """Cari CV berdasarkan makna. category harus salah satu config.CATEGORIES; section salah satu config.SECTIONS."""
    raise NotImplementedError("B2.2")


def get_resume(resume_id: str) -> dict:
    """Ambil seluruh seksi satu CV (urut chunk_idx) beserta metadata."""
    raise NotImplementedError("B2.3")


def match_job(job_description: str, category: str | None = None, top_k: int = 5) -> list[dict]:
    """Cocokkan deskripsi lowongan dengan CV. Skor chunk digabung per CV."""
    raise NotImplementedError("B2.4")


def compare_candidates(resume_ids: list[str]) -> list[dict]:
    """Kembalikan profil ringkas 2 sampai 3 CV untuk dibandingkan."""
    raise NotImplementedError("B2.6")
