"""Tool Analyst (Pandas atas data/resumes_meta.csv). Tahap B2.5 dan B2.6.

Angka dihitung di sini, bukan oleh LLM.
"""


def category_distribution() -> dict[str, int]:
    """Jumlah CV per kategori."""
    raise NotImplementedError("B2.5")


def filter_candidates(
    category: str | None = None,
    min_years: float | None = None,
    education_level: str | None = None,
    title_contains: str | None = None,
    limit: int = 20,
) -> dict:
    """Saring metadata CV. Kembalikan baris yang cocok dan jumlah total."""
    raise NotImplementedError("B2.6")


def skill_frequency(category: str, top_n: int = 15) -> list[dict]:
    """Skill yang paling sering muncul di satu kategori."""
    raise NotImplementedError("B2.6")
