"""Shortlist kandidat: tabel ringkas dari metadata dan ekspor CSV untuk diunduh dari Streamlit."""
import pandas as pd

from src.tools.analyst_tools import LIST_SEPARATOR, load_meta

COLUMNS = {
    "resume_id": "ID CV",
    "category": "Kategori",
    "headline": "Judul CV",
    "years_experience": "Perkiraan tahun pengalaman",
    "education_level": "Perkiraan pendidikan",
    "skills": "Skill (10 pertama)",
}


def add_ids(shortlist: list[str], new_ids: list[str]) -> list[str]:
    """Tambahkan ID tanpa duplikat, urutan tambah dipertahankan."""
    return list(dict.fromkeys([*shortlist, *map(str, new_ids)]))


def shortlist_table(resume_ids: list[str], meta: pd.DataFrame | None = None) -> pd.DataFrame:
    """Satu baris per kandidat di shortlist, urut sesuai urutan ditambahkan. ID yang tidak ada di metadata dilewati."""
    meta = load_meta() if meta is None else meta
    rows = meta.set_index("resume_id").reindex([i for i in resume_ids if i in set(meta["resume_id"])]).reset_index()
    rows["skills"] = rows["skills"].fillna("").map(lambda s: LIST_SEPARATOR.join(s.split(LIST_SEPARATOR)[:10]))
    return rows[list(COLUMNS)].rename(columns=COLUMNS)


def to_csv_bytes(table: pd.DataFrame) -> bytes:
    # utf-8-sig supaya Excel membaca karakter non-ASCII dengan benar
    return table.to_csv(index=False).encode("utf-8-sig")
