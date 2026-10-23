"""Parse kolom Resume_html menjadi seksi dan metadata. Tahap B1.1 dan B1.2.

Fungsi di sini murni (tanpa jaringan) supaya mudah diuji.
"""
from dataclasses import dataclass, field
from datetime import date


@dataclass
class ParsedResume:
    resume_id: str
    category: str
    sections: dict[str, str] = field(default_factory=dict)  # nama seksi ternormalisasi -> teks
    job_titles: list[str] = field(default_factory=list)
    job_dates: list[tuple[str, str]] = field(default_factory=list)  # ("MM/YYYY", "MM/YYYY" | "Current")
    degrees: list[str] = field(default_factory=list)


def normalize_section(title: str) -> str:
    """Petakan judul seksi mentah ("Work History", "Professional Summary") ke salah satu config.SECTIONS."""
    raise NotImplementedError("B1.1")


def parse_resume(resume_id: str, category: str, html: str, text: str) -> ParsedResume:
    """Ambil seksi, jabatan, tanggal, dan gelar dari HTML. Bila tidak ada seksi terparse, kembalikan
    satu seksi "other" berisi `text` (cadangan)."""
    raise NotImplementedError("B1.1")


def estimate_years_experience(job_dates: list[tuple[str, str]], today: date | None = None) -> float | None:
    """Perkiraan total tahun pengalaman dari rentang tanggal kerja. "Current" berarti `today`.
    Kembalikan None bila tidak ada tanggal yang bisa dibaca."""
    raise NotImplementedError("B1.2")


def education_level(degrees: list[str]) -> str:
    """Petakan daftar gelar kotor ke: high_school, associate, bachelor, master, doctorate, atau unknown."""
    raise NotImplementedError("B1.2")
