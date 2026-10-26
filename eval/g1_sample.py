"""Pemeriksaan hasil parsing: python -m eval.g1_sample [--n 40]

Menulis eval/g1_sample.md berisi CV acak berdampingan: teks asli (dipotong), seksi terparse,
tanggal kerja, perkiraan tahun pengalaman, gelar, dan level pendidikan. Dipakai untuk pemeriksaan manual.
"""
import argparse

from src.config import ROOT
from src.ingest.build_index import load_resumes
from src.ingest.parse_html import education_level, estimate_years_experience, parse_resume, reference_date

OUTPUT = ROOT / "eval" / "g1_sample.md"


def main(n: int = 40, seed: int = 7) -> None:
    df = load_resumes().sample(n=n, random_state=seed)
    lines = ["# Sampel pemeriksaan parsing", "", f"{n} CV acak (seed {seed}).", ""]
    for row in df.itertuples():
        parsed = parse_resume(row.ID, row.Category, row.Resume_html, row.Resume_str)
        years = estimate_years_experience(parsed.job_dates, today=reference_date(parsed.job_dates, row.Resume_str))
        lines += [
            f"## {row.ID} ({row.Category})",
            f"- Seksi: {', '.join(f'{k} ({len(v)} kar)' for k, v in parsed.sections.items())}",
            f"- Jabatan: {'; '.join(parsed.job_titles[:6])}",
            f"- Tanggal kerja: {parsed.job_dates}",
            f"- Perkiraan tahun pengalaman: {years}",
            f"- Gelar: {parsed.degrees[:4]} -> {education_level(parsed.degrees)}",
            f"- Skill (10 pertama): {parsed.skills[:10]}",
            "",
            "```",
            " ".join(row.Resume_str.split())[:1500],
            "```",
            "",
        ]
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Ditulis: {OUTPUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=40)
    args = parser.parse_args()
    main(n=args.n)
