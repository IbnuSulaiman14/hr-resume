from src.ingest.chunking import build_chunks, chunk_prefix, count_tokens
from src.ingest.parse_html import ParsedResume


def _resume(sections: dict[str, str]) -> ParsedResume:
    return ParsedResume(resume_id="42", category="CHEF", sections=sections)


def test_one_chunk_per_short_section_with_prefix():
    chunks = build_chunks(_resume({"summary": "Chef with catering background.", "skills": "Cooking\nMenu"}))

    assert [c.section for c in chunks] == ["summary", "skills"]
    assert [c.chunk_idx for c in chunks] == [0, 1]
    assert chunks[0].text == "[CHEF] [summary]\nChef with catering background."


def test_long_section_split_within_limit_with_overlap():
    lines = [f"Managed banquet number {i} for corporate clients with seasonal menu." for i in range(120)]
    chunks = build_chunks(_resume({"experience": "\n".join(lines)}), max_tokens=100, overlap_tokens=20)

    assert len(chunks) > 1
    assert [c.chunk_idx for c in chunks] == list(range(len(chunks)))
    prefix = chunk_prefix("CHEF", "experience")
    bodies = [c.text.removeprefix(prefix) for c in chunks]
    assert all(c.text.startswith(prefix) for c in chunks)
    assert all(0 < count_tokens(b) <= 100 for b in bodies)
    # Overlap: baris terakhir chunk pertama muncul lagi di awal chunk kedua
    assert bodies[0].split("\n")[-1] in bodies[1]


def test_empty_sections_are_skipped():
    chunks = build_chunks(_resume({"summary": "  ", "skills": "SQL"}))

    assert [(c.section, c.chunk_idx) for c in chunks] == [("skills", 0)]
