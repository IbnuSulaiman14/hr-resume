import pandas as pd

from src.shortlist import add_ids, shortlist_table, to_csv_bytes


def _meta():
    return pd.DataFrame(
        [
            {"resume_id": "1", "category": "CHEF", "headline": "CHEF", "years_experience": 5.0,
             "education_level": "associate", "skills": "; ".join(f"s{i}" for i in range(12))},
            {"resume_id": "2", "category": "HR", "headline": "HR MANAGER", "years_experience": None,
             "education_level": "bachelor", "skills": "payroll"},
        ]
    )


def test_add_ids_keeps_order_without_duplicates():
    assert add_ids(["2"], ["1", "2", 3]) == ["2", "1", "3"]


def test_table_follows_shortlist_order_and_skips_unknown_ids():
    table = shortlist_table(["2", "999", "1"], _meta())

    assert list(table["ID CV"]) == ["2", "1"]
    assert table.loc[1, "Skill (10 pertama)"].count(";") == 9


def test_csv_has_bom_and_header():
    data = to_csv_bytes(shortlist_table(["1"], _meta()))

    assert data.startswith(b"\xef\xbb\xbf")
    assert "ID CV,Kategori" in data.decode("utf-8-sig").splitlines()[0]
