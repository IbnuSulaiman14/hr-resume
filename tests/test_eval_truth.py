"""Angka kebenaran dasar di eval/questions.yaml harus sama dengan hitungan Pandas langsung atas metadata."""
from collections import Counter

import pandas as pd
import pytest
import yaml

from src.config import META_CSV, ROOT

pytestmark = pytest.mark.skipif(not META_CSV.exists(), reason="resumes_meta.csv belum dibuat")


@pytest.fixture(scope="module")
def meta():
    return pd.read_csv(META_CSV, dtype={"resume_id": str})


@pytest.fixture(scope="module")
def truth():
    questions = yaml.safe_load((ROOT / "eval" / "questions.yaml").read_text(encoding="utf-8"))
    return {q["id"]: q["truth"]["must_contain"] for q in questions if "truth" in q}


def test_category_counts(meta, truth):
    counts = meta["category"].value_counts()
    assert str(counts["CHEF"]) in truth["stats-01"]
    assert str(counts["ACCOUNTANT"]) in truth["followup-03"]


def test_largest_category_tie(meta, truth):
    counts = meta["category"].value_counts()
    leaders = counts[counts == counts.max()].index
    assert str(counts.max()) in truth["stats-02"]
    assert {"INFORMATION-TECHNOLOGY", "BUSINESS-DEVELOPMENT"} == set(leaders)


def test_finance_over_five_years(meta, truth):
    finance = meta[meta["category"] == "FINANCE"]
    assert str(int((finance["years_experience"] > 5).sum())) in truth["stats-03"]


def test_top_it_skill(meta, truth):
    it = meta[meta["category"] == "INFORMATION-TECHNOLOGY"]["skills"].fillna("")
    counter = Counter(s for row in it for s in {x for x in row.split("; ") if x})
    assert counter.most_common(1)[0][0] in truth["stats-04"]
