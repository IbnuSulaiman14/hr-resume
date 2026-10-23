import pytest

from src import config


def test_get_setting_reads_environment(monkeypatch):
    monkeypatch.setenv("QDRANT_COLLECTION", "uji")
    assert config.get_setting("QDRANT_COLLECTION") == "uji"


def test_require_setting_raises_when_missing(monkeypatch):
    monkeypatch.delenv("SETELAN_TIDAK_ADA", raising=False)
    with pytest.raises(RuntimeError):
        config.require_setting("SETELAN_TIDAK_ADA")


def test_categories_match_dataset():
    assert len(config.CATEGORIES) == 24
    assert len(set(config.CATEGORIES)) == 24
