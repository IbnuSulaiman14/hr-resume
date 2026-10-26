from types import SimpleNamespace

from src import rerank
from src.chat import combine_usage
from src.tools import context


def test_apply_rerank_sorts_by_rerank_score_and_keeps_vector_score():
    items = [{"id": "a", "score": 0.9}, {"id": "b", "score": 0.8}, {"id": "c", "score": 0.7}]

    result = rerank.apply_rerank(items, [0.1, 0.95, 0.5], top_k=2)

    assert [r["id"] for r in result] == ["b", "c"]
    assert result[0] == {"id": "b", "score": 0.8, "rerank_score": 0.95}


def test_apply_rerank_without_scores_keeps_vector_order():
    items = [{"id": "a"}, {"id": "b"}, {"id": "c"}]

    assert rerank.apply_rerank(items, None, top_k=2) == [{"id": "a"}, {"id": "b"}]


def test_rerank_off_returns_none(monkeypatch):
    monkeypatch.setenv("RERANK_MODEL", "off")

    assert rerank.rerank_model() is None
    assert rerank.rerank_scores("chef", ["a chef"]) is None


def test_rerank_scores_maps_indices_and_records_cost(monkeypatch):
    monkeypatch.setenv("RERANK_MODEL", "qwen/qwen3-reranker-8b")
    monkeypatch.setenv("OPENROUTER_API_KEY", "kunci-tiruan")
    sent = {}

    def fake_post(url, headers, json, timeout):
        sent.update(url=url, json=json)
        body = {"results": [{"index": 1, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.2}],
                "usage": {"total_tokens": 120, "cost": 0.00002}}
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: body)

    monkeypatch.setattr(rerank.httpx, "post", fake_post)
    context.start_run()

    scores = rerank.rerank_scores("chef catering", ["line cook", "banquet chef"])

    assert scores == [0.2, 0.9]
    assert sent["url"].endswith("/rerank") and sent["json"]["model"] == "qwen/qwen3-reranker-8b"
    assert context.get_usage() == [{"kind": "rerank", "tokens": 120, "cost_usd": 0.00002}]


def test_rerank_failure_falls_back_to_none(monkeypatch):
    monkeypatch.setenv("RERANK_MODEL", "qwen/qwen3-reranker-8b")
    monkeypatch.setenv("OPENROUTER_API_KEY", "kunci-tiruan")

    def broken_post(*args, **kwargs):
        raise TimeoutError("lambat")

    monkeypatch.setattr(rerank.httpx, "post", broken_post)

    assert rerank.rerank_scores("chef", ["a chef"]) is None


def test_combine_usage_adds_rerank_cost():
    llm = {"input_tokens": 100, "output_tokens": 10, "total_tokens": 110, "llm_calls": 2, "cost_usd": 0.001}
    extra = [{"kind": "rerank", "tokens": 300, "cost_usd": 0.0001}]

    usage = combine_usage(llm, extra)

    assert usage["rerank_calls"] == 1 and usage["rerank_tokens"] == 300
    assert usage["cost_usd"] == 0.0011
