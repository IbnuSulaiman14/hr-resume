from src.tools import context


def test_collects_sources_and_tool_calls_per_run():
    context.start_run()
    context.add_tool_call("search_resumes", {"query": "python"}, agent="matcher")
    context.add_sources([{"resume_id": "1", "category": "CHEF", "section": "skills", "score": 0.8, "snippet": "x"}])

    assert context.get_tool_calls() == [{"name": "search_resumes", "args": {"query": "python"}, "agent": "matcher"}]
    assert [s["resume_id"] for s in context.get_sources()] == ["1"]


def test_start_run_clears_previous_run():
    context.start_run()
    context.add_sources([{"resume_id": "1"}])
    context.start_run()

    assert context.get_sources() == []
    assert context.get_tool_calls() == []


def test_add_without_start_run_is_ignored():
    context._sources.set(None)
    context.add_sources([{"resume_id": "1"}])

    assert context.get_sources() == []
