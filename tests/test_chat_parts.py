from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from src.agents.supervisor import trim_history
from src.chat import dedupe_sources
from src.usage import UsageTracker


def _turns(n):
    messages = []
    for i in range(n):
        messages += [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}]
    return messages


def test_trim_history_keeps_last_turns():
    trimmed = trim_history(_turns(8), turns=6)

    assert len(trimmed) == 12
    assert isinstance(trimmed[0], HumanMessage) and trimmed[0].content == "q2"
    assert isinstance(trimmed[-1], AIMessage) and trimmed[-1].content == "a7"


def test_trim_history_starts_with_user_message():
    messages = [{"role": "assistant", "content": "halo"}] + _turns(1)

    assert [m.content for m in trim_history(messages, turns=6)] == ["q0", "a0"]


def test_trim_history_empty():
    assert trim_history([], turns=6) == []


def _llm_result(input_tokens, output_tokens):
    message = AIMessage("x", usage_metadata={"input_tokens": input_tokens, "output_tokens": output_tokens,
                                             "total_tokens": input_tokens + output_tokens})
    return LLMResult(generations=[[ChatGeneration(message=message)]])


def test_usage_tracker_sums_all_calls():
    tracker = UsageTracker()
    tracker.on_llm_end(_llm_result(100, 20))
    tracker.on_llm_end(_llm_result(50, 5))

    assert (tracker.input_tokens, tracker.output_tokens, tracker.llm_calls) == (150, 25, 2)
    assert tracker.cost_usd("deepseek/deepseek-v4-flash-0731") > 0
    assert tracker.cost_usd("model-tanpa-harga") is None


def test_usage_tracker_falls_back_to_llm_output():
    tracker = UsageTracker()
    tracker.on_llm_end(LLMResult(generations=[[]], llm_output={"token_usage": {"prompt_tokens": 7, "completion_tokens": 3}}))

    assert tracker.total_tokens == 10


def test_dedupe_sources_keeps_best_score_in_first_position():
    sources = [
        {"resume_id": "1", "section": "skills", "score": 0.5},
        {"resume_id": "2", "section": "skills", "score": 0.7},
        {"resume_id": "1", "section": "skills", "score": 0.9},
    ]

    assert dedupe_sources(sources) == [
        {"resume_id": "1", "section": "skills", "score": 0.9},
        {"resume_id": "2", "section": "skills", "score": 0.7},
    ]
