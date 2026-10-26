"""Prompt wajib dari Langfuse: diuji dengan klien Langfuse tiruan (tanpa jaringan)."""
from types import SimpleNamespace

import pytest
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from src import prompts


class FakeLangfuse:
    def __init__(self, text="Kamu adalah supervisor.", version=3, fail=False):
        self.text, self.version, self.fail = text, version, fail
        self.calls = []

    def get_prompt(self, name, **kwargs):
        self.calls.append((name, kwargs))
        if self.fail:
            raise ConnectionError("jaringan putus")
        return SimpleNamespace(prompt=self.text, version=self.version)


def test_get_prompt_reads_production_label(monkeypatch):
    fake = FakeLangfuse()
    monkeypatch.setattr(prompts, "get_langfuse", lambda: fake)

    prompt = prompts.get_prompt("supervisor-system")

    assert (prompt.text, prompt.version) == ("Kamu adalah supervisor.", 3)
    name, kwargs = fake.calls[0]
    assert name == "supervisor-system" and kwargs["label"] == "production"
    assert prompts.prompt_metadata(prompt) == {"langfuse_prompt": prompt.client}


def test_missing_keys_raise_clear_error(monkeypatch):
    monkeypatch.setattr(prompts, "get_langfuse", lambda: None)

    with pytest.raises(prompts.PromptUnavailableError, match="Kunci Langfuse"):
        prompts.get_prompt("matcher-system")


def test_unreachable_langfuse_raises_instead_of_using_local_copy(monkeypatch):
    monkeypatch.setattr(prompts, "get_langfuse", lambda: FakeLangfuse(fail=True))

    with pytest.raises(prompts.PromptUnavailableError, match="matcher-system"):
        prompts.get_prompt("matcher-system")


class RecordingModel(BaseChatModel):
    """Model tiruan yang menyimpan pesan yang diterimanya."""

    seen: list = []

    @property
    def _llm_type(self) -> str:
        return "recording"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(messages)
        return ChatResult(generations=[ChatGeneration(message=AIMessage("ok"))])

    def bind_tools(self, tools, **kwargs):
        return self


def test_agent_uses_latest_langfuse_prompt_without_rebuild(monkeypatch):
    fake = FakeLangfuse(text="versi 1")
    monkeypatch.setattr(prompts, "get_langfuse", lambda: fake)
    model = RecordingModel(seen=[])
    agent = create_agent(model=model, tools=[], middleware=[prompts.langfuse_system_prompt("analyst-system")])

    agent.invoke({"messages": [HumanMessage("halo")]})
    fake.text = "versi 2"  # prompt diubah di Langfuse, agent tidak dibangun ulang
    agent.invoke({"messages": [HumanMessage("halo")]})

    system_texts = [m[0].content for m in model.seen]
    assert system_texts == ["versi 1", "versi 2"]
