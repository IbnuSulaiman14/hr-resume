"""Sub-agent Matcher: agent ReAct (LangGraph) dengan tool RAG. Tahap B3.2.

Prompt: prompts.get_prompt("matcher-system").
"""
from langchain.agents import create_agent

from src.models import get_chat_model
from src.prompts import Prompt, get_prompt
from src.tools.matcher_tools import MATCHER_TOOLS


def build_matcher(prompt: Prompt | None = None):
    prompt = prompt or get_prompt("matcher-system")
    return create_agent(model=get_chat_model(), tools=MATCHER_TOOLS, system_prompt=prompt.text, name="matcher")
