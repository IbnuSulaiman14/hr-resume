"""Sub-agent Matcher: agent ReAct (LangGraph) dengan tool RAG.

System prompt "matcher-system" diambil dari Langfuse setiap kali model dipanggil (src/prompts.py).
"""
from langchain.agents import create_agent

from src.models import get_chat_model
from src.prompts import langfuse_system_prompt
from src.tools.matcher_tools import MATCHER_TOOLS

PROMPT_NAME = "matcher-system"


def build_matcher():
    return create_agent(
        model=get_chat_model(),
        tools=MATCHER_TOOLS,
        middleware=[langfuse_system_prompt(PROMPT_NAME)],
        name="matcher",
    )
