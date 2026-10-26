"""Sub-agent Analyst: agent ReAct (LangGraph) dengan tool Pandas.

System prompt "analyst-system" diambil dari Langfuse setiap kali model dipanggil (src/prompts.py).
"""
from langchain.agents import create_agent

from src.models import get_chat_model
from src.prompts import langfuse_system_prompt
from src.tools.analyst_tools import ANALYST_TOOLS

PROMPT_NAME = "analyst-system"


def build_analyst():
    return create_agent(
        model=get_chat_model(),
        tools=ANALYST_TOOLS,
        middleware=[langfuse_system_prompt(PROMPT_NAME)],
        name="analyst",
    )
