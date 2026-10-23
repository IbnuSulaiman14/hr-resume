"""Sub-agent Analyst: agent ReAct (LangGraph) dengan tool Pandas. Tahap B3.2.

Prompt: prompts.get_prompt("analyst-system").
"""
from langchain.agents import create_agent

from src.models import get_chat_model
from src.prompts import Prompt, get_prompt
from src.tools.analyst_tools import ANALYST_TOOLS


def build_analyst(prompt: Prompt | None = None):
    prompt = prompt or get_prompt("analyst-system")
    return create_agent(model=get_chat_model(), tools=ANALYST_TOOLS, system_prompt=prompt.text, name="analyst")
