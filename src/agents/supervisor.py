"""Supervisor: agent LangGraph yang memanggil Matcher dan Analyst sebagai tool. Tahap B3.3.

Prompt: prompts.get_prompt("supervisor-system"). Riwayat dikirim dari luar (dipotong ke
config.HISTORY_TURNS giliran); graf tidak menyimpan state.
"""
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from src.agents.analyst import build_analyst
from src.agents.matcher import build_matcher
from src.models import get_chat_model
from src.prompts import Prompt, get_prompt
from src.tools import context

# Batas langkah graf sub-agent supaya loop tool yang tidak selesai berhenti dengan galat, bukan berputar terus
SUB_AGENT_RECURSION_LIMIT = 16


def prompt_metadata(prompt: Prompt) -> dict:
    """Metadata yang membuat Langfuse menautkan panggilan LLM di dalam graf ke versi prompt-nya."""
    return {"langfuse_prompt": prompt.client} if prompt.client else {}


def _run_sub_agent(agent, prompt: Prompt, name: str, task: str, config: RunnableConfig) -> str:
    context.add_tool_call(name, {"task": task}, agent="supervisor")
    # Config induk diteruskan supaya callback (token, Langfuse) ikut merekam panggilan di dalam sub-agent.
    # Metadata prompt supervisor yang diwarisi diganti dengan prompt sub-agent ini.
    sub_config = {
        **config,
        "metadata": {**(config.get("metadata") or {}), **prompt_metadata(prompt)},
        "recursion_limit": SUB_AGENT_RECURSION_LIMIT,
    }
    result = agent.invoke({"messages": [HumanMessage(task)]}, config=sub_config)
    answer = result["messages"][-1].content
    return answer or f"{name} tidak menghasilkan jawaban."


def build_supervisor():
    """Kembalikan graf supervisor dan prompt-nya (untuk metadata trace)."""
    matcher_prompt = get_prompt("matcher-system")
    analyst_prompt = get_prompt("analyst-system")
    matcher = build_matcher(matcher_prompt)
    analyst = build_analyst(analyst_prompt)

    @tool("matcher")
    def matcher_tool(task: str, config: RunnableConfig) -> str:
        """Spesialis pencarian CV (RAG di Qdrant): cari kandidat berdasarkan makna, cocokkan CV dengan
        deskripsi lowongan, ambil CV utuh dari ID, atau bandingkan 2 sampai 3 CV.

        Args:
            task: instruksi lengkap dan mandiri untuk spesialis. Sertakan kriteria pencarian, teks lowongan,
                dan ID CV dari percakapan bila ada, karena spesialis tidak melihat riwayat percakapan.
        """
        return _run_sub_agent(matcher, matcher_prompt, "matcher", task, config)

    @tool("analyst")
    def analyst_tool(task: str, config: RunnableConfig) -> str:
        """Spesialis statistik CV (Pandas): jumlah CV per kategori, saring berdasarkan perkiraan tahun
        pengalaman, level pendidikan, atau kata di jabatan, dan skill yang paling sering muncul per kategori.

        Args:
            task: pertanyaan lengkap dan mandiri untuk spesialis, termasuk kategori dan batas angka yang diminta.
        """
        return _run_sub_agent(analyst, analyst_prompt, "analyst", task, config)

    prompt = get_prompt("supervisor-system")
    graph = create_agent(
        model=get_chat_model(), tools=[matcher_tool, analyst_tool], system_prompt=prompt.text, name="supervisor"
    )
    return graph, prompt


def trim_history(messages: list, turns: int) -> list:
    """Ambil `turns` giliran terakhir (satu giliran = pesan pengguna + jawaban).

    `messages` berisi dict {"role": "user" | "assistant", "content": str}. Hasilnya pesan LangChain
    yang selalu diawali pesan pengguna."""
    recent = messages[-turns * 2 :] if turns > 0 else []
    while recent and recent[0]["role"] != "user":
        recent = recent[1:]
    return [
        HumanMessage(m["content"]) if m["role"] == "user" else AIMessage(m["content"])
        for m in recent
        if m.get("content")
    ]
