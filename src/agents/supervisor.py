"""Supervisor: agent LangGraph yang memanggil Matcher dan Analyst sebagai tool.

System prompt "supervisor-system" diambil dari Langfuse setiap kali model dipanggil (src/prompts.py).
Riwayat dikirim dari luar (dipotong ke config.HISTORY_TURNS giliran); graf tidak menyimpan state.
"""
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from src.agents.analyst import PROMPT_NAME as ANALYST_PROMPT
from src.agents.analyst import build_analyst
from src.agents.matcher import PROMPT_NAME as MATCHER_PROMPT
from src.agents.matcher import build_matcher
from src.models import get_chat_model
from src.prompts import get_prompt, langfuse_system_prompt, prompt_metadata
from src.tools import context

PROMPT_NAME = "supervisor-system"
# Batas langkah graf sub-agent supaya loop tool yang tidak selesai berhenti dengan galat, bukan berputar terus
SUB_AGENT_RECURSION_LIMIT = 16


def _run_sub_agent(agent, prompt_name: str, name: str, task: str, config: RunnableConfig) -> str:
    context.add_tool_call(name, {"task": task}, agent="supervisor")
    # Config induk diteruskan supaya callback (token, Langfuse) ikut merekam panggilan di dalam sub-agent.
    # Metadata prompt supervisor yang diwarisi diganti dengan prompt sub-agent ini.
    sub_config = {
        **config,
        "metadata": {**(config.get("metadata") or {}), **prompt_metadata(get_prompt(prompt_name))},
        "recursion_limit": SUB_AGENT_RECURSION_LIMIT,
    }
    result = agent.invoke({"messages": [HumanMessage(task)]}, config=sub_config)
    answer = result["messages"][-1].content
    return answer or f"{name} tidak menghasilkan jawaban."


def build_supervisor():
    matcher = build_matcher()
    analyst = build_analyst()

    @tool("matcher")
    def matcher_tool(task: str, config: RunnableConfig) -> str:
        """Spesialis CV berbasis RAG di Qdrant: cari kandidat berdasarkan makna, cocokkan CV dengan deskripsi
        lowongan, ambil CV utuh dari ID, bandingkan 2 sampai 3 CV, cek skill yang dimiliki atau belum dimiliki
        satu kandidat (skill gap), dan susun riwayat kerja kronologis satu kandidat (timeline).

        Args:
            task: instruksi lengkap dan mandiri untuk spesialis. Sertakan kriteria pencarian, teks lowongan,
                daftar skill, dan ID CV dari percakapan bila ada, karena spesialis tidak melihat riwayat percakapan.
        """
        return _run_sub_agent(matcher, MATCHER_PROMPT, "matcher", task, config)

    @tool("analyst")
    def analyst_tool(task: str, config: RunnableConfig) -> str:
        """Spesialis statistik CV (Pandas): jumlah CV per kategori, saring berdasarkan perkiraan tahun
        pengalaman, level pendidikan, atau kata di jabatan, dan skill yang paling sering muncul per kategori.

        Args:
            task: pertanyaan lengkap dan mandiri untuk spesialis, termasuk kategori dan batas angka yang diminta.
        """
        return _run_sub_agent(analyst, ANALYST_PROMPT, "analyst", task, config)

    return create_agent(
        model=get_chat_model(),
        tools=[matcher_tool, analyst_tool],
        middleware=[langfuse_system_prompt(PROMPT_NAME)],
        name="supervisor",
    )


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
