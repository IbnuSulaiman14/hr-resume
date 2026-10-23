"""Aplikasi Streamlit Asisten Rekruter. Tahap B4.

Jalankan: streamlit run app.py
Riwayat chat disimpan di st.session_state dan dikirim ulang (6 giliran terakhir) tiap pertanyaan.
"""
import json
import traceback
import uuid

import pandas as pd
import streamlit as st

from src.agents.supervisor import build_supervisor
from src.chat import ask
from src.config import HISTORY_TURNS
from src.tools.analyst_tools import load_meta

st.set_page_config(page_title="Asisten Rekruter", page_icon="🧑‍💼", layout="wide")

EXAMPLE_QUESTIONS = [
    "Cari kandidat chef yang punya pengalaman katering dan pengembangan menu.",
    "Lowongan IT Support: menangani tiket helpdesk, instalasi perangkat, dan troubleshooting jaringan. Siapa yang cocok?",
    "Berapa jumlah CV per kategori?",
    "Berapa CV di kategori Finance dengan perkiraan pengalaman di atas 5 tahun?",
    "Skill apa yang paling sering muncul di kategori Information-Technology?",
    "Bandingkan dua kandidat teratas dari jawaban sebelumnya.",
]


@st.cache_resource(show_spinner="Menyiapkan agent...")
def load_agent():
    # Graf supervisor dan sub-agent dibuat sekali per proses, bukan setiap pesan
    return build_supervisor()


@st.cache_data
def dataset_info() -> tuple[int, int]:
    meta = load_meta()
    return len(meta), meta["category"].nunique()


def init_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("session_id", f"st-{uuid.uuid4().hex[:8]}")
    st.session_state.setdefault("usage_total", {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "questions": 0})
    st.session_state.setdefault("pending_question", None)


def reset_chat() -> None:
    for key in ["messages", "usage_total", "session_id", "pending_question"]:
        st.session_state.pop(key, None)


def render_details(details: dict) -> None:
    """Expander tool call, sumber RAG, riwayat, dan baris token di bawah satu jawaban."""
    tool_calls = details.get("tool_calls", [])
    with st.expander(f"Tool calls ({len(tool_calls)})"):
        if not tool_calls:
            st.caption("Tidak ada tool yang dipanggil.")
        for i, call in enumerate(tool_calls, start=1):
            st.markdown(f"**{i}. `{call['name']}`** · dipanggil oleh *{call.get('agent') or '-'}*")
            st.code(json.dumps(call["args"], ensure_ascii=False, indent=2), language="json")

    sources = details.get("sources", [])
    with st.expander(f"Sumber RAG ({len(sources)})"):
        if not sources:
            st.caption("Jawaban ini tidak memakai sumber CV.")
        else:
            table = pd.DataFrame(sources).reindex(columns=["resume_id", "category", "section", "score", "snippet"])
            st.dataframe(
                table,
                hide_index=True,
                width="stretch",
                column_config={
                    "resume_id": st.column_config.TextColumn("ID CV"),
                    "category": "Kategori",
                    "section": "Seksi",
                    "score": st.column_config.NumberColumn("Skor", format="%.3f"),
                    "snippet": st.column_config.TextColumn("Cuplikan", width="large"),
                },
            )

    history = details.get("history_sent", [])
    with st.expander(f"Riwayat yang dikirim ke agent ({len(history) // 2} giliran)"):
        if not history:
            st.caption("Pertanyaan pertama, belum ada riwayat.")
        for message in history:
            who = "Pengguna" if message["role"] == "user" else "Asisten"
            st.markdown(f"**{who}:** {message['content'][:300]}{'...' if len(message['content']) > 300 else ''}")

    usage = details.get("usage", {})
    if usage:
        cost = f" · ~US${usage['cost_usd']:.5f}" if usage.get("cost_usd") is not None else ""
        st.caption(
            f"Token masuk {usage['input_tokens']:,} · keluar {usage['output_tokens']:,} · "
            f"total {usage['total_tokens']:,} · {usage['llm_calls']} panggilan LLM · "
            f"{details.get('seconds', 0)} detik{cost}"
        )


def sidebar() -> None:
    with st.sidebar:
        st.title("🧑‍💼 Asisten Rekruter")
        st.caption("Chatbot RAG multi-agent untuk mencari, mencocokkan, dan menganalisis CV kandidat.")
        try:
            total_cv, total_categories = dataset_info()
            col1, col2 = st.columns(2)
            col1.metric("CV", f"{total_cv:,}")
            col2.metric("Kategori", total_categories)
        except Exception:
            st.warning("Metadata CV belum tersedia.")

        st.subheader("Contoh pertanyaan")
        for question in EXAMPLE_QUESTIONS:
            if st.button(question, width="stretch"):
                st.session_state.pending_question = question

        st.subheader("Penggunaan sesi")
        total = st.session_state.usage_total
        st.markdown(
            f"- Pertanyaan: **{total['questions']}**\n"
            f"- Token masuk: **{total['input_tokens']:,}**\n"
            f"- Token keluar: **{total['output_tokens']:,}**\n"
            f"- Perkiraan biaya: **US${total['cost_usd']:.4f}**"
        )
        st.button("Reset percakapan", on_click=reset_chat, width="stretch")

        st.divider()
        st.caption(
            "Alat bantu penyaringan awal. Aplikasi tidak memutuskan siapa yang diterima dan tidak menyaring "
            "berdasarkan atribut pribadi. Data tidak memuat nama kandidat, nama perusahaan asli, dan gaji. "
            "Tahun pengalaman dan level pendidikan adalah perkiraan hasil ekstraksi."
        )


def answer_question(question: str) -> None:
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Mencari di CV kandidat..."):
            try:
                result = ask(question, history, session_id=st.session_state.session_id, agent=load_agent())
            except Exception:
                # Detail galat hanya ke log server; pengguna mendapat pesan ramah
                print(traceback.format_exc())
                result = None
        if result is None:
            message = "Maaf, terjadi galat saat memproses pertanyaan. Coba lagi sebentar lagi atau ubah pertanyaannya."
            st.error(message)
            st.session_state.messages.append({"role": "assistant", "content": message, "details": {}, "error": True})
            return
        details = {
            "tool_calls": result.tool_calls,
            "sources": result.sources,
            "usage": result.usage,
            "seconds": result.seconds,
            "history_sent": history[-HISTORY_TURNS * 2 :],
        }
        st.markdown(result.answer)
        render_details(details)

    st.session_state.messages.append({"role": "assistant", "content": result.answer, "details": details})
    total = st.session_state.usage_total
    total["input_tokens"] += result.usage.get("input_tokens", 0)
    total["output_tokens"] += result.usage.get("output_tokens", 0)
    total["cost_usd"] += result.usage.get("cost_usd") or 0.0
    total["questions"] += 1


def main() -> None:
    init_state()
    sidebar()
    st.header("Tanya tentang kandidat")
    st.caption("Contoh: cari kandidat, tempel deskripsi lowongan, minta statistik, atau tanya lanjutan seperti \"yang kedua\".")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message.get("error"):
                st.error(message["content"])
            else:
                st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("details"):
                render_details(message["details"])

    typed = st.chat_input("Tulis pertanyaan tentang kandidat...")
    question = typed or st.session_state.pending_question
    if question:
        st.session_state.pending_question = None
        answer_question(question)
        st.rerun()  # perbarui angka penggunaan sesi di sidebar


main()
