"""Aplikasi Streamlit Asisten Rekruter.

Jalankan: streamlit run app.py
Riwayat chat disimpan di st.session_state dan dikirim ulang (6 giliran terakhir) tiap pertanyaan.
"""
import json
import traceback
import uuid

import pandas as pd
import streamlit as st

from src.agents.supervisor import build_supervisor
from src.charts import build_chart, chart_table
from src.chat import ask
from src.config import HISTORY_TURNS
from src.prompts import PromptUnavailableError
from src.shortlist import add_ids, shortlist_table, to_csv_bytes
from src.tools.analyst_tools import load_meta

st.set_page_config(page_title="Asisten Rekruter", page_icon="🧑‍💼", layout="wide")

# Tiap pertanyaan berdiri sendiri (tanpa bergantung riwayat) dan semuanya bersama memicu kesembilan tool:
# search_resumes, match_job, category_distribution, filter_candidates, skill_frequency,
# compare_candidates, get_resume, skill_gap, dan candidate_timeline.
WELCOME_CARDS = [
    (
        ":material/search:", "Cari kandidat",
        "Cari kandidat chef yang punya pengalaman katering dan pengembangan menu.",
    ),
    (
        ":material/work:", "Cocokkan dengan lowongan",
        "Lowongan IT Support: menangani tiket helpdesk, instalasi perangkat, dan troubleshooting jaringan. "
        "Siapa yang cocok?",
    ),
    (
        ":material/bar_chart:", "Hitung statistik",
        "Kategori apa yang punya CV terbanyak, dan berapa CV di kategori Finance dengan perkiraan pengalaman "
        "di atas 5 tahun?",
    ),
    (
        ":material/psychology:", "Skill terbanyak",
        "Skill apa yang paling sering muncul di kategori Information-Technology?",
    ),
    (
        ":material/compare_arrows:", "Bandingkan kandidat",
        "Cari tiga kandidat Accountant dengan pengalaman audit, lalu bandingkan ketiganya.",
    ),
    (
        ":material/person_search:", "Telusuri satu kandidat",
        "Cari satu kandidat chef terbaik untuk pengembangan menu. Khusus untuk kandidat itu, tampilkan isi CV "
        "lengkapnya, cek apakah menu costing, inventory control, dan ServSafe terlihat di CV-nya, lalu susun "
        "riwayat kerjanya secara kronologis.",
    ),
]

ASSISTANT_AVATAR = ":material/support_agent:"
USER_AVATAR = ":material/person:"

STYLE = """
<style>
[data-testid="stMainBlockContainer"] { max-width: 62rem; padding-top: 2.5rem; }
[data-testid="stSidebar"] [data-testid="stMetricValue"] { font-size: 1.5rem; }
[data-testid="stChatMessage"] { padding-top: 0.75rem; padding-bottom: 0.75rem; }
/* Menu ⋮: buang Print dan Record screen (tidak berguna di aplikasi ini) beserta garis pemisahnya.
   Yang tersisa hanya pemilih tema. Penanda ini bagian dalam Streamlit; bila berubah di versi lain, item muncul lagi. */
[data-testid="stMainMenuItem-print"],
[data-testid="stMainMenuItem-recordScreencast"],
[data-testid="stMainMenuDivider"] { display: none; }
</style>
"""


@st.cache_resource(show_spinner="Menyiapkan agent...")
def load_agent():
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
    st.session_state.setdefault("shortlist", [])


def reset_chat() -> None:
    for key in ["messages", "usage_total", "session_id", "pending_question"]:
        st.session_state.pop(key, None)


def set_pending(question: str) -> None:
    st.session_state.pending_question = question


def add_to_shortlist(picker_key: str) -> None:
    st.session_state.shortlist = add_ids(st.session_state.shortlist, st.session_state.get(picker_key, []))
    st.session_state[picker_key] = []


def clear_shortlist() -> None:
    st.session_state.shortlist = []


def tool_badges(tool_calls: list[dict]) -> str:
    """Lencana nama tool: ungu untuk rute supervisor (matcher, analyst), biru untuk tool spesialis."""
    badges = {}
    for call in tool_calls:
        color = "violet" if call.get("agent") == "supervisor" else "blue"
        name = call["name"].replace("_", "\\_")  # underscore bisa dibaca markdown sebagai miring
        badges[call["name"]] = f":{color}-badge[{name}]"
    return " ".join(badges.values())


def render_charts(charts: list[dict]) -> None:
    """Grafik dari data yang dihitung tool Analyst, plus tabel datanya."""
    for spec in charts:
        st.altair_chart(build_chart(spec), width="stretch")
        with st.expander(f"Data grafik: {spec['title']}"):
            st.dataframe(chart_table(spec), hide_index=True, width="stretch")


def render_sources(sources: list[dict], key: str) -> None:
    table = pd.DataFrame(sources).reindex(columns=["resume_id", "category", "section", "score", "rerank_score", "snippet"])
    if table["rerank_score"].isna().all():
        table = table.drop(columns="rerank_score")  # reranker tidak dipakai atau gagal untuk jawaban ini
    score = st.column_config.ProgressColumn
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={
            "resume_id": st.column_config.TextColumn("ID CV"),
            "category": "Kategori",
            "section": "Seksi",
            "score": score("Skor vektor", min_value=0.0, max_value=1.0, format="%.3f"),
            "rerank_score": score("Skor rerank", min_value=0.0, max_value=1.0, format="%.3f"),
            "snippet": st.column_config.TextColumn("Cuplikan", width="large"),
        },
    )
    candidate_ids = list(dict.fromkeys(str(s["resume_id"]) for s in sources))
    picker_key = f"pick_{key}"
    pick, add = st.columns([4, 1], vertical_alignment="bottom")
    pick.multiselect("Pilih kandidat untuk shortlist", candidate_ids, key=picker_key)
    add.button("Tambah ke shortlist", key=f"add_{key}", on_click=add_to_shortlist, args=(picker_key,), width="stretch")


def render_details(details: dict, key: str, latest: bool = False) -> None:
    """Grafik (bila ada), lencana tool, lalu expander sumber RAG (terbuka untuk jawaban terbaru), tool call,
    riwayat, dan baris token."""
    if details.get("charts"):
        render_charts(details["charts"])

    tool_calls = details.get("tool_calls", [])
    if tool_calls:
        st.markdown(tool_badges(tool_calls))

    sources = details.get("sources", [])
    with st.expander(f"Sumber RAG ({len(sources)})", expanded=latest and bool(sources)):
        if not sources:
            st.caption("Jawaban ini tidak memakai sumber CV.")
        else:
            render_sources(sources, key)

    with st.expander(f"Tool calls ({len(tool_calls)})"):
        if not tool_calls:
            st.caption("Tidak ada tool yang dipanggil.")
        for i, call in enumerate(tool_calls, start=1):
            st.markdown(f"**{i}. `{call['name']}`** · dipanggil oleh *{call.get('agent') or '-'}*")
            st.code(json.dumps(call["args"], ensure_ascii=False, indent=2), language="json")

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
        rerank = f" · {usage['rerank_calls']} panggilan reranker" if usage.get("rerank_calls") else ""
        st.caption(
            f"Token masuk {usage['input_tokens']:,} · keluar {usage['output_tokens']:,} · "
            f"total {usage['total_tokens']:,} · {usage['llm_calls']} panggilan LLM{rerank} · "
            f"{details.get('seconds', 0)} detik{cost}"
        )


def sidebar_shortlist() -> None:
    shortlist = st.session_state.shortlist
    st.subheader(f"Shortlist ({len(shortlist)})")
    if not shortlist:
        st.caption("Pilih kandidat dari panel Sumber RAG di bawah jawaban untuk menambahkannya.")
        return
    try:
        table = shortlist_table(shortlist)
    except Exception:
        st.warning("Metadata CV belum tersedia.")
        return
    st.dataframe(table[["ID CV", "Kategori", "Judul CV"]], hide_index=True, width="stretch")
    st.download_button(
        "Unduh shortlist (CSV)", to_csv_bytes(table), file_name="shortlist_kandidat.csv", mime="text/csv",
        icon=":material/download:", width="stretch",
    )
    st.button("Kosongkan shortlist", on_click=clear_shortlist, icon=":material/delete:", width="stretch")


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

        sidebar_shortlist()

        st.subheader("Penggunaan sesi")
        total = st.session_state.usage_total
        col1, col2 = st.columns(2)
        col1.metric("Pertanyaan", total["questions"])
        col2.metric("Token", f"{total['input_tokens'] + total['output_tokens']:,}")
        st.caption(
            f"Masuk {total['input_tokens']:,} · keluar {total['output_tokens']:,} · "
            f"perkiraan biaya US${total['cost_usd']:.4f}"
        )
        st.button("Reset percakapan", on_click=reset_chat, icon=":material/restart_alt:", width="stretch")

        with st.expander("Batasan dan etika"):
            st.caption(
                "Alat bantu penyaringan awal. Aplikasi tidak memutuskan siapa yang diterima dan tidak menyaring "
                "berdasarkan atribut pribadi. Data tidak memuat nama kandidat, nama perusahaan asli, dan gaji. "
                "Tahun pengalaman dan level pendidikan adalah perkiraan hasil ekstraksi."
            )

        # Streamlit belum punya API untuk mengganti tema per pengguna dari Python; pemilihnya ada di menu bawaan
        st.caption(":material/contrast: Mode terang atau gelap: menu **⋮** di kanan atas, bagian **Theme**.")


def render_welcome() -> None:
    """Layar awal: penjelasan singkat dan kartu pertanyaan yang bisa langsung diklik."""
    st.title("Tanya tentang kandidat")
    st.markdown(
        "Cari kandidat, cocokkan dengan lowongan, minta statistik, atau cek skill yang kurang. "
        "Setiap jawaban menyertakan **ID CV** dan **sumbernya**, jadi bisa dicek ke CV aslinya."
    )
    columns = st.columns(2)
    for i, (icon, title, question) in enumerate(WELCOME_CARDS):
        with columns[i % 2].container(border=True):
            st.markdown(f"**{icon} {title}**")
            st.caption(question)
            st.button("Tanyakan", key=f"welcome_{i}", on_click=set_pending, args=(question,))
    st.caption("Atau ketik pertanyaan sendiri di kolom bawah, termasuk tempel deskripsi lowongan lengkap.")


def answer_question(question: str) -> None:
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(question)

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        error = None
        with st.status("Supervisor memilih spesialis dan mencari di CV...", expanded=False) as status:
            try:
                result = ask(question, history, session_id=st.session_state.session_id, agent=load_agent())
            except PromptUnavailableError:
                print(traceback.format_exc())
                error = "Prompt agent tidak bisa diambil dari Langfuse. Periksa kunci dan koneksi Langfuse, lalu coba lagi."
            except Exception:
                print(traceback.format_exc())
                error = "Maaf, terjadi galat saat memproses pertanyaan. Coba lagi sebentar lagi atau ubah pertanyaannya."
            if error:
                status.update(label="Gagal memproses pertanyaan", state="error")
            else:
                status.update(label=f"Selesai dalam {result.seconds} detik", state="complete")
        if error:
            st.error(error)
            st.session_state.messages.append({"role": "assistant", "content": error, "details": {}, "error": True})
            return
        details = {
            "tool_calls": result.tool_calls,
            "sources": result.sources,
            "usage": result.usage,
            "charts": result.charts,
            "seconds": result.seconds,
            "history_sent": history[-HISTORY_TURNS * 2 :],
        }
        st.markdown(result.answer)
        render_details(details, key=str(len(st.session_state.messages)), latest=True)

    st.session_state.messages.append({"role": "assistant", "content": result.answer, "details": details})
    total = st.session_state.usage_total
    total["input_tokens"] += result.usage.get("input_tokens", 0)
    total["output_tokens"] += result.usage.get("output_tokens", 0)
    total["cost_usd"] += result.usage.get("cost_usd") or 0.0
    total["questions"] += 1


def main() -> None:
    st.markdown(STYLE, unsafe_allow_html=True)
    init_state()
    sidebar()

    if not st.session_state.messages and not st.session_state.pending_question:
        render_welcome()

    last = len(st.session_state.messages) - 1
    for index, message in enumerate(st.session_state.messages):
        avatar = ASSISTANT_AVATAR if message["role"] == "assistant" else USER_AVATAR
        with st.chat_message(message["role"], avatar=avatar):
            if message.get("error"):
                st.error(message["content"])
            else:
                st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("details"):
                render_details(message["details"], key=str(index), latest=index == last)

    typed = st.chat_input("Tulis pertanyaan tentang kandidat...")
    question = typed or st.session_state.pending_question
    if question:
        st.session_state.pending_question = None
        answer_question(question)
        st.rerun()  # perbarui sidebar (penggunaan sesi) dan render ulang jawaban dari riwayat


main()
