"""Supervisor: agent LangGraph yang memanggil Matcher dan Analyst sebagai tool. Tahap B3.3.

Prompt: prompts.get_prompt("supervisor-system"). Riwayat dikirim dari luar (dipotong ke
config.HISTORY_TURNS giliran); graf tidak menyimpan state.
"""


def build_supervisor():
    raise NotImplementedError("B3.3")


def trim_history(messages: list, turns: int) -> list:
    """Ambil `turns` giliran terakhir (satu giliran = pesan pengguna + jawaban)."""
    raise NotImplementedError("B3.4")
