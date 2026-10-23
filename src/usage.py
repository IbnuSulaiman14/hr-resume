"""Penghitung token semua panggilan LLM dalam satu permintaan (supervisor + sub-agent). Tahap B3.4.

Dipasang sebagai callback LangChain. Callback ikut turun ke sub-agent karena config diteruskan,
jadi satu tracker menjumlahkan token dari setiap panggilan model dalam satu pertanyaan.
"""
import threading

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

# Harga OpenRouter per 1 juta token (US$) untuk estimasi biaya; diperbarui bila model diganti
PRICES_PER_MILLION = {"deepseek/deepseek-v4-flash-0731": (0.044, 0.133)}


class UsageTracker(BaseCallbackHandler):
    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.llm_calls = 0
        self._lock = threading.Lock()  # sub-agent bisa berjalan paralel di thread lain

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        input_tokens = output_tokens = 0
        for generations in response.generations:
            for generation in generations:
                usage = getattr(getattr(generation, "message", None), "usage_metadata", None)
                if usage:
                    input_tokens += usage.get("input_tokens", 0)
                    output_tokens += usage.get("output_tokens", 0)
        if not (input_tokens or output_tokens):
            # Cadangan untuk penyedia yang hanya mengisi llm_output
            token_usage = (response.llm_output or {}).get("token_usage") or {}
            input_tokens = token_usage.get("prompt_tokens", 0)
            output_tokens = token_usage.get("completion_tokens", 0)
        with self._lock:
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            self.llm_calls += 1

    def cost_usd(self, model: str) -> float | None:
        price = PRICES_PER_MILLION.get(model)
        if price is None:
            return None
        return (self.input_tokens * price[0] + self.output_tokens * price[1]) / 1_000_000

    def as_dict(self, model: str | None = None) -> dict:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "llm_calls": self.llm_calls,
            "cost_usd": self.cost_usd(model) if model else None,
        }
