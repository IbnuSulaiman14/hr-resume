"""Penghitung token semua panggilan LLM dalam satu permintaan (supervisor + sub-agent). Tahap B3.4."""


class UsageTracker:
    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens
