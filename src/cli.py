"""Uji agent dari terminal: python -m src.cli "pertanyaan" ["pertanyaan lanjutan" ...].

Mencetak jawaban, tool call, sumber, dan token tanpa Streamlit. Beberapa pertanyaan sekaligus
dijalankan berurutan dalam satu percakapan, untuk menguji riwayat.
"""
import sys
import uuid

from src.chat import ask
from src.observability import flush


def main() -> None:
    questions = sys.argv[1:]
    if not questions:
        print('Pakai: python -m src.cli "pertanyaan" ["pertanyaan lanjutan" ...]')
        return
    sys.stdout.reconfigure(encoding="utf-8")
    history: list[dict] = []
    session_id = f"cli-{uuid.uuid4().hex[:8]}"
    for question in questions:
        result = ask(question, history, session_id=session_id)
        print(f"\n=== Pertanyaan: {question}\n")
        print(result.answer)
        print("\n--- Tool calls")
        for call in result.tool_calls:
            print(f"  [{call['agent']}] {call['name']} {call['args']}")
        print(f"--- Sumber ({len(result.sources)})")
        for source in result.sources[:10]:
            score = f"{source['score']:.3f}" if source.get("score") is not None else "-"
            print(f"  {source['resume_id']} {source['category']} {source['section']} skor {score}")
        usage = result.usage
        print(
            f"--- Token: masuk {usage['input_tokens']}, keluar {usage['output_tokens']}, "
            f"{usage['llm_calls']} panggilan LLM, {result.seconds} detik, trace {result.trace_id}"
        )
        history += [{"role": "user", "content": question}, {"role": "assistant", "content": result.answer}]
    flush()


if __name__ == "__main__":
    main()
