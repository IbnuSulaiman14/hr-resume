"""Jalankan: python -m eval.run_eval [--only id1,id2]. Tahap B5.2.

Membaca eval/questions.yaml, menjalankan tiap pertanyaan lewat supervisor, lalu menghitung:
ketepatan rute, keterlacakan (ID di jawaban ada di sumber), penolakan benar, dan kesamaan angka
dengan hitungan Pandas. Menulis laporan ke eval/report.md.
"""
import argparse
import re
import statistics
import sys
import uuid
from datetime import datetime

import yaml

from src.chat import ChatResult, ask
from src.config import ROOT
from src.observability import flush

QUESTIONS = ROOT / "eval" / "questions.yaml"
REPORT = ROOT / "eval" / "report.md"
RESUME_ID = re.compile(r"\b\d{7,8}\b")


def observed_route(result: ChatResult) -> str:
    agents = {c["name"] for c in result.tool_calls if c.get("agent") == "supervisor"}
    if not agents:
        return "none"
    if agents == {"matcher", "analyst"}:
        return "both"
    return agents.pop()


def ids_in(text: str) -> list[str]:
    return list(dict.fromkeys(RESUME_ID.findall(text)))


def traceable(result: ChatResult) -> bool | None:
    """Semua ID CV yang disebut di jawaban harus ada di daftar sumber. None bila jawaban tidak menyebut ID."""
    mentioned = set(ids_in(result.answer))
    if not mentioned:
        return None
    return mentioned <= {s["resume_id"] for s in result.sources}


def evaluate(question: dict, result: ChatResult, previous: ChatResult | None) -> dict:
    tools_called = [c["name"] for c in result.tool_calls if c.get("agent") != "supervisor"]
    route = observed_route(result)
    checks: dict[str, bool | None] = {
        "route": route == question["route"],
        "tools": set(question.get("tools", [])) <= set(tools_called) if question.get("tools") else None,
        "traceable": traceable(result),
    }
    answer = result.answer.lower()
    expect = question.get("expect", {})
    if "category_in_sources" in expect:
        categories = {s["category"] for s in result.sources}
        checks["category"] = bool(categories & set(expect["category_in_sources"])) if categories else False
    if "truth" in question:
        checks["numbers"] = all(token.lower() in answer for token in question["truth"]["must_contain"])
    if "refuse_or_state_limit" in expect:
        checks["refusal"] = route == "none" and any(k in answer for k in expect.get("any_of", []))
    if previous is not None:
        previous_ids = ids_in(previous.answer)
        called_ids = [
            str(i)
            for c in result.tool_calls
            for i in ([c["args"].get("resume_id")] if "resume_id" in c["args"] else c["args"].get("resume_ids", []))
        ]
        if "resume_id_matches_turn_result" in expect:
            position = expect["resume_id_matches_turn_result"]
            checks["followup"] = len(previous_ids) >= position and previous_ids[position - 1] in called_ids
        if expect.get("resume_ids_from_previous_turn"):
            checks["followup"] = bool(called_ids) and set(called_ids) <= set(previous_ids[:3])
    return {
        "id": question["id"],
        "group": question["group"],
        "route_expected": question["route"],
        "route_observed": route,
        "tools_called": tools_called,
        "checks": checks,
        "passed": all(v for v in checks.values() if v is not None),
        "seconds": result.seconds,
        "tokens": result.usage.get("total_tokens", 0),
        "cost_usd": result.usage.get("cost_usd") or 0.0,
        "answer": result.answer,
    }


def run(question: dict) -> tuple[ChatResult, ChatResult | None]:
    session_id = f"eval-{question['id']}-{uuid.uuid4().hex[:6]}"
    history: list[dict] = []
    previous = None
    for turn in question.get("turns", []):
        previous = ask(turn, history, session_id=session_id)
        history += [{"role": "user", "content": turn}, {"role": "assistant", "content": previous.answer}]
    return ask(question["question"], history, session_id=session_id), previous


def rate(rows: list[dict], key: str) -> str:
    values = [r["checks"][key] for r in rows if r["checks"].get(key) is not None]
    return f"{sum(values)}/{len(values)} ({sum(values) / len(values):.0%})" if values else "-"


def write_report(rows: list[dict]) -> None:
    seconds = [r["seconds"] for r in rows]
    lines = [
        "# Laporan evaluasi",
        "",
        f"Dijalankan {datetime.now():%Y-%m-%d %H:%M}, {len(rows)} pertanyaan.",
        "",
        "| Metrik | Hasil | Target |",
        "|---|---|---|",
        f"| Ketepatan rute | {rate(rows, 'route')} | 85% atau lebih |",
        f"| Keterlacakan (ID di jawaban ada di sumber) | {rate(rows, 'traceable')} | 100% |",
        f"| Penolakan benar | {rate(rows, 'refusal')} | 3/3 |",
        f"| Angka sama dengan Pandas | {rate(rows, 'numbers')} | 100% |",
        f"| Kategori sumber sesuai | {rate(rows, 'category')} | - |",
        f"| Tool yang diharapkan terpanggil | {rate(rows, 'tools')} | - |",
        f"| Pertanyaan lanjutan memakai ID yang benar | {rate(rows, 'followup')} | - |",
        f"| Latensi median | {statistics.median(seconds):.1f} detik (maks {max(seconds):.1f}) | di bawah 15 detik |",
        f"| Token rata-rata per pertanyaan | {statistics.mean(r['tokens'] for r in rows):,.0f} | - |",
        f"| Total biaya | US${sum(r['cost_usd'] for r in rows):.4f} | - |",
        "",
        "## Per pertanyaan",
        "",
        "| ID | Rute (harap/teramati) | Tool | Cek | Lulus | Detik |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        checks = ", ".join(f"{k}={'ok' if v else 'GAGAL'}" for k, v in r["checks"].items() if v is not None)
        lines.append(
            f"| {r['id']} | {r['route_expected']}/{r['route_observed']} | {', '.join(r['tools_called']) or '-'} "
            f"| {checks} | {'ya' if r['passed'] else 'TIDAK'} | {r['seconds']} |"
        )
    lines += ["", "## Jawaban", ""]
    for r in rows:
        lines += [f"### {r['id']}", "", r["answer"].strip(), ""]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="ID pertanyaan dipisah koma")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    questions = yaml.safe_load(QUESTIONS.read_text(encoding="utf-8"))
    if args.only:
        wanted = set(args.only.split(","))
        questions = [q for q in questions if q["id"] in wanted]

    rows = []
    for question in questions:
        try:
            result, previous = run(question)
            row = evaluate(question, result, previous)
        except Exception as exc:
            row = {"id": question["id"], "group": question["group"], "route_expected": question["route"],
                   "route_observed": "galat", "tools_called": [], "checks": {"route": False}, "passed": False,
                   "seconds": 0.0, "tokens": 0, "cost_usd": 0.0, "answer": f"GALAT: {type(exc).__name__}: {exc}"}
        rows.append(row)
        failed = [k for k, v in row["checks"].items() if v is False]
        print(f"{row['id']:12s} rute {row['route_observed']:8s} {'LULUS' if row['passed'] else 'GAGAL ' + str(failed)} "
              f"({row['seconds']} detik)")
    write_report(rows)
    flush()
    print(f"Laporan: {REPORT}")
    print(f"Ketepatan rute {rate(rows, 'route')} | keterlacakan {rate(rows, 'traceable')} | "
          f"penolakan {rate(rows, 'refusal')} | angka {rate(rows, 'numbers')}")


if __name__ == "__main__":
    main()
