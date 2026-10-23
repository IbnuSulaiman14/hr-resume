"""Jalankan: python -m src.ingest.build_index [--limit N]. Tahap B1.5 dan B1.6.

Langkah: muat CSV -> parse -> metadata -> chunk -> embed (batch, lewat OpenRouter) -> upsert ke Qdrant
(ID uuid5 dari resume_id + chunk_idx, jadi idempoten) -> tulis data/resumes_meta.csv.
"""


def main(limit: int | None = None) -> None:
    raise NotImplementedError("B1.5")


if __name__ == "__main__":
    main()
