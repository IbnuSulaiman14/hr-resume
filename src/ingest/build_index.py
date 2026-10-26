"""Jalankan: python -m src.ingest.build_index [--limit N] [--recreate] [--payload-only].

Langkah: muat CSV -> parse -> metadata -> chunk -> embed (batch, lewat OpenRouter) -> upsert ke Qdrant
(ID uuid5 dari resume_id + chunk_idx, jadi idempoten) -> tulis data/resumes_meta.csv.

`--limit N` hanya membatasi CV yang di-embed (sampel acak tetap); metadata selalu ditulis untuk semua CV.
`--recreate` menghapus koleksi dulu, dipakai bila aturan chunking berubah supaya tidak ada titik lama tertinggal.
`--payload-only` hanya memperbarui metadata (tahun pengalaman, pendidikan) di titik yang sudah ada.
"""
import argparse
import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from qdrant_client import models

from src.config import META_CSV, RAW_CSV
from src.ingest.chunking import Chunk, build_chunks, chunk_prefix
from src.ingest.parse_html import (
    ParsedResume,
    education_level,
    employment_gaps,
    estimate_years_experience,
    job_timeline,
    parse_resume,
    reference_date,
)
from src.models import get_embeddings
from src.vectorstore import collection_name, get_qdrant_client

BATCH_SIZE = 128
EMBED_WORKERS = 4
# Namespace tetap supaya ID titik sama di setiap ingest ulang
POINT_NAMESPACE = uuid.UUID("6f1c2b7e-3d4a-4e8f-9a51-2c7d8e9f0a1b")
LIST_SEPARATOR = "; "


def point_id(resume_id: str, chunk_idx: int) -> str:
    return str(uuid.uuid5(POINT_NAMESPACE, f"{resume_id}:{chunk_idx}"))


def load_resumes(path=RAW_CSV) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"ID": str})
    return df.dropna(subset=["Category"])


def metadata_row(parsed: ParsedResume, text: str = "") -> dict:
    today = reference_date(parsed.job_dates, text)
    return {
        "resume_id": parsed.resume_id,
        "category": parsed.category,
        "headline": parsed.headline,
        "years_experience": estimate_years_experience(parsed.job_dates, today=today),
        "education_level": education_level(parsed.degrees),
        "job_titles": LIST_SEPARATOR.join(parsed.job_titles),
        "skills": LIST_SEPARATOR.join(parsed.skills),
        # Disimpan sebagai JSON karena Streamlit Cloud tidak punya Resume.csv untuk mem-parse ulang HTML
        "timeline": json.dumps(job_timeline(parsed.jobs, today), ensure_ascii=False),
        "gaps": json.dumps(employment_gaps(parsed.job_dates, today), ensure_ascii=False),
    }


def chunk_payload(chunk: Chunk, meta: dict) -> dict:
    return {
        "resume_id": chunk.resume_id,
        "category": chunk.category,
        "section": chunk.section,
        "chunk_idx": chunk.chunk_idx,
        # Simpan teks tanpa awalan supaya cuplikan di UI bersih; awalan hanya untuk embedding
        "text": chunk.text.removeprefix(chunk_prefix(chunk.category, chunk.section)),
        "years_experience": meta["years_experience"],
        "education_level": meta["education_level"],
    }


def ensure_collection(client, name: str, dim: int, recreate: bool) -> None:
    if recreate and client.collection_exists(name):
        client.delete_collection(name)
    if not client.collection_exists(name):
        client.create_collection(name, vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE))
    indexes = {
        "resume_id": models.PayloadSchemaType.KEYWORD,
        "category": models.PayloadSchemaType.KEYWORD,
        "section": models.PayloadSchemaType.KEYWORD,
        "education_level": models.PayloadSchemaType.KEYWORD,
        "years_experience": models.PayloadSchemaType.FLOAT,
    }
    for field, schema in indexes.items():
        client.create_payload_index(name, field_name=field, field_schema=schema)


def _retry(fn, label: str, attempts: int = 4):
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:
            if attempt == attempts - 1:
                raise
            wait = 2 ** attempt * 5
            print(f"  {label} gagal ({type(exc).__name__}), coba lagi dalam {wait} detik")
            time.sleep(wait)
    raise RuntimeError("tidak tercapai")


def _missing_chunks(client, name: str, chunks: list[Chunk]) -> list[Chunk]:
    """Buang chunk yang titiknya sudah ada, supaya ingest ulang setelah gagal tidak membayar embedding dua kali.
    Bila aturan chunking berubah, pakai --recreate karena ID sama bisa berisi teks lain."""
    missing = []
    for start in range(0, len(chunks), 500):
        batch = chunks[start : start + 500]
        found = client.retrieve(name, ids=[point_id(c.resume_id, c.chunk_idx) for c in batch], with_payload=False)
        found_ids = {str(p.id) for p in found}
        missing += [c for c in batch if point_id(c.resume_id, c.chunk_idx) not in found_ids]
    return missing


def update_payloads(client, name: str, meta: dict[str, dict], batch: int = 200) -> None:
    """Perbarui metadata di payload semua titik per CV tanpa embedding ulang (tanpa biaya)."""
    operations = [
        models.SetPayloadOperation(
            set_payload=models.SetPayload(
                payload={"years_experience": row["years_experience"], "education_level": row["education_level"]},
                filter=models.Filter(
                    must=[models.FieldCondition(key="resume_id", match=models.MatchValue(value=resume_id))]
                ),
            )
        )
        for resume_id, row in meta.items()
    ]
    for start in range(0, len(operations), batch):
        client.batch_update_points(name, update_operations=operations[start : start + batch], wait=True)
    print(f"Payload diperbarui untuk {len(operations)} CV")


def main(limit: int | None = None, recreate: bool = False, payload_only: bool = False) -> None:
    df = load_resumes()
    parsed = [parse_resume(r.ID, r.Category, r.Resume_html, r.Resume_str) for r in df.itertuples()]
    texts = dict(zip(df["ID"], df["Resume_str"].fillna("")))
    meta = {p.resume_id: metadata_row(p, texts[p.resume_id]) for p in parsed}
    pd.DataFrame(meta.values()).to_csv(META_CSV, index=False)
    print(f"Metadata: {len(meta)} CV -> {META_CSV.name}")
    if payload_only:
        update_payloads(get_qdrant_client(), collection_name(), meta)
        return

    if limit:
        sample_ids = set(df.sample(n=min(limit, len(df)), random_state=42)["ID"])
        parsed = [p for p in parsed if p.resume_id in sample_ids]
    chunks = [c for p in parsed for c in build_chunks(p)]
    print(f"Chunk: {len(chunks)} dari {len(parsed)} CV")

    embeddings = get_embeddings()
    client = get_qdrant_client()
    name = collection_name()
    dim = len(embeddings.embed_query("dimension probe"))
    ensure_collection(client, name, dim, recreate)
    todo = _missing_chunks(client, name, chunks)
    print(f"Perlu di-embed: {len(todo)} chunk (sisanya sudah ada di Qdrant)")

    started = time.perf_counter()
    batches = [todo[i : i + BATCH_SIZE] for i in range(0, len(todo), BATCH_SIZE)]
    done = 0
    # Embedding (panggilan jaringan yang lambat) paralel per gelombang. Upsert menunggu gelombangnya
    # selesai, jadi bila gagal tidak ada vektor yang sudah dibayar tetapi belum tersimpan.
    with ThreadPoolExecutor(max_workers=EMBED_WORKERS) as pool:
        for wave_start in range(0, len(batches), EMBED_WORKERS):
            wave = batches[wave_start : wave_start + EMBED_WORKERS]
            vector_batches = pool.map(
                lambda b: _retry(lambda: embeddings.embed_documents([c.text for c in b]), "embed"), wave
            )
            for batch, vectors in zip(wave, vector_batches):
                points = [
                    models.PointStruct(
                        id=point_id(c.resume_id, c.chunk_idx), vector=v, payload=chunk_payload(c, meta[c.resume_id])
                    )
                    for c, v in zip(batch, vectors)
                ]
                _retry(lambda: client.upsert(name, points=points, wait=True), "upsert")
                done += len(batch)
            print(f"  {done}/{len(todo)} chunk ({time.perf_counter() - started:.0f} detik)")

    count = client.count(name, exact=True).count
    print(f"Selesai. Titik di koleksi '{name}': {count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="jumlah CV yang di-embed (sampel acak tetap)")
    parser.add_argument("--recreate", action="store_true", help="hapus dan buat ulang koleksi")
    parser.add_argument("--payload-only", action="store_true", help="hanya perbarui metadata payload, tanpa embedding")
    args = parser.parse_args()
    main(limit=args.limit, recreate=args.recreate, payload_only=args.payload_only)
