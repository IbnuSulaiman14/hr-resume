# Asisten Rekruter

Chatbot RAG multi-agent untuk mencari, mencocokkan, dan membandingkan kandidat dari 2.484 CV (Resume Dataset, Kaggle). Setiap jawaban menyertakan ID CV dan sumbernya. Capstone Module 3 Purwadhika.

> Status: kerangka. README final diisi pada tugas B6.3 di `docs/BACKLOG.md`.

## Aplikasi
- Demo: _(isi link Streamlit Community Cloud)_
- Video penjelasan: _(isi link)_

## Fitur
- Cari kandidat berdasarkan makna, dengan filter kategori dan seksi.
- Cocokkan CV dengan deskripsi lowongan.
- Statistik pool kandidat (sebaran kategori, filter pengalaman, skill terbanyak).
- Sumber RAG, tool call, dan penggunaan token ditampilkan di setiap jawaban.
- Prompt dikelola di Langfuse; setiap permintaan menghasilkan trace.

## Arsitektur
_(tempel diagram dari `docs/PRD.md` bagian 3)_

Stack: LangChain, LangGraph, Qdrant Cloud, OpenRouter (chat dan embedding), Langfuse, Streamlit.

## Menjalankan
```
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env      # isi kunci
python -m src.ingest.build_index
streamlit run app.py
```

## Evaluasi
_(isi hasil `python -m eval.run_eval`)_

## Batasan
- Data tidak memuat nama kandidat, perusahaan asli, dan gaji.
- Tahun pengalaman dan level pendidikan adalah perkiraan hasil ekstraksi.
- Aplikasi adalah alat bantu penyaringan awal. Aplikasi tidak memutuskan siapa yang diterima dan tidak menyaring berdasarkan atribut pribadi.
