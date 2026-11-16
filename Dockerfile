# Image API Asisten Rekruter (FastAPI + uvicorn), siap untuk Google Cloud Run.
# Build: docker build -t hr-resume-api .
# Jalan: docker run --rm -p 8080:8080 --env-file .env hr-resume-api   lalu buka http://localhost:8080/docs
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependensi dipasang lebih dulu supaya lapisan ini di-cache selama requirements.txt tidak berubah
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src ./src
COPY server.py .
COPY data/resumes_meta.csv ./data/resumes_meta.csv

RUN useradd --create-home --uid 10001 app
USER app

# Cloud Run mengisi PORT (bawaan 8080). exec menjadikan uvicorn PID 1 agar menerima SIGTERM dan sempat
# mengirim sisa trace Langfuse; Cloud Run memberi sekitar 10 detik sebelum mematikan paksa.
EXPOSE 8080
CMD ["sh", "-c", "exec uvicorn server:app --host 0.0.0.0 --port ${PORT:-8080} --timeout-graceful-shutdown 8"]
