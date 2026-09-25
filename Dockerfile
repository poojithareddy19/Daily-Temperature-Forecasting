FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app

WORKDIR /app

COPY --from=builder /install /usr/local

COPY configs/ ./configs/
COPY params.yaml ./
COPY src/ ./src/

# Build the model into the image (data is fetched from a public URL):
RUN python -m src.data.ingest \
    && python -m src.features.build_features \
    && python -m src.models.train

EXPOSE 8000

CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]