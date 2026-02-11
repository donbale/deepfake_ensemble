# ============================================================
# Deepfake Ensemble API - Multi-stage Docker Build
# ============================================================

# --- Stage 1: Builder (compile dlib) ---
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# --- Stage 2: Runtime ---
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libopenblas0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

WORKDIR /app

# Copy application code
COPY detectors/ ./detectors/
COPY main.py .
COPY optimal_weights.json* ./
COPY tool_definition.json* ./
COPY scripts/download_models.py* ./scripts/

# Create directories
RUN mkdir -p /models/hf-cache /models/dlib

# Environment defaults
ENV DEVICE=cpu \
    PORT=8080 \
    HF_HOME=/models/hf-cache \
    TRANSFORMERS_CACHE=/models/hf-cache \
    PYTHONUNBUFFERED=1 \
    FAKE_THRESHOLD=0.5 \
    LOG_LEVEL=INFO \
    UVICORN_LOG_LEVEL=info

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

CMD ["python", "main.py"]
