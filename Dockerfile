# Multi-stage production container build for Abuse Ring Detector (Model F Inference API)
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project specification and package sources for the wheel build
COPY pyproject.toml .
COPY src/ /app/src/

# Install dependencies using pip into local wheel directory
RUN pip install --no-cache-dir --upgrade pip wheel setuptools
RUN pip wheel --no-cache-dir --wheel-dir /app/wheels .

# Final minimal production image
FROM python:3.11-slim AS runner

WORKDIR /app

# Create non-root system user for security
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /bin/sh -m appuser

# Copy built wheels and install
COPY --from=builder /app/wheels /wheels
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels

# Copy application source code and configuration
COPY src/ /app/src/
COPY configs/ /app/configs/
COPY pyproject.toml /app/
COPY model_f_r1_manifest.json /app/model_f_r1_manifest.json
COPY inference_contract_r1.json /app/inference_contract_r1.json

# Create logs & artifacts directories with non-root ownership
RUN mkdir -p /app/logs /app/artifacts && \
    chown -R appuser:appgroup /app

# Environment variables
ENV PYTHONPATH="/app/src" \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    HOST=0.0.0.0 \
    WORKERS=4 \
    SHADOW_MODE=true \
    ENFORCE_DECISIONS=false

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["uvicorn", "abuse_ring_detector.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
