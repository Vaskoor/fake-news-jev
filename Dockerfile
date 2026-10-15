# MIT License
#
# Copyright (c) 2024 Bhaskar Kaphle <kaphlebhaskar012345@gmail.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software
# is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# =============================================================================
# Multi-stage Dockerfile for fake-news-jev
#
# Stage 1 (builder): compile wheels for all dependencies into /wheels
# Stage 2 (runtime): slim python:3.11-slim image, install wheels only, copy
#                    src + config + api + ui, expose 8000 (FastAPI) and
#                    8501 (Streamlit), healthcheck on /health.
# =============================================================================

ARG PYTHON_VERSION=3.11-slim

# ---------------------------------------------------------------------------
# Stage 1 — build dependencies
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION} AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Install build tooling required to compile numpy/scipy/lightgbm/xgboost from
# sdist when binary wheels are unavailable for the target platform.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        git \
        libssl-dev \
        libffi-dev \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

# Build wheels for every requirement; --prefer-binary avoids source builds
# when a manylinux wheel is available (much faster + reproducible).
RUN python -m pip install --upgrade pip wheel setuptools && \
    pip wheel --prefer-binary --wheel-dir=/wheels -r requirements.txt

# ---------------------------------------------------------------------------
# Stage 2 — runtime image
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION} AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src \
    JEV_PROVIDER=simulated \
    FAKE_NEWS_ENV=production

LABEL org.opencontainers.image.title="fake-news-jev" \
      org.opencontainers.image.description="Production-grade fake-news detection system (3-stage hybrid: Jev + LightGBM + evidence check)" \
      org.opencontainers.image.author="Bhaskar Kaphle <kaphlebhaskar012345@gmail.com>" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.source="https://github.com/kaphlebhaskar/fake-news-jev"

WORKDIR /app

# Install runtime system packages (curl for healthcheck, no build tools).
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install compiled wheels from the builder stage.
COPY --from=builder /wheels /wheels
RUN pip install --no-index --find-links=/wheels /wheels/* && \
    rm -rf /wheels

# Application code + configs + entrypoints.
COPY src/        /app/src/
COPY config/     /app/config/
COPY api/        /app/api/
COPY ui/         /app/ui/
COPY scripts/    /app/scripts/

# Persisted artefact directories (mounted as volumes in compose).
RUN mkdir -p /app/data/raw /app/data/processed /app/data/external \
             /app/models /app/results

# Default to the FastAPI server on :8000 — `ui` service overrides CMD to
# `streamlit run ui/app.py --server.port 8501` in docker-compose.yml.
EXPOSE 8000 8501

# Health check probes /health on the API container; the UI container
# overrides HEALTHCHECK to hit Streamlit's "_" endpoint in compose.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl --silent --fail http://127.0.0.1:8000/health || exit 1

CMD ["python", "-m", "uvicorn", "api.main:app", \
     "--host", "0.0.0.0", "--port", "8000"]
