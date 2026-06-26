# ==============================================================================
# Stage 1: Base image
# ==============================================================================
# Pinned version of Python 3.10.16 slim bookworm with digest for byte-for-byte reproducibility
FROM python:3.10.16-slim-bookworm@sha256:f9fd9a142c9e3bc54d906053b756eb7e7e386ee1cf784d82c251cf640c502512 AS base

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="/app"

# Disable pip caches
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1

# Timezone and locale environment settings for deterministic execution
ENV TZ=UTC
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Install runtime OpenMP dependency required by scientific packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ==============================================================================
# Stage 2: Builder
# ==============================================================================
FROM base AS builder

# Pinned version of uv installer corresponding to verified local setup
COPY --from=ghcr.io/astral-sh/uv:0.11.24 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

WORKDIR /app

# Install compilation toolchain and git
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy configuration and lockfiles
COPY pyproject.toml uv.lock ./

# Install external dependencies using cache mount
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# Copy package source code
COPY src/ ./src/
COPY README.md ./

# Sync to install local topohyfa package
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Generate package versions snapshot
RUN uv pip freeze > package_versions.txt

# ==============================================================================
# Stage 3: Production runtime
# ==============================================================================
FROM base AS production

ARG GIT_COMMIT=""
ARG BUILD_DATE=""

LABEL org.opencontainers.image.title="TopoHYFA" \
      org.opencontainers.image.version="0.1.0" \
      org.opencontainers.image.authors="HYFA Team" \
      org.opencontainers.image.source="https://github.com/rrsathe/TopoHYFA" \
      org.opencontainers.image.description="Topology-aware Hypergraph Foundation Model for multi-tissue gene expression imputation" \
      git_commit=$GIT_COMMIT \
      build_date=$BUILD_DATE

WORKDIR /app

# Create non-root system user and group
RUN groupadd -g 10001 appgroup && \
    useradd -r -u 10001 -g appgroup -d /app -s /sbin/nologin appuser

# Copy virtual env and version details with proper ownership
COPY --from=builder --chown=appuser:appgroup /app/.venv /app/.venv
COPY --from=builder --chown=appuser:appgroup /app/package_versions.txt /app/package_versions.txt
COPY --from=builder --chown=appuser:appgroup /app/pyproject.toml /app/pyproject.toml
COPY --from=builder --chown=appuser:appgroup /app/uv.lock /app/uv.lock

# Copy configuration, source code, and R scripts
COPY --chown=appuser:appgroup configs/ ./configs/
COPY --chown=appuser:appgroup src/ ./src/
COPY --chown=appuser:appgroup Imputation/ ./Imputation/

# Create runtime mount points
RUN mkdir -p data results && chown -R appuser:appgroup data results

# Declare volumes for datasets and outputs
VOLUME ["/app/data", "/app/results"]

# Copy runner entrypoint
COPY --chown=appuser:appgroup entrypoint.sh ./
RUN chmod +x entrypoint.sh

# Copy root research scripts
COPY --chown=appuser:appgroup \
     train_gtex.py \
     infer.py \
     student_pipeline.py \
     run_disease_prediction.py \
     benchmark_original.py \
     benchmark_teebot.py \
     evaluate_GTEx_v8_normalised.py \
     eval_15.py \
     visualize_interpretability.py \
     probe.py \
     prep_handoff.py \
     patch_evaluate.py \
     patch_evaluate_2.py \
     ./

# Run as non-root user
USER appuser

# Use virtual environment path
ENV PATH="/app/.venv/bin:$PATH"

# Healthcheck validating package imports
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from src.data import Data" || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["python", "student_pipeline.py"]
