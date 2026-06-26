# ==============================================================================
# Stage 1: Base image with shared configuration
# ==============================================================================
# Pinned version of Python 3.10 slim bookworm for byte-for-byte reproducibility
FROM python:3.10.14-slim-bookworm AS base

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="/app"

# Pip-related environment variables
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1

# Install runtime system dependencies (e.g., libgomp for openmp in PyTorch/scipy/numpy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ==============================================================================
# Stage 2: Builder image (installing dependencies)
# ==============================================================================
FROM base AS builder

# Install uv from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set uv configuration variables
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

WORKDIR /app

# Install build dependencies (git is required for blitzgsea; build-essential is for C-extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy configuration and lockfiles for dependency resolution
COPY pyproject.toml uv.lock ./

# Install dependencies using build cache mount to speed up subsequent builds
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# Copy package source code
COPY src/ ./src/
COPY README.md ./

# Sync to install the local topohyfa package in the virtual environment
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Generate package version details file
RUN uv pip freeze > package_versions.txt

# ==============================================================================
# Stage 3: Production runtime image
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

# Create a non-root system user and group first to allow COPY --chown ownership
RUN groupadd -g 10001 appgroup && \
    useradd -r -u 10001 -g appgroup -d /app -s /sbin/nologin appuser

# Copy the virtual environment and package version details from builder stage
COPY --from=builder --chown=appuser:appgroup /app/.venv /app/.venv
COPY --from=builder --chown=appuser:appgroup /app/package_versions.txt /app/package_versions.txt
COPY --from=builder --chown=appuser:appgroup /app/pyproject.toml /app/pyproject.toml
COPY --from=builder --chown=appuser:appgroup /app/uv.lock /app/uv.lock

# Copy configurations, source code, and imputation scripts
COPY --chown=appuser:appgroup configs/ ./configs/
COPY --chown=appuser:appgroup src/ ./src/
COPY --chown=appuser:appgroup Imputation/ ./Imputation/

# Create data and results mount points (these are mounted as volumes)
RUN mkdir -p data results && chown -R appuser:appgroup data results

# Declare volumes for datasets and outputs
VOLUME ["/app/data", "/app/results"]

# Copy the core script runner entrypoint
COPY --chown=appuser:appgroup entrypoint.sh ./
RUN chmod +x entrypoint.sh

# Copy all research scripts in the root directory
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

# Use the non-root user
USER appuser

# Set PATH to use the virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# Run a lightweight health check to ensure container readiness
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from src.data import Data" || exit 1

# Define default entrypoint and command
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["python", "student_pipeline.py"]
