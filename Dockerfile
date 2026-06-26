# ==============================================================================
# Stage 1: Base image with shared configuration
# ==============================================================================
FROM python:3.10-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="/app"

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

# ==============================================================================
# Stage 3: Production runtime image
# ==============================================================================
FROM base AS production

WORKDIR /app

# Copy the virtual environment from builder stage
COPY --from=builder /app/.venv /app/.venv

# Copy configurations and imputation scripts
COPY configs/ ./configs/
COPY Imputation/ ./Imputation/

# Create data and results mount points and ensure correct ownership later
RUN mkdir -p data results

# Copy the core script runner entrypoint
COPY entrypoint.sh ./
RUN chmod +x entrypoint.sh

# Copy all research scripts in the root directory
COPY train_gtex.py \
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

# Create a non-root system user and group (avoid running container processes as root)
RUN groupadd -g 10001 appgroup && \
    useradd -r -u 10001 -g appgroup -d /app -s /sbin/nologin appuser

# Set permissions for the application folder and volumes
RUN chown -R appuser:appgroup /app

# Use the non-root user
USER appuser

# Set PATH to use the virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# Define default entrypoint and command
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["python", "student_pipeline.py"]
