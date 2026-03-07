# ============================================================
# Seren ML Pipeline — Docker Image
# ============================================================
# Multi-stage build for training the stress prediction model.
#
# Usage:
#   docker build -t seren-ml ./ml
#   docker run seren-ml                    # Train with synthetic data
#   docker run -v ./data:/app/data seren-ml  # Train with real WESAD data
# ============================================================

FROM python:3.11-slim AS base

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ============================================================
# Test stage — runs unit tests
# ============================================================
FROM base AS test

COPY src/ /app/src/
COPY tests/ /app/tests/
COPY config/ /app/config/

WORKDIR /app
RUN python -m pytest tests/ -v --tb=short

# ============================================================
# Train stage — runs model training
# ============================================================
FROM base AS train

COPY src/ /app/src/
COPY config/ /app/config/

# Create output directories
RUN mkdir -p /app/models /app/data /app/assets/ml

# Default environment
ENV PYTHONPATH=/app/src
ENV USE_MLFLOW=true
ENV MLFLOW_TRACKING_URI=http://mlflow:5000
ENV LOG_LEVEL=INFO

WORKDIR /app/src

CMD ["python", "train.py"]
