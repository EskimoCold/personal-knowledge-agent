FROM ghcr.io/astral-sh/uv:0.11.8 AS uv
FROM python:3.12-slim AS base

COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_LINK_MODE=copy PATH="/app/.venv/bin:$PATH" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY pyproject.toml uv.lock ./

FROM base AS embeddings
ARG EMBEDDING_EXTRA=local-embeddings
ENV HF_HOME=/model-cache
RUN mkdir -p /model-cache && chown 10001:10001 /model-cache
RUN uv sync --locked --no-dev --extra "$EMBEDDING_EXTRA"
COPY app ./app
USER 10001:10001

FROM base AS app
ENV CMAKE_ARGS="-DGGML_NATIVE=OFF" CMAKE_BUILD_PARALLEL_LEVEL=2

RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake \
    && rm -rf /var/lib/apt/lists/*

RUN uv sync --locked --no-dev --extra local-model
COPY app ./app
COPY alembic.ini ./
COPY migrations ./migrations

USER 10001:10001
