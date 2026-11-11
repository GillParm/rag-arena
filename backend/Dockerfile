# syntax=docker/dockerfile:1.7
# API image. Build from the repo root so the published leaderboard can be baked in:
#   docker build -f backend/Dockerfile -t rag-arena-api .
# Cloud Run sets $PORT; locally it defaults to 8300.

FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

FROM base AS build
COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never
WORKDIR /src
COPY backend/pyproject.toml backend/uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project
COPY backend/src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

# Download the local models at build time so a cold start never waits on Hugging Face.
# CI passes BAKE_MODELS=false to keep the build fast.
FROM build AS models
ARG BAKE_MODELS=true
ARG EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
ARG RERANKER_MODEL=BAAI/bge-reranker-base
RUN mkdir -p /models && if [ "$BAKE_MODELS" = "true" ]; then \
      /opt/venv/bin/python -c "import sys; from fastembed import TextEmbedding; \
from fastembed.rerank.cross_encoder import TextCrossEncoder; \
TextEmbedding(sys.argv[1], cache_dir='/models'); TextCrossEncoder(sys.argv[2], cache_dir='/models')" \
      "$EMBEDDING_MODEL" "$RERANKER_MODEL"; \
    fi

FROM base AS runtime
RUN useradd --create-home --uid 10001 app \
    && mkdir -p /app/data /app/results/runs/main \
    && chown -R app:app /app
COPY --from=build /opt/venv /opt/venv
COPY --from=models --chown=app:app /models /models
COPY --chown=app:app results/runs/main/summary.json /app/results/runs/main/summary.json
ENV PATH=/opt/venv/bin:$PATH \
    RAG_ARENA_MODEL_CACHE_DIR=/models \
    RAG_ARENA_DATA_DIR=/app/data \
    RAG_ARENA_RESULTS_DIR=/app/results \
    HF_HUB_OFFLINE=1 \
    PORT=8300
WORKDIR /app
USER app
EXPOSE 8300
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health', timeout=4)"
CMD ["sh", "-c", "exec uvicorn --factory rag_arena.api.app:create_app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]
