FROM python:3.14-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY . .
RUN uv sync --frozen --no-dev

RUN uv run huggingface-cli download Qwen/Qwen3-ASR-1.7B && \
    uv run huggingface-cli download Qwen/Qwen3-ForcedAligner-0.6B

FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY --from=builder /app /app
COPY --from=builder /root/.cache/huggingface /root/.cache/huggingface

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=60s \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uv", "run", "poe", "serve"]
