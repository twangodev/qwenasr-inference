FROM python:3.14-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project --extra runpod

COPY . .
RUN uv sync --frozen --no-dev --extra runpod

RUN uv run hf download Qwen/Qwen3-ASR-1.7B && \
    uv run hf download Qwen/Qwen3-ForcedAligner-0.6B

FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY --from=builder /app /app
COPY --from=builder /root/.cache/huggingface /root/.cache/huggingface

CMD ["uv", "run", "python", "-m", "deploy.runpod.handler"]
