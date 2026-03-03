# qwenasr-inference

[![Python](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Ftwangodev%2Fqwenasr-inference%2Fmain%2Fpyproject.toml)](https://www.python.org/downloads/)
[![Replicate](https://img.shields.io/badge/replicate-twangodev%2Fqwenasr-%23000000?logo=replicate&logoColor=white)](https://replicate.com/twangodev/qwenasr)
[![GHCR](https://img.shields.io/badge/ghcr.io-twangodev%2Fqwenasr--inference-blue?logo=github)](https://ghcr.io/twangodev/qwenasr-inference)
[![License](https://img.shields.io/github/license/twangodev/qwenasr-inference)](LICENSE)

Inference server for Qwen3 speech recognition and alignment.

## Quickstart

```bash
docker run --gpus all -p 8000:8000 ghcr.io/twangodev/qwenasr-inference
```

Or build locally:

```bash
docker compose up --build
```

Or run without Docker:

```bash
uv sync
uv run poe serve
```

API available at `http://localhost:8000/docs`.

## Usage

```bash
curl -X POST http://localhost:8000/transcribe -F "file=@audio.wav"
```

## Configuration

| Variable | Default | Description |
|---|---|---|
| `ASR_MODEL` | `Qwen/Qwen3-ASR-1.7B` | ASR model |
| `FORCED_ALIGNER_MODEL` | `Qwen/Qwen3-ForcedAligner-0.6B` | Aligner model |
| `DEVICE` | auto-detect | Device (`cuda:0`, `cpu`) |
| `MAX_NEW_TOKENS` | `2048` | Max generation tokens |
