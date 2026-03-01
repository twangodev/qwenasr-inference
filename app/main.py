from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from qwen_asr import Qwen3ASRModel

from . import __version__
from .routes import router

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE != "cpu" else torch.float32


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = Qwen3ASRModel.from_pretrained(
        "Qwen/Qwen3-ASR-1.7B",
        forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
        forced_aligner_kwargs=dict(device_map=DEVICE, dtype=DTYPE),
        device_map=DEVICE,
        dtype=DTYPE,
        max_new_tokens=2048,
    )
    yield


app = FastAPI(
    title="Qwen ASR Inference",
    description="Speech recognition and forced alignment using Qwen3-ASR and Qwen3-ForcedAligner.",
    version=__version__,
    lifespan=lifespan,
)
app.include_router(router)
