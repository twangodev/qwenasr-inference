from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from qwen_asr import Qwen3ASRModel

from . import __version__
from .config import settings
from .routes import router

DEVICE = settings.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16 if DEVICE != "cpu" else torch.float32


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = Qwen3ASRModel.from_pretrained(
        settings.asr_model,
        forced_aligner=settings.forced_aligner_model,
        forced_aligner_kwargs=dict(device_map=DEVICE, dtype=DTYPE),
        device_map=DEVICE,
        dtype=DTYPE,
        max_new_tokens=settings.max_new_tokens,
    )
    yield


app = FastAPI(
    title="Qwen ASR Inference",
    description="Speech recognition and forced alignment using Qwen3-ASR and Qwen3-ForcedAligner.",
    version=__version__,
    lifespan=lifespan,
)
app.include_router(router)
