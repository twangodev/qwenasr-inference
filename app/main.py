from contextlib import asynccontextmanager

from fastapi import FastAPI
from qwen_asr import Qwen3ASRModel

from .routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = Qwen3ASRModel.from_pretrained(
        "Qwen/Qwen3-ASR-1.7B",
        forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
    )
    yield


app = FastAPI(lifespan=lifespan)
app.include_router(router)