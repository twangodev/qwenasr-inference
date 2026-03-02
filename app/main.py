from contextlib import asynccontextmanager

from fastapi import FastAPI

from . import __version__
from .engine import TranscriptionEngine
from .routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine = TranscriptionEngine()
    engine.load()
    app.state.engine = engine
    yield


app = FastAPI(
    title="Qwen ASR Inference",
    description="Speech recognition and forced alignment using Qwen3-ASR and Qwen3-ForcedAligner.",
    version=__version__,
    lifespan=lifespan,
)
app.include_router(router)
