import asyncio
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Request, UploadFile

from .schemas import TranscriptionResponse
from .transcribe import transcribe

router = APIRouter()


@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(request: Request, file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix if file.filename else ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp.flush()

        result = await asyncio.to_thread(
            transcribe, request.app.state.model, tmp.name
        )
    return result


@router.get("/health")
async def health():
    return {"status": "ok"}