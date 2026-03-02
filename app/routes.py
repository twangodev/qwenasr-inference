import asyncio
import base64
import tempfile
import urllib.request
from pathlib import Path

from fastapi import APIRouter, File, Request, UploadFile

from .schemas import TranscriptionRequest, TranscriptionResponse

router = APIRouter()


@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(request: Request, file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix if file.filename else ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp.flush()

        result = await asyncio.to_thread(request.app.state.engine.transcribe, tmp.name)
    return result


@router.post("/transcribe/json", response_model=TranscriptionResponse)
async def transcribe_json(request: Request, body: TranscriptionRequest):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        if body.audio_base64:
            tmp.write(base64.b64decode(body.audio_base64))
        elif body.audio_url:
            urllib.request.urlretrieve(body.audio_url, tmp.name)
        tmp.flush()

        result = await asyncio.to_thread(request.app.state.engine.transcribe, tmp.name)
    return result


@router.get("/health")
async def health():
    return {"status": "ok"}
