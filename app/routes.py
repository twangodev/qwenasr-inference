import asyncio
import base64
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, Request, UploadFile

from .schemas import (
    ForceAlignRequest,
    ForceAlignResponse,
    TranscriptionRequest,
    TranscriptionResponse,
)

router = APIRouter()


async def _transcribe(engine, audio_bytes: bytes, suffix: str) -> TranscriptionResponse:
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        return await asyncio.to_thread(engine.transcribe, tmp.name)


async def _align(
    engine, audio_bytes: bytes, suffix: str, text: str, language: str
) -> ForceAlignResponse:
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        return await asyncio.to_thread(engine.align, tmp.name, text, language)


@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_file(request: Request, file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix if file.filename else ".wav"
    audio_bytes = await file.read()
    return await _transcribe(request.app.state.engine, audio_bytes, suffix)


@router.post("/transcribe/json", response_model=TranscriptionResponse)
async def transcribe_json(request: Request, body: TranscriptionRequest):
    audio_bytes = base64.b64decode(body.audio_base64)
    return await _transcribe(request.app.state.engine, audio_bytes, ".wav")


@router.post("/align", response_model=ForceAlignResponse)
async def align_file(
    request: Request,
    file: UploadFile = File(...),
    text: str = Form(...),
    language: str = Form(...),
):
    suffix = Path(file.filename).suffix if file.filename else ".wav"
    audio_bytes = await file.read()
    return await _align(request.app.state.engine, audio_bytes, suffix, text, language)


@router.post("/align/json", response_model=ForceAlignResponse)
async def align_json(request: Request, body: ForceAlignRequest):
    audio_bytes = base64.b64decode(body.audio_base64)
    return await _align(
        request.app.state.engine, audio_bytes, ".wav", body.text, body.language
    )


@router.get("/health")
async def health():
    return {"status": "ok"}
