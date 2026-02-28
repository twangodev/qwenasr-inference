from qwen_asr import Qwen3ASRModel, ForcedAligner

from .schemas import TranscriptionResponse, WordTimestamp


def transcribe(model: Qwen3ASRModel, aligner: ForcedAligner, audio_path: str) -> TranscriptionResponse:
    result = model.transcribe(audio_path, return_time_stamps=True)
    text = result["text"]
    language = result.get("language", "unknown")
    raw_timestamps = result.get("time_stamps", [])

    timestamps = [
        WordTimestamp(text=entry["text"], start_time=entry["start"], end_time=entry["end"])
        for entry in raw_timestamps
    ]

    return TranscriptionResponse(text=text, language=language, timestamps=timestamps)