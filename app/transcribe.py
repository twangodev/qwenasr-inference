from qwen_asr import Qwen3ASRModel

from .schemas import TranscriptionResponse, WordTimestamp


def transcribe(model: Qwen3ASRModel, audio_path: str) -> TranscriptionResponse:
    results = model.transcribe(audio_path, return_time_stamps=True)
    r = results[0]

    timestamps = []
    if r.time_stamps is not None:
        timestamps = [
            WordTimestamp(text=item.text, start_time=item.start_time, end_time=item.end_time)
            for item in r.time_stamps
        ]

    return TranscriptionResponse(
        text=r.text,
        language=r.language,
        timestamps=timestamps,
    )