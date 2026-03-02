import torch
from qwen_asr import Qwen3ASRModel

from .config import settings
from .schemas import ForceAlignResponse, TranscriptionResponse, WordTimestamp


class TranscriptionEngine:
    def __init__(self):
        self.model: Qwen3ASRModel | None = None

    def load(self):
        device = settings.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if device != "cpu" else torch.float32

        self.model = Qwen3ASRModel.from_pretrained(
            settings.asr_model,
            forced_aligner=settings.forced_aligner_model,
            forced_aligner_kwargs=dict(device_map=device, dtype=dtype),
            device_map=device,
            dtype=dtype,
            max_new_tokens=settings.max_new_tokens,
        )
        return self

    def transcribe(self, audio_path: str) -> TranscriptionResponse:
        results = self.model.transcribe(audio_path, return_time_stamps=True)
        r = results[0]

        timestamps = []
        if r.time_stamps is not None:
            timestamps = [
                WordTimestamp(
                    text=item.text, start_time=item.start_time, end_time=item.end_time
                )
                for item in r.time_stamps
            ]

        return TranscriptionResponse(
            text=r.text,
            language=r.language,
            timestamps=timestamps,
        )

    def align(self, audio_path: str, text: str, language: str) -> ForceAlignResponse:
        results = self.model.forced_aligner.align(audio_path, text, language)
        timestamps = [
            WordTimestamp(
                text=item.text, start_time=item.start_time, end_time=item.end_time
            )
            for item in results[0].items
        ]
        return ForceAlignResponse(timestamps=timestamps)
