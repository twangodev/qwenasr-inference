import os

from cog import BasePredictor, Input, Path  # type: ignore[import-not-found]

import torch

from app.engine import TranscriptionEngine

WEIGHTS_DIR = "weights"


def use_local_weights():
    if os.path.isdir(WEIGHTS_DIR):
        os.environ["HF_HOME"] = WEIGHTS_DIR


class Predictor(BasePredictor):
    def setup(self):
        use_local_weights()
        self.engine = TranscriptionEngine()
        self.engine.load(dtype=torch.float16)

    def predict(
        self,
        audio: Path = Input(description="Audio file to process"),
        mode: str = Input(
            choices=["transcribe", "align"],
            default="transcribe",
            description="Operation mode",
        ),
        text: str = Input(
            default="",
            description="Text to align (required when mode is 'align')",
        ),
        language: str = Input(
            default="English",
            choices=[
                "Chinese",
                "English",
                "Cantonese",
                "French",
                "German",
                "Italian",
                "Japanese",
                "Korean",
                "Portuguese",
                "Russian",
                "Spanish",
            ],
            description="Language (used with align mode)",
        ),
    ) -> dict:
        audio_path = str(audio)

        if mode == "transcribe":
            result = self.engine.transcribe(audio_path)
        else:
            if not text:
                raise ValueError("'text' is required when mode is 'align'")
            result = self.engine.align(audio_path, text, language)

        return result.dict()
