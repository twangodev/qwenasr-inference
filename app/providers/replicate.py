from cog import BasePredictor, Input, Path  # type: ignore[import-not-found]

from app.engine import TranscriptionEngine


class Predictor(BasePredictor):
    def setup(self):
        self.engine = TranscriptionEngine()
        self.engine.load()

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
            default="en",
            description="Language code (used with align mode)",
        ),
    ) -> dict:
        audio_path = str(audio)

        if mode == "transcribe":
            result = self.engine.transcribe(audio_path)
        else:
            if not text:
                raise ValueError("'text' is required when mode is 'align'")
            result = self.engine.align(audio_path, text, language)

        return result.model_dump()