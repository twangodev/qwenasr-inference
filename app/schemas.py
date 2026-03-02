from pydantic import BaseModel, model_validator


class WordTimestamp(BaseModel):
    text: str
    start_time: float
    end_time: float


class TranscriptionResponse(BaseModel):
    text: str
    language: str
    timestamps: list[WordTimestamp]


class TranscriptionRequest(BaseModel):
    audio_base64: str | None = None
    audio_url: str | None = None

    @model_validator(mode="after")
    def check_one_source(self):
        if not self.audio_base64 and not self.audio_url:
            raise ValueError("Provide either audio_base64 or audio_url")
        return self
