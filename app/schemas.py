from pydantic import BaseModel


class WordTimestamp(BaseModel):
    text: str
    start_time: float
    end_time: float


class TranscriptionResponse(BaseModel):
    text: str
    language: str
    timestamps: list[WordTimestamp]


class TranscriptionRequest(BaseModel):
    audio_base64: str


class ForceAlignRequest(BaseModel):
    audio_base64: str
    text: str
    language: str


class ForceAlignResponse(BaseModel):
    timestamps: list[WordTimestamp]
