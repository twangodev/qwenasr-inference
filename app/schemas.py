from pydantic import BaseModel


class WordTimestamp(BaseModel):
    text: str
    start_time: float
    end_time: float


class TranscriptionResponse(BaseModel):
    text: str
    language: str
    timestamps: list[WordTimestamp]