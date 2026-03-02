from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.schemas import TranscriptionResponse, WordTimestamp

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"
JFK_WAV = SAMPLES_DIR / "jfk.wav"


@pytest.fixture()
def jfk_wav_bytes():
    return JFK_WAV.read_bytes()


@pytest.fixture()
def mock_engine():
    """A mock TranscriptionEngine that returns canned results."""
    engine = MagicMock()
    engine.transcribe.return_value = TranscriptionResponse(
        text="And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country.",
        language="English",
        timestamps=[
            WordTimestamp(text="And", start_time=0.0, end_time=0.3),
            WordTimestamp(text="so", start_time=0.3, end_time=0.5),
        ],
    )
    return engine


@pytest.fixture()
def client(mock_engine):
    from app.main import app

    app.state.engine = mock_engine
    return TestClient(app)
