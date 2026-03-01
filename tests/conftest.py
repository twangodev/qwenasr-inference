from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"
JFK_WAV = SAMPLES_DIR / "jfk.wav"


@pytest.fixture()
def jfk_wav_bytes():
    return JFK_WAV.read_bytes()


@pytest.fixture()
def mock_model():
    """A mock Qwen3ASRModel that returns canned results."""
    result = MagicMock()
    result.text = "And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country."
    result.language = "English"

    ts1 = MagicMock()
    ts1.text = "And"
    ts1.start_time = 0.0
    ts1.end_time = 0.3

    ts2 = MagicMock()
    ts2.text = "so"
    ts2.start_time = 0.3
    ts2.end_time = 0.5

    result.time_stamps = [ts1, ts2]

    model = MagicMock()
    model.transcribe.return_value = [result]
    return model


@pytest.fixture()
def client(mock_model):
    from app.main import app

    app.state.model = mock_model
    return TestClient(app)
