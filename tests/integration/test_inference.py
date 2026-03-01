import pytest
from fastapi.testclient import TestClient

from tests.conftest import JFK_WAV


@pytest.fixture(scope="module")
def real_client():
    from qwen_asr import Qwen3ASRModel

    from app.main import app

    app.state.model = Qwen3ASRModel.from_pretrained(
        "Qwen/Qwen3-ASR-1.7B",
        forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
    )
    return TestClient(app)


def test_transcribe_jfk(real_client):
    resp = real_client.post(
        "/transcribe",
        files={"file": ("jfk.wav", JFK_WAV.read_bytes(), "audio/wav")},
    )
    assert resp.status_code == 200

    data = resp.json()
    assert "ask not what your country can do for you" in data["text"].lower()
    assert data["language"] == "English"
    assert len(data["timestamps"]) > 0

    for ts in data["timestamps"]:
        assert ts["text"]
        assert ts["start_time"] >= 0
        assert ts["end_time"] > ts["start_time"]