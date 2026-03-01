import pytest
from fastapi.testclient import TestClient

from app.main import app
from tests.conftest import JFK_WAV


@pytest.fixture(scope="module")
def real_client():
    with TestClient(app) as client:
        yield client


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
