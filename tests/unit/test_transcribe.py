def test_transcribe_returns_text_and_timestamps(client, jfk_wav_bytes):
    resp = client.post("/transcribe", files={"file": ("jfk.wav", jfk_wav_bytes, "audio/wav")})
    assert resp.status_code == 200

    data = resp.json()
    assert "ask not what your country can do for you" in data["text"].lower()
    assert data["language"] == "English"
    assert len(data["timestamps"]) >= 1
    assert all("text" in ts and "start_time" in ts and "end_time" in ts for ts in data["timestamps"])


def test_transcribe_calls_model_with_timestamps(client, jfk_wav_bytes, mock_model):
    client.post("/transcribe", files={"file": ("jfk.wav", jfk_wav_bytes, "audio/wav")})
    mock_model.transcribe.assert_called_once()
    args = mock_model.transcribe.call_args
    assert args.kwargs.get("return_time_stamps") is True


def test_transcribe_no_file_returns_422(client):
    resp = client.post("/transcribe")
    assert resp.status_code == 422