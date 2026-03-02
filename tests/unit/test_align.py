import base64


def test_align_file_upload(client, jfk_wav_bytes, mock_engine):
    resp = client.post(
        "/align",
        files={"file": ("jfk.wav", jfk_wav_bytes, "audio/wav")},
        data={"text": "hello world", "language": "English"},
    )
    assert resp.status_code == 200

    data = resp.json()
    assert len(data["timestamps"]) == 2
    assert data["timestamps"][0]["text"] == "hello"
    mock_engine.align.assert_called_once()


def test_align_json_base64(client, jfk_wav_bytes, mock_engine):
    audio_b64 = base64.b64encode(jfk_wav_bytes).decode()
    resp = client.post(
        "/align/json",
        json={
            "audio_base64": audio_b64,
            "text": "hello world",
            "language": "English",
        },
    )
    assert resp.status_code == 200

    data = resp.json()
    assert len(data["timestamps"]) == 2
    assert data["timestamps"][1]["text"] == "world"
    mock_engine.align.assert_called_once()


def test_align_json_missing_text(client):
    resp = client.post(
        "/align/json",
        json={"audio_base64": "abc", "language": "English"},
    )
    assert resp.status_code == 422


def test_align_json_missing_audio(client):
    resp = client.post(
        "/align/json",
        json={"text": "hello world", "language": "English"},
    )
    assert resp.status_code == 422
