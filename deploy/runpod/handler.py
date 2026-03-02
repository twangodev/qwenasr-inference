import base64
import tempfile
import urllib.request

import runpod

from app.engine import TranscriptionEngine

engine = TranscriptionEngine()
engine.load()


def handler(job):
    job_input = job["input"]

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        if "audio_base64" in job_input:
            tmp.write(base64.b64decode(job_input["audio_base64"]))
        elif "audio_url" in job_input:
            urllib.request.urlretrieve(job_input["audio_url"], tmp.name)
        else:
            return {"error": "Provide either audio_base64 or audio_url"}
        tmp.flush()

        result = engine.transcribe(tmp.name)

    return result.model_dump()


runpod.serverless.start({"handler": handler})
