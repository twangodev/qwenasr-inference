import base64
import tempfile
import urllib.request

import modal

image = (
    modal.Image.debian_slim(python_version="3.14")
    .pip_install("qwen-asr>=0.0.6", "fastapi[standard]>=0.134.0", "pydantic-settings")
    .run_commands(
        "python -c \"from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-ASR-1.7B')\"",
        "python -c \"from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-ForcedAligner-0.6B')\"",
    )
)

app = modal.App("qwenasr-inference", image=image)


@app.cls(gpu="A10G")
class ASRService:
    @modal.enter()
    def load_model(self):
        from app.engine import TranscriptionEngine

        self.engine = TranscriptionEngine()
        self.engine.load()

    @modal.fastapi_endpoint(method="POST")
    async def transcribe(self, request: dict):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            if "audio_base64" in request:
                tmp.write(base64.b64decode(request["audio_base64"]))
            elif "audio_url" in request:
                urllib.request.urlretrieve(request["audio_url"], tmp.name)
            else:
                return {"error": "Provide either audio_base64 or audio_url"}
            tmp.flush()

            result = self.engine.transcribe(tmp.name)
        return result.model_dump()

    @modal.fastapi_endpoint(method="GET")
    async def health(self):
        return {"status": "ok"}
