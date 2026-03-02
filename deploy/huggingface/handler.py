import base64
import tempfile
import urllib.request

import torch
from qwen_asr import Qwen3ASRModel


class EndpointHandler:
    def __init__(self, path=""):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device != "cpu" else torch.float32

        self.model = Qwen3ASRModel.from_pretrained(
            "Qwen/Qwen3-ASR-1.7B",
            forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
            forced_aligner_kwargs=dict(device_map=device, dtype=dtype),
            device_map=device,
            dtype=dtype,
            max_new_tokens=2048,
        )

    def __call__(self, data):
        inputs = data.get("inputs", data)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            if isinstance(inputs, str):
                tmp.write(base64.b64decode(inputs))
            elif isinstance(inputs, dict) and "audio_base64" in inputs:
                tmp.write(base64.b64decode(inputs["audio_base64"]))
            elif isinstance(inputs, dict) and "audio_url" in inputs:
                urllib.request.urlretrieve(inputs["audio_url"], tmp.name)
            else:
                return {
                    "error": "Provide inputs as base64 string, or dict with audio_base64/audio_url"
                }
            tmp.flush()

            results = self.model.transcribe(tmp.name, return_time_stamps=True)
            r = results[0]

        timestamps = []
        if r.time_stamps is not None:
            timestamps = [
                {
                    "text": item.text,
                    "start_time": item.start_time,
                    "end_time": item.end_time,
                }
                for item in r.time_stamps
            ]

        return {
            "text": r.text,
            "language": r.language,
            "timestamps": timestamps,
        }
