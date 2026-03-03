#!/usr/bin/env python
"""Download model weights from Hugging Face.

Reads model names from app.config.Settings defaults.
Skips files that are already cached locally.
"""

from huggingface_hub import snapshot_download

from app.config import Settings

MODELS = [Settings.asr_model, Settings.forced_aligner_model]


def main():
    for model in MODELS:
        print(f"Downloading {model}...")
        snapshot_download(model)
        print(f"Done: {model}")


if __name__ == "__main__":
    main()
