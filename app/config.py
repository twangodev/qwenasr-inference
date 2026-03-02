from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    asr_model: str = "Qwen/Qwen3-ASR-1.7B"
    forced_aligner_model: str = "Qwen/Qwen3-ForcedAligner-0.6B"
    device: str = ""
    max_new_tokens: int = 2048


settings = Settings()
