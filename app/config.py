def _get_base_settings_class():
    """Resolve the base class for Settings.

    Uses pydantic-settings when available (enables env var overrides).
    Falls back to pydantic BaseModel in environments where pydantic-settings
    is incompatible (e.g. Cog/Replicate which bundles pydantic v1).
    See: https://github.com/replicate/cog/issues/1562
    """
    try:
        from pydantic_settings import BaseSettings

        return BaseSettings
    except ImportError:
        from pydantic import BaseModel

        return BaseModel


class Settings(_get_base_settings_class()):  # type: ignore[misc]
    asr_model: str = "Qwen/Qwen3-ASR-1.7B"
    forced_aligner_model: str = "Qwen/Qwen3-ForcedAligner-0.6B"
    device: str = ""
    max_new_tokens: int = 2048


settings = Settings()
