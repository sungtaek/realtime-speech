from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ASR_", extra="ignore")

    backend: Literal["nemo", "faster_whisper"] = "nemo"
    model_name: str = "nvidia/parakeet-tdt-0.6b-v2"
    model_path: str | None = None
    device: str = "cuda"
    target_lang: str | None = None
    strip_lang_tags: bool = True
    whisper_model: str = "large-v3"
    whisper_partial_model: str = "small"
    whisper_compute_type: str = "float16"
    whisper_partial_compute_type: str | None = None
    whisper_beam_size: int = Field(default=5, ge=1)
    whisper_partial_beam_size: int = Field(default=1, ge=1)
    whisper_no_speech_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    whisper_condition_on_previous_text: bool = False
    sample_rate: Literal[16000] = 16000

    vad_aggressiveness: int = Field(default=2, ge=0, le=3)
    vad_frame_ms: Literal[10, 20, 30] = 20
    start_trigger_ms: int = Field(default=160, ge=10)
    end_silence_ms: int = Field(default=700, ge=100)
    preroll_ms: int = Field(default=300, ge=0)
    partial_interval_ms: int = Field(default=800, ge=100)
    min_partial_ms: int = Field(default=600, ge=100)
    max_utterance_ms: int = Field(default=30000, ge=1000)


@lru_cache
def get_settings() -> Settings:
    return Settings()
