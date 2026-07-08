"""Application configuration loaded exclusively from environment variables."""

from functools import lru_cache
from typing import Literal

from pydantic import AnyHttpUrl, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Validated runtime settings; secrets are masked in logs and repr output."""

    hf_token: SecretStr = Field(alias="HF_TOKEN")
    llm_api_key: SecretStr = Field(alias="LLM_API_KEY")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", alias="LOG_LEVEL"
    )
    llm_base_url: AnyHttpUrl = Field(
        default="https://api.openai.com/v1", alias="LLM_BASE_URL"
    )
    llm_model: str = Field(default="gpt-4o-mini", alias="LLM_MODEL")
    request_timeout_seconds: float = Field(default=25.0, ge=1.0, le=120.0)

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return one immutable-style settings instance per warm function process."""

    return Settings()  # type: ignore[call-arg]
