"""Environment-backed configuration for experiment scripts.

Example:
    >>> from config import Config
    >>> cfg = Config()  # Reads values from .env and the shell environment.
    >>> cfg.judge_provider
    'anthropic'
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal


class Config(BaseSettings):
    """Load runtime settings from `.env` and environment variables.

    Attributes:
        model: Default Hugging Face model name used by the experiments.
        hf_token: Token used to authenticate against Hugging Face Hub.
        judge_provider: Backend used by `pydantic_ai.Agent` for judging.
        judge_model_name: Judge model identifier for the configured provider.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    model: str = "google/gemma-3-4b-it"
    hf_token: str
    judge_provider: Literal["openai", "anthropic", "ollama"] = "ollama"
    judge_model_name: str = "gpt-oss:20b"
