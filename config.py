from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    model: str = "google/gemma-3-4b-it"
    hf_token: str
    judge_provider: Literal["openai", "anthropic", "ollama"] = "ollama"
    judge_model_name: str = "gpt-oss:20b"
