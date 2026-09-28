"""Env-driven configuration. All secrets/config flow through here -- never hardcode."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent
METHOD_RULES_PATH = REPO_ROOT / "config" / "method_rules.yaml"


class Settings(BaseSettings):
    """Application settings, loaded from environment / .env."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    anthropic_api_key: str = Field(default="")
    anthropic_model: str = Field(default="claude-sonnet-4-6")
    anthropic_report_model: str = Field(default="claude-sonnet-4-6")

    fred_api_key: str = Field(default="")

    langchain_tracing_v2: bool = Field(default=False)
    langchain_api_key: str = Field(default="")
    langchain_project: str = Field(default="cannibalization-agent")

    duckdb_path: str = Field(default="data/processed/store.duckdb")

    random_seed: int = Field(default=42)

    log_level: str = Field(default="INFO")


@lru_cache
def get_settings() -> Settings:
    return Settings()
