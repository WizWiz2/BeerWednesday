"""Configuration helpers for the Beer Wednesday bot."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Settings:
    """Runtime configuration loaded from environment variables."""

    telegram_token: str
    groq_api_key: str
    groq_model: str = "llava-v1.5-7b-4096-preview"
    groq_base_url: str = "https://api.groq.com/openai/v1/chat/completions"
    temperature: float = 0.7
    max_tokens: int = 1024

    @classmethod
    def load(cls) -> "Settings":
        """Load settings from environment variables.

        Raises:
            RuntimeError: If required variables are missing.
        """

        telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
        groq_api_key = os.getenv("GROQ_API_KEY")
        groq_model = os.getenv("GROQ_MODEL", cls.groq_model)
        groq_base_url = os.getenv("GROQ_BASE_URL", cls.groq_base_url)
        temperature_str = os.getenv("GROQ_TEMPERATURE")
        max_tokens_str = os.getenv("GROQ_MAX_TOKENS")

        deprecated_models = {
            "llama-3.2-11b-vision-preview": cls.groq_model,
        }

        if groq_model in deprecated_models:
            replacement = deprecated_models[groq_model]
            LOGGER.warning(
                "Groq model '%s' is no longer supported. Falling back to '%s'.",
                groq_model,
                replacement,
            )
            groq_model = replacement

        missing = [
            name
            for name, value in {
                "TELEGRAM_BOT_TOKEN": telegram_token,
                "GROQ_API_KEY": groq_api_key,
            }.items()
            if not value
        ]

        if missing:
            raise RuntimeError(
                "Отсутствуют обязательные переменные окружения: " + ", ".join(missing)
            )

        temperature = float(temperature_str) if temperature_str else cls.temperature
        max_tokens = int(max_tokens_str) if max_tokens_str else cls.max_tokens

        return cls(
            telegram_token=telegram_token,
            groq_api_key=groq_api_key,
            groq_model=groq_model,
            groq_base_url=groq_base_url,
            temperature=temperature,
            max_tokens=max_tokens,
        )
