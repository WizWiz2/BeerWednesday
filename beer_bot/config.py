"""Configuration helpers for the Beer Wednesday bot."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Iterable, Tuple


LOGGER = logging.getLogger(__name__)


def _lookup_env(names: Iterable[str]) -> Tuple[str | None, str | None]:
    """Return the first non-empty environment value from the provided names."""

    for name in names:
        raw_value = os.getenv(name)
        if raw_value is None:
            continue
        value = raw_value.strip()
        if value:
            if name != names[0]:
                LOGGER.info(
                    "Используется альтернативное имя переменной окружения %s", name
                )
            return value, name
    return None, None


@dataclass(frozen=True)
class Settings:
    """Runtime configuration loaded from environment variables."""

    telegram_token: str
    groq_api_key: str
    groq_model: str = "llama-3.2-vision"
    groq_base_url: str = "https://api.groq.com/openai/v1/chat/completions"
    temperature: float = 0.7
    max_tokens: int = 1024

    @classmethod
    def load(cls) -> "Settings":
        """Load settings from environment variables.

        Raises:
            RuntimeError: If required variables are missing.
        """

        telegram_token, _ = _lookup_env(
            ("TELEGRAM_BOT_TOKEN", "telegram_bot_token", "TELEGRAM_TOKEN")
        )
        groq_api_key, _ = _lookup_env(("GROQ_API_KEY", "groq_api_key"))
        groq_model = os.getenv("GROQ_MODEL", cls.groq_model)
        groq_base_url = os.getenv("GROQ_BASE_URL", cls.groq_base_url)
        temperature_str = os.getenv("GROQ_TEMPERATURE")
        max_tokens_str = os.getenv("GROQ_MAX_TOKENS")

        missing = [
            name
            for name, value in {
                "TELEGRAM_BOT_TOKEN": telegram_token,
                "GROQ_API_KEY": groq_api_key,
            }.items()
            if not value
        ]

        if missing:
            availability = ", ".join(
                f"{name}={'есть' if os.getenv(name) else 'нет'}"
                for name in ("TELEGRAM_BOT_TOKEN", "GROQ_API_KEY")
            )
            railway_hint = ""
            if os.getenv("RAILWAY_ENVIRONMENT") or os.getenv("RAILWAY_PROJECT_ID"):
                railway_hint = (
                    " Похоже, бот запущен на Railway. Убедитесь, что переменные заданы "
                    "в разделе Service Variables или через `railway variables set` и "
                    "после изменения выполнен новый деплой. Проверить значения можно "
                    "командой `railway variables list`."
                )

            raise RuntimeError(
                "Отсутствуют обязательные переменные окружения: "
                + ", ".join(missing)
                + f" (проверено: {availability})."
                + railway_hint
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
