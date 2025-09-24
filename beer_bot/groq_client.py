"""Client for interacting with Groq's multimodal chat completions API."""
from __future__ import annotations

import base64
import logging
from typing import Any, Dict, Optional

import httpx

LOGGER = logging.getLogger(__name__)


def _image_to_data_url(image_bytes: bytes, mime_type: str = "image/jpeg") -> str:
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


class GroqVisionClient:
    """Thin wrapper around the Groq Chat Completions endpoint."""

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        base_url: str,
        temperature: float,
        max_tokens: int,
        timeout: float = 30.0,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._base_url = base_url
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout

    async def review_beer(
        self,
        image_bytes: bytes,
        *,
        caption: Optional[str] = None,
    ) -> str:
        """Send the beer photo to Groq and return a witty review."""

        prompt = (
            "Ты – ироничный пивной сомелье. Посмотри на фото кружки или банки пива, "
            "опиши, что видишь и придумай короткий, смешной и дружелюбный отзыв. "
            "Если на фото нет пива, мягко пошути об этом."
        )

        user_content: list[Dict[str, Any]] = [
            {
                "type": "text",
                "text": caption or "Держи фото сегодняшнего крафта.",
            },
            {
                "type": "image_url",
                "image_url": {"url": _image_to_data_url(image_bytes)},
            },
        ]

        payload: Dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        LOGGER.debug("Sending request to Groq: %s", payload)

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                response = await client.post(self._base_url, headers=headers, json=payload)
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                LOGGER.error(
                    "Groq API returned %s: %s",
                    exc.response.status_code,
                    exc.response.text,
                )
                raise
            except httpx.HTTPError:
                LOGGER.exception("Failed to reach Groq API")
                raise

        data = response.json()
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as exc:  # pragma: no cover - defensive
            LOGGER.exception("Unexpected response structure from Groq: %s", data)
            raise RuntimeError("Не удалось разобрать ответ от Groq") from exc
