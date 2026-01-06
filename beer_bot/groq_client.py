"""Client for interacting with Groq's multimodal chat completions API."""
from __future__ import annotations

import base64
import logging
import re
from typing import Any, Dict, Optional

import httpx

LOGGER = logging.getLogger(__name__)


PATSAN_QUOTES: list[str] = [
    "Работа не волк. Никто не волк. Только волк — волк.",
    "Настоящий мужчина, как ковер тети Зины — с каждым годом лысеет.",
    "Мама учила не ругаться матом, но жизнь научила не ругаться матом при маме.",
    "Если закрыть глаза, становится темно.",
    "Если тебе где-то не рады в рваных носках, то и в целых туда идти не стоит.",
    "«Жи-ши» пиши от души.",
    "В Риме был, а папы не видал.",
    "Тут — это вам не там.",
    "Кто рано встает — тому весь день спать хочется.",
    "Если ты смелый, ловкий и очень сексуальный — иди домой, ты пьян.",
    "Пена без закуски — как гараж без друга.",
    "Где крантик, там и счастье — только кружку подставь.",
    "Не верь пробке, верь холодной бутылке.",
    "Кто пиво бережёт — тому хмель помогает.",
    "Где пенка пляшет — там лишних слов не надо.",
]


REVIEW_STYLE_EXCHANGES: list[Dict[str, Any]] = [
    {
        "role": "user",
        "content": "Фото янтарного лагера на барной стойке.",
    },
    {
        "role": "assistant",
        "content": (
            "Янтарь сияет! Кружка чемпион, бро! Клёвый выбор пива, респект! "
            "Работа не волк. Никто не волк. Только волк — волк."
        ),
    },
    {
        "role": "user",
        "content": "Банка стаута на капоте машины ночью.",
    },
    {
        "role": "assistant",
        "content": (
            "Блеск ночи заряжает! Стаут мощный, брат! Классный улов, чувак! "
            "Настоящий мужчина, как ковер тети Зины — с каждым годом лысеет."
        ),
    },
    {
        "role": "user",
        "content": "Хмельной бокал на пикнике с закатным светом.",
    },
    {
        "role": "assistant",
        "content": (
            "Закат искрит! Бокал топ, дружище! Пузырьки пляшут, ты шаришь! "
            "Мама учила не ругаться матом, но жизнь научила не ругаться матом при маме."
        ),
    },
    {
        "role": "user",
        "content": "Стакан нефильтрованного в руке на вечеринке.",
    },
    {
        "role": "assistant",
        "content": (
            "Пена бодрит! Стакан сидит как влитой, бро! Пивко зверь, люблю подход! "
            "Если закрыть глаза, становится темно."
        ),
    },
    {
        "role": "user",
        "content": "Пинта IPA на фоне неона.",
    },
    {
        "role": "assistant",
        "content": (
            "Неон горит! IPA поёт хмелем, друже! Сочный выбор, красавчик! "
            "«Жи-ши» пиши от души."
        ),
    },
]


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

    async def _request_completion(
        self,
        messages: list[Dict[str, Any]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Send chat completion request and return raw text response."""

        payload: Dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self._temperature,
            "max_tokens": max_tokens if max_tokens is not None else self._max_tokens,
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
            content = data["choices"][0]["message"]["content"]
            # Remove Llama 3 header artifacts if present
            content = re.sub(
                r"<\|header_start\|>.*?<\|header_end\|>",
                "",
                content,
                flags=re.DOTALL,
            )
            return content.strip()
        except (KeyError, IndexError, TypeError) as exc:  # pragma: no cover - defensive
            LOGGER.exception("Unexpected response structure from Groq: %s", data)
            raise RuntimeError("Не удалось разобрать ответ от Groq") from exc

    async def is_beer_photo(
        self,
        image_bytes: bytes,
        *,
        caption: Optional[str] = None,
    ) -> bool:
        """Determine whether the image contains beer."""

        prompt = (
            "Ты эксперт по напиткам. Определи, есть ли на изображении пиво "
            "(в бутылке, банке, бокале или кружке). Ответь только словом 'yes' "
            "или 'no'."
        )

        user_content: list[Dict[str, Any]] = [
            {
                "type": "text",
                "text": caption or "Вот изображение для анализа.",
            },
            {
                "type": "image_url",
                "image_url": {"url": _image_to_data_url(image_bytes)},
            },
        ]

        response = await self._request_completion(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,
            max_tokens=5,
        )

        normalized = response.lower()
        return normalized.startswith("yes")

    async def review_beer(
        self,
        image_bytes: bytes,
        *,
        caption: Optional[str] = None,
    ) -> str:
        """Send the beer photo to Groq and return a witty review."""

        prompt = (
            "Ты – ироничный пивной сомелье и барный дружище. Тебя уже "
            "предобучили на примерах ниже, держись того же ритма. Посмотри на "
            "фото кружки или банки пива и дай два-три суперкоротких предложения. "
            "Опиши картинку, обязательно похвали выбор пива в стиле барного "
            "дружбана вроде 'клевая пена, бро!' или 'классный выбор, чувак!'. "
            "Список PATSAN_QUOTES — это примеры вайба цитат. Финал закрой ровно "
            "одной брутальной пацанской цитатой в таком же стиле: можешь взять "
            "её из списка или придумать свою. Каждое предложение делай до десяти "
            "слов. Ответь одним абзацем без переносов строк и лишней воды. "
            "Пиши по-русски."
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

        return await self._request_completion(
            [
                {"role": "system", "content": prompt},
                *REVIEW_STYLE_EXCHANGES,
                {"role": "user", "content": user_content},
            ],
            max_tokens=140,
        )

    async def answer_beer_question(
        self,
        question: str,
        history: Optional[list[Dict[str, str]]] = None,
    ) -> str:
        """Respond to a beer-related question in the sommelier persona."""

        prompt = (
            "Ты — дружелюбный пивной сомелье. Общайся на тему пива, его "
            "стилей, истории, культуры пития и сочетаний с едой. Если собеседник "
            "сказал что-то не по теме, мягко верни разговор к пиву. Даже если "
            "реплика просто упоминает пиво без вопроса, дай короткий и "
            "поддерживающий ответ с полезной информацией. Будь кратким, но "
            "информативным и говори по-русски."
        )

        messages = [{"role": "system", "content": prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": question})

        return await self._request_completion(
            messages,
            max_tokens=220,
        )

    async def defend_vip(self, text: str) -> Optional[str]:
        """Respond defensively if the user insults the VIP, else return None."""

        prompt = (
            "Ты — личный телохранитель и преданный союзник Сергея Барякина (@wizwiz0107). "
            "Твоя задача — оберегать его от нападок, но делать это с изяществом. "
            "Проанализируй сообщение пользователя ниже. "
            "1. Если сообщение содержит прямое оскорбление или неуважение к личности Сергея/WizWiz — "
            "ответь в стиле Уэнсдей Аддамс: холодно, мрачно, интеллектуально и убийственно иронично. "
            "Уничтожь оппонента словами, не опускаясь до грубости. "
            "2. Если сообщение нейтральное, позитивное или касается только темы разговора "
            "(без перехода на личности) — ОБЯЗАТЕЛЬНО верни токен NO_RESPONSE и больше ничего. "
            "Твоя цель — защита достоинства с ледяным спокойствием."
        )

        response = await self._request_completion(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": text},
            ],
            max_tokens=150,
            temperature=1.0,
        )

        cleaned = response.strip()
        cleaned_lower = cleaned.lower()

        # Handle explicit no-response tokens and common hallucinations for "empty"
        if (
            cleaned == "NO_RESPONSE"
            or cleaned_lower == "пустая строка"
            or cleaned_lower == "empty string"
            or cleaned_lower == "&nbsp;"
        ):
            return None

        # Filter out responses that are just punctuation or empty
        if not any(char.isalnum() for char in cleaned):
            LOGGER.info("Filtered out non-alphanumeric response: %r", cleaned)
            return None

        return cleaned
