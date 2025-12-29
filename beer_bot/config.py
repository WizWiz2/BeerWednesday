"""Configuration helpers for the Beer Wednesday bot."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional


LOGGER = logging.getLogger(__name__)


DEFAULT_POSTCARD_PROMPT = (
    "A vibrant illustrated invitation postcard for a weekly get-together called"
    " 'Пивная среда'. Capture a cozy bar in the evening with a group of male"
    " friends smiling and clinking tall beer glasses. Use warm cinematic lighting,"
    " lots of amber highlights, wood textures and playful details. Keep the scene"
    " free from any visible text or lettering within the image."
)

DEFAULT_POSTCARD_NEGATIVE_PROMPT = (
    "text, typography, lettering, captions, subtitles, female, woman, girl"
)

DEFAULT_POSTCARD_CAPTION = (
    "🍻 Пивная среда уже завтра! Стартуем в 19:30 — приходи пораньше и"
    " захвати друзей."
)

DEFAULT_BARGHOPPING_PROMPT = (
    "A stylish illustrated invitation postcard for a monthly barhopping night"
    " called 'Бархоппинг'. Show friends moving between atmospheric bars,"
    " comparing cocktails and craft beer under the evening lights of a vibrant"
    " city street. Keep the mood adventurous yet cozy and avoid any visible"
    " lettering in the scene."
)

DEFAULT_BARGHOPPING_NEGATIVE_PROMPT = DEFAULT_POSTCARD_NEGATIVE_PROMPT

DEFAULT_BARGHOPPING_CAPTION = (
    "🍹 Бархоппинг уже совсем скоро! Собираемся на закате, чтобы пройтись по"
    " любимым барам и открыть новые."
)

DEFAULT_POSTCARD_SCENARIOS = (
    "Два космонавта и одна космобиологиня парят в невесомости орбитального"
    " бара, тостуют за встречу под мягким светом Земли в иллюминаторах, стиль"
    " реалистичной научной фантастики.",
    "Пять панк-хакеров в киберпанковом ночном Токио сидят за неоновым"
    " стойлом с прозрачными кружками светящегося пива, вокруг голограммы и"
    " тёплый мокрый асфальт.",
    "Три эльфа-пивовара и гном-ремесленник в фэнтезийной таверне под"
    " гигантским деревом, иллюстрация в духе эпического хай-фэнтези.",
    "Шестеро друзей в стилизованном плакате советской агитации 1930-х,"
    " энергично поднимают кружки в красно-золотых лучах и строгости плаката.",
    "Одинокий бард и два рыцаря в эстетике 'страдающего средневековья'"
    " сидят у костра возле замка, грустно отпивают густое пиво из глиняных"
    " кубков, стиль старинной миниатюры.",
    "Три учёных в лаборатории стимпанка среди медных труб и шестерёнок"
    " исследуют янтарное пиво в колбах, графика в духе викторианских гравюр.",
    "Семь космических археологов в ретрофутуристических скафандрах празднуют"
    " находку среди руин инопланетного города, стиль pulp sci-fi обложки.",
    "Четверо художников-импрессионистов пишут закатную набережную и попивают"
    " светлый эль, сцена размазана мягкими мазками в стиле Моне.",
    "Две рок-звезды и диджей на кибер-рейве 2080-х, неон, лазеры, хромированные"
    " кружки с пенящимся напитком, стиль глитч-арт.",
    "Компания из пяти друзей на карнавальном корабле эпохи Возрождения,"
    " мраморные колонны, маски и золотые бокалы, картина в стиле венецианских"
    " мастеров.",
)


@dataclass(frozen=True)
class Settings:
    """Runtime configuration loaded from environment variables."""

    telegram_token: str
    groq_api_key: str
    groq_model: str = "llama-3.2-11b-vision"
    groq_base_url: str = "https://api.groq.com/openai/v1/chat/completions"
    temperature: float = 0.7
    max_tokens: int = 1024
    huggingface_api_token: Optional[str] = None
    huggingface_model: str = "black-forest-labs/FLUX.1-dev"
    huggingface_base_url: Optional[str] = None
    postcard_chat_id: Optional[int] = None
    postcard_prompt: str = DEFAULT_POSTCARD_PROMPT
    postcard_negative_prompt: Optional[str] = DEFAULT_POSTCARD_NEGATIVE_PROMPT
    postcard_caption: str = DEFAULT_POSTCARD_CAPTION
    postcard_timezone: str = "Asia/Almaty"
    postcard_weekday: int = 2
    postcard_hour: int = 21
    postcard_minute: int = 0
    postcard_scenarios: List[str] = field(
        default_factory=lambda: list(DEFAULT_POSTCARD_SCENARIOS)
    )
    barhopping_chat_id: Optional[int] = None
    barhopping_prompt: str = DEFAULT_BARGHOPPING_PROMPT
    barhopping_negative_prompt: Optional[str] = DEFAULT_BARGHOPPING_NEGATIVE_PROMPT
    barhopping_caption: str = DEFAULT_BARGHOPPING_CAPTION
    barhopping_timezone: str = "Asia/Almaty"
    barhopping_hour: int = 12
    barhopping_minute: int = 0
    barhopping_poll_question: str = "Кто идёт на бархоппинг?"

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
        huggingface_api_token = os.getenv("HUGGINGFACE_API_TOKEN")
        huggingface_model = os.getenv("HUGGINGFACE_MODEL", cls.huggingface_model)
        huggingface_base_url = os.getenv("HUGGINGFACE_BASE_URL")
        postcard_chat_id_raw = os.getenv("POSTCARD_CHAT_ID")
        postcard_prompt = os.getenv("POSTCARD_PROMPT", cls.postcard_prompt)
        postcard_negative_prompt = os.getenv(
            "POSTCARD_NEGATIVE_PROMPT", cls.postcard_negative_prompt or ""
        )
        postcard_caption = os.getenv("POSTCARD_CAPTION", cls.postcard_caption)
        postcard_timezone = os.getenv("POSTCARD_TIMEZONE", cls.postcard_timezone)
        postcard_weekday_raw = os.getenv("POSTCARD_WEEKDAY")
        postcard_hour_raw = os.getenv("POSTCARD_HOUR")
        postcard_minute_raw = os.getenv("POSTCARD_MINUTE")
        barhopping_chat_id_raw = os.getenv("BARHOPPING_CHAT_ID") or os.getenv(
            "BARGHOPPING_CHAT_ID"
        )
        barhopping_prompt = os.getenv("BARHOPPING_PROMPT") or os.getenv(
            "BARGHOPPING_PROMPT", cls.barhopping_prompt
        )
        barhopping_negative_prompt = os.getenv("BARHOPPING_NEGATIVE_PROMPT") or os.getenv(
            "BARGHOPPING_NEGATIVE_PROMPT", cls.barhopping_negative_prompt or ""
        )
        barhopping_caption = os.getenv("BARHOPPING_CAPTION") or os.getenv(
            "BARGHOPPING_CAPTION", cls.barhopping_caption
        )
        barhopping_timezone = os.getenv("BARHOPPING_TIMEZONE") or os.getenv(
            "BARGHOPPING_TIMEZONE", cls.barhopping_timezone
        )
        barhopping_hour_raw = os.getenv("BARHOPPING_HOUR") or os.getenv(
            "BARGHOPPING_HOUR"
        )
        barhopping_minute_raw = os.getenv("BARHOPPING_MINUTE") or os.getenv(
            "BARGHOPPING_MINUTE"
        )
        barhopping_poll_question = os.getenv("BARHOPPING_POLL_QUESTION") or os.getenv(
            "BARGHOPPING_POLL_QUESTION", cls.barhopping_poll_question
        )

        deprecated_models = {
            "llava-v1.5-7b-4096-preview": cls.groq_model,
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

        postcard_chat_id: Optional[int]
        if postcard_chat_id_raw:
            try:
                postcard_chat_id = int(postcard_chat_id_raw)
            except ValueError:
                LOGGER.error(
                    "POSTCARD_CHAT_ID должно быть целым числом, получили '%s'",
                    postcard_chat_id_raw,
                )
                postcard_chat_id = None
        else:
            postcard_chat_id = None

        barhopping_chat_id: Optional[int]
        if barhopping_chat_id_raw:
            try:
                barhopping_chat_id = int(barhopping_chat_id_raw)
            except ValueError:
                LOGGER.error(
                    "BARGHOPPING_CHAT_ID должно быть целым числом, получили '%s'",
                    barhopping_chat_id_raw,
                )
                barhopping_chat_id = None
        else:
            barhopping_chat_id = postcard_chat_id

        def _parse_int(
            raw_value: Optional[str],
            *,
            name: str,
            default: int,
            minimum: int,
            maximum: int,
        ) -> int:
            if raw_value is None:
                return default

            try:
                parsed = int(raw_value)
            except ValueError:
                LOGGER.error(
                    "%s должно быть целым числом, получили '%s'",
                    name,
                    raw_value,
                )
                return default

            if parsed < minimum or parsed > maximum:
                LOGGER.error(
                    "%s должно быть в диапазоне %s–%s, получили '%s'",
                    name,
                    minimum,
                    maximum,
                    raw_value,
                )
                return default

            return parsed

        postcard_weekday = _parse_int(
            postcard_weekday_raw,
            name="POSTCARD_WEEKDAY",
            default=cls.postcard_weekday,
            minimum=0,
            maximum=6,
        )
        postcard_hour = _parse_int(
            postcard_hour_raw,
            name="POSTCARD_HOUR",
            default=cls.postcard_hour,
            minimum=0,
            maximum=23,
        )
        postcard_minute = _parse_int(
            postcard_minute_raw,
            name="POSTCARD_MINUTE",
            default=cls.postcard_minute,
            minimum=0,
            maximum=59,
        )

        barhopping_hour = _parse_int(
            barhopping_hour_raw,
            name="BARGHOPPING_HOUR",
            default=cls.barhopping_hour,
            minimum=0,
            maximum=23,
        )
        barhopping_minute = _parse_int(
            barhopping_minute_raw,
            name="BARGHOPPING_MINUTE",
            default=cls.barhopping_minute,
            minimum=0,
            maximum=59,
        )

        return cls(
            telegram_token=telegram_token,
            groq_api_key=groq_api_key,
            groq_model=groq_model,
            groq_base_url=groq_base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            huggingface_api_token=huggingface_api_token,
            huggingface_model=huggingface_model,
            huggingface_base_url=huggingface_base_url,
            postcard_chat_id=postcard_chat_id,
            postcard_prompt=postcard_prompt,
            postcard_negative_prompt=postcard_negative_prompt or None,
            postcard_caption=postcard_caption,
            postcard_timezone=postcard_timezone,
            postcard_weekday=postcard_weekday,
            postcard_hour=postcard_hour,
            postcard_minute=postcard_minute,
            postcard_scenarios=list(DEFAULT_POSTCARD_SCENARIOS),
            barhopping_chat_id=barhopping_chat_id,
            barhopping_prompt=barhopping_prompt,
            barhopping_negative_prompt=barhopping_negative_prompt or None,
            barhopping_caption=barhopping_caption,
            barhopping_timezone=barhopping_timezone,
            barhopping_hour=barhopping_hour,
            barhopping_minute=barhopping_minute,
            barhopping_poll_question=barhopping_poll_question,
        )

    @property
    def huggingface_url(self) -> str:
        """Return the resolved Hugging Face endpoint URL for the configured model."""

        if self.huggingface_base_url:
            return self.huggingface_base_url
        return f"https://api-inference.huggingface.co/models/{self.huggingface_model}"
