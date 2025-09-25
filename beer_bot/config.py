"""Configuration helpers for the Beer Wednesday bot."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional


LOGGER = logging.getLogger(__name__)


DEFAULT_POSTCARD_PROMPT = (
    "A vibrant illustrated invitation postcard for the weekly gathering"
    " 'Пивная среда'. Show a joyful, diverse group of friends enjoying beer"
    " together in a distinctive setting that matches the scene description."
    " Highlight people of different genders, backgrounds and ages, and let"
    " the scenario define the cast size and composition. Use warm cinematic"
    " lighting, plenty of amber highlights, rich textures and playful"
    " storytelling details. Keep the scene free from any visible text or"
    " lettering within the image."
)

DEFAULT_POSTCARD_NEGATIVE_PROMPT = "text, typography, lettering, captions, subtitles"

DEFAULT_POSTCARD_CAPTION = (
    "🍻 Пивная среда уже завтра! Стартуем в 19:30 — приходи пораньше и"
    " захвати друзей."
)

DEFAULT_POSTCARD_SCENARIOS = (
    "Два космонавта и одна космобиологиня парят в невесомости орбитального"
    " бара, тостуют за встречу под мягким светом Земли в иллюминаторах, стиль"
    " реалистичной научной фантастики.",
    "Пять панк-хакеров: две девушки, небинарный диджей и двое парней в"
    " киберпанковом ночном Токио сидят за неоновым стойлом с прозрачными"
    " кружками светящегося пива, вокруг голограммы и тёплый мокрый асфальт.",
    "Три эльфа-пивовара (две эльфийки и один эльф) и гном-ремесленник в"
    " фэнтезийной таверне под гигантским деревом, иллюстрация в духе эпического"
    " хай-фэнтези.",
    "Шестеро друзей разных возрастов в стилизованном плакате советской"
    " агитации 1930-х, энергично поднимают кружки в красно-золотых лучах и"
    " строгости плаката.",
    "Одинокий бард, рыцарка и молодой оруженосец в эстетике 'страдающего"
    " средневековья' сидят у костра возле замка, грустно отпивают густое пиво"
    " из глиняных кубков, стиль старинной миниатюры.",
    "Три учёных в лаборатории стимпанка: женщина-инженер, мужчина-алхимик и"
    " пожилая профессорка среди медных труб и шестерёнок исследуют янтарное"
    " пиво в колбах, графика в духе викторианских гравюр.",
    "Семь космических археологов разных культур в ретрофутуристических"
    " скафандрах празднуют находку среди руин инопланетного города, стиль"
    " pulp sci-fi обложки.",
    "Четверо художников-импрессионистов: две женщины и двое мужчин пишут"
    " закатную набережную и попивают светлый эль, сцена размазана мягкими"
    " мазками в стиле Моне.",
    "Две рок-звезды, диджей и визажистка на кибер-рейве 2080-х, неон, лазеры,"
    " хромированные кружки с пенящимся напитком, стиль глитч-арт.",
    "Компания из пяти друзей на карнавальном корабле эпохи Возрождения:"
    " маски, бархатные костюмы и золотые бокалы, картина в стиле венецианских"
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
    postcard_timezone: str = "Europe/Moscow"
    postcard_scenarios: List[str] = field(
        default_factory=lambda: list(DEFAULT_POSTCARD_SCENARIOS)
    )

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
            postcard_scenarios=list(DEFAULT_POSTCARD_SCENARIOS),
        )

    @property
    def huggingface_url(self) -> str:
        """Return the resolved Hugging Face endpoint URL for the configured model."""

        if self.huggingface_base_url:
            return self.huggingface_base_url
        return f"https://api-inference.huggingface.co/models/{self.huggingface_model}"
