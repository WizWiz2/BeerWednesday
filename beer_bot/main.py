"""Entry point for the Beer Wednesday Telegram bot."""
from __future__ import annotations

import logging
from datetime import time
from typing import NoReturn
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    PollAnswerHandler,
    filters,
)

from .config import Settings
from .groq_client import GroqVisionClient
from .postcard_client import HuggingFacePostcardClient
from . import handlers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
LOGGER = logging.getLogger(__name__)


def _build_application(settings: Settings) -> Application:
    """Create the telegram application with all handlers configured."""

    application = ApplicationBuilder().token(settings.telegram_token).build()

    groq_client = GroqVisionClient(
        api_key=settings.groq_api_key,
        model=settings.groq_model,
        base_url=settings.groq_base_url,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
    )

    application.bot_data["groq_client"] = groq_client

    application.bot_data["postcard_prompt"] = settings.postcard_prompt
    application.bot_data["postcard_negative_prompt"] = settings.postcard_negative_prompt
    application.bot_data["postcard_caption"] = settings.postcard_caption
    application.bot_data["postcard_scenarios"] = settings.postcard_scenarios
    application.bot_data["postcard_scenario_index"] = 0

    if settings.huggingface_api_token:
        postcard_client = HuggingFacePostcardClient(
            api_token=settings.huggingface_api_token,
            model=settings.huggingface_model,
            base_url=settings.huggingface_base_url,
        )
        application.bot_data["postcard_client"] = postcard_client
    else:
        LOGGER.warning(
            "HUGGINGFACE_API_TOKEN не задан — генерация открыток будет недоступна."
        )

    application.add_handler(CommandHandler("start", handlers.start))
    application.add_handler(CommandHandler("help", handlers.help_command))
    application.add_handler(CommandHandler("chatid", handlers.chat_id_command))
    application.add_handler(CommandHandler("postcard", handlers.postcard_command))
    application.add_handler(CommandHandler("debug_postcards", handlers.debug_postcards_command))
    application.add_handler(MessageHandler(filters.PHOTO, handlers.handle_photo))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handlers.handle_text))
    application.add_handler(PollAnswerHandler(handlers.handle_poll_answer))
    application.add_error_handler(handlers.error_handler)

    _schedule_weekly_postcard(application, settings)

    return application


def _schedule_weekly_postcard(application: Application, settings: Settings) -> None:
    """Register a weekly job that sends the Beer Wednesday postcard."""

    if not settings.postcard_chat_id:
        LOGGER.warning(
            "POSTCARD_CHAT_ID не задан — еженедельная отправка открытки отключена."
        )
        return

    if application.job_queue is None:
        LOGGER.warning(
            "Job queue не инициализирован — еженедельная рассылка открыток отключена."
        )
        LOGGER.warning(
            "Убедитесь, что python-telegram-bot установлен с extra 'job-queue'."
        )
        return

    if "postcard_client" not in application.bot_data:
        LOGGER.warning(
            "Postcard client не сконфигурирован — пропускаем расписание открыток."
        )
        return

    try:
        tzinfo = ZoneInfo(settings.postcard_timezone)
    except Exception:  # pragma: no cover - defensive branch
        LOGGER.exception(
            "Не удалось определить таймзону '%s' для расписания открыток.",
            settings.postcard_timezone,
        )
        return

    application.job_queue.run_daily(
        handlers.scheduled_postcard_job,
        time=time(
            hour=settings.postcard_hour,
            minute=settings.postcard_minute,
            tzinfo=tzinfo,
        ),
        days=(settings.postcard_weekday,),  # 0=Sunday, 6=Saturday
        data={"prompt": settings.postcard_prompt},
        name="weekly_beer_postcard",
        chat_id=settings.postcard_chat_id,
    )
    LOGGER.info(
        "Weekly postcard job scheduled for chat %s at %02d:%02d %s (weekday=%s, 0=Sunday).",
        settings.postcard_chat_id,
        settings.postcard_hour,
        settings.postcard_minute,
        settings.postcard_timezone,
        settings.postcard_weekday,
    )


def run_bot() -> None:
    """Initialise and start the Telegram bot."""
    load_dotenv()
    settings = Settings.load()

    application = _build_application(settings)

    LOGGER.info("Bot is running. Press Ctrl+C to stop.")
    application.run_polling()


def main() -> NoReturn:  # pragma: no cover - thin wrapper
    """Synchronously run the bot."""
    try:
        run_bot()
    except KeyboardInterrupt:
        LOGGER.info("Bot stopped by user")


if __name__ == "__main__":  # pragma: no cover
    main()
