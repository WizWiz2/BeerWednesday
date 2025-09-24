"""Entry point for the Beer Wednesday Telegram bot."""
from __future__ import annotations

import logging
from typing import NoReturn

from dotenv import load_dotenv
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
)

from .config import Settings
from .groq_client import GroqVisionClient
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

    application.add_handler(CommandHandler("start", handlers.start))
    application.add_handler(CommandHandler("help", handlers.help_command))
    application.add_handler(MessageHandler(filters.PHOTO, handlers.handle_photo))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handlers.handle_text))
    application.add_error_handler(handlers.error_handler)

    return application


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
