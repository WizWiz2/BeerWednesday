"""Telegram handlers for the Beer Wednesday bot."""
from __future__ import annotations

import logging
from io import BytesIO
from typing import Optional

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import ContextTypes

from .groq_client import GroqVisionClient

LOGGER = logging.getLogger(__name__)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a greeting message when the bot is started."""
    await update.message.reply_text(
        "Привет! Пришли мне фото пива, и я попрошу сомелье оставить отзыв."
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Explain how to use the bot."""
    await update.message.reply_text(
        "Скинь фото крафтового пива (можно с подписью), и я вышлю ироничный отзыв."
    )


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Process an incoming photo message."""
    if not update.message:
        return

    groq_client: Optional[GroqVisionClient] = context.application.bot_data.get("groq_client")
    if not groq_client:
        await update.message.reply_text("Groq клиент не настроен. Обратитесь к администратору.")
        return

    photo = update.message.photo[-1] if update.message.photo else None
    if not photo:
        await update.message.reply_text("Не вижу фото. Попробуй отправить ещё раз.")
        return

    caption = update.message.caption

    image_buffer = BytesIO()
    telegram_file = await context.bot.get_file(photo.file_id)
    await telegram_file.download_to_memory(image_buffer)
    image_bytes = image_buffer.getvalue()

    try:
        is_beer = await groq_client.is_beer_photo(image_bytes, caption=caption)
    except Exception:  # pragma: no cover - runtime guard
        LOGGER.exception("Failed to detect beer in photo")
        return

    if not is_beer:
        LOGGER.info("Ignoring non-beer photo from chat %s", update.effective_chat.id)
        return

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)

    try:
        review = await groq_client.review_beer(image_bytes, caption=caption)
    except Exception as exc:  # pragma: no cover - runtime guard
        LOGGER.exception("Failed to get review from Groq")
        await update.message.reply_text(
            "Не удалось получить отзыв от сомелье. Попробуй позже."
        )
        return

    await update.message.reply_text(review)


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Ignore plain text messages to avoid spam."""
    if not update.message:
        return


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:  # pragma: no cover
    """Log errors raised by handlers."""
    LOGGER.error("Update %s caused error %s", update, context.error)
