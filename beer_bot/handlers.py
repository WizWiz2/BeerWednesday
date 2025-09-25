"""Telegram handlers for the Beer Wednesday bot."""
from __future__ import annotations

import logging
from io import BytesIO
from typing import Optional

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import ContextTypes

from .config import DEFAULT_POSTCARD_PROMPT
from .groq_client import GroqVisionClient
from .postcard_client import HuggingFacePostcardClient

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


async def postcard_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Generate a Beer Wednesday invitation postcard on demand."""

    if not update.message:
        return

    prompt_base: str = context.application.bot_data.get("postcard_prompt") or DEFAULT_POSTCARD_PROMPT
    extra = update.message.text.partition(" ")[2].strip() if update.message.text else ""

    if extra:
        prompt = f"{prompt_base}\nДополнительные пожелания: {extra}"
    else:
        prompt = prompt_base

    await update.message.reply_chat_action(action=ChatAction.UPLOAD_PHOTO)

    await _send_postcard(
        chat_id=update.effective_chat.id,
        context=context,
        prompt=prompt,
        reply_to_message_id=update.message.message_id,
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


async def scheduled_postcard_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send the weekly Beer Wednesday postcard to the configured chat."""

    if not context.job or context.job.chat_id is None:
        LOGGER.warning("Postcard job triggered without chat_id; skipping")
        return

    prompt = ""
    if context.job.data and isinstance(context.job.data, dict):
        prompt = context.job.data.get("prompt", "")

    await context.bot.send_chat_action(
        chat_id=context.job.chat_id,
        action=ChatAction.UPLOAD_PHOTO,
    )

    await _send_postcard(
        chat_id=context.job.chat_id,
        context=context,
        prompt=prompt
        or context.application.bot_data.get("postcard_prompt")
        or DEFAULT_POSTCARD_PROMPT,
    )


async def _send_postcard(
    *,
    chat_id: int,
    context: ContextTypes.DEFAULT_TYPE,
    prompt: str,
    reply_to_message_id: Optional[int] = None,
) -> None:
    """Generate postcard and send it to the specified chat."""

    client: Optional[HuggingFacePostcardClient] = context.application.bot_data.get(
        "postcard_client"
    )

    if not client:
        LOGGER.warning("Postcard client is not configured")
        if reply_to_message_id:
            await context.bot.send_message(
                chat_id=chat_id,
                text="Генерация открыток недоступна: нет доступа к Hugging Face API.",
                reply_to_message_id=reply_to_message_id,
            )
        else:
            await context.bot.send_message(
                chat_id=chat_id,
                text="Генерация открыток временно недоступна.",
            )
        return

    negative_prompt: Optional[str] = context.application.bot_data.get(
        "postcard_negative_prompt"
    )
    caption: str = context.application.bot_data.get("postcard_caption", "")

    try:
        image_bytes = await client.generate_postcard(
            prompt,
            negative_prompt=negative_prompt,
        )
    except Exception:  # pragma: no cover - runtime guard
        LOGGER.exception("Failed to generate postcard")
        fail_text = "Не получилось сгенерировать открытку, попробуй позже."
        if reply_to_message_id:
            await context.bot.send_message(
                chat_id=chat_id,
                text=fail_text,
                reply_to_message_id=reply_to_message_id,
            )
        else:
            await context.bot.send_message(chat_id=chat_id, text=fail_text)
        return

    await context.bot.send_photo(
        chat_id=chat_id,
        photo=image_bytes,
        caption=caption,
        reply_to_message_id=reply_to_message_id,
    )
