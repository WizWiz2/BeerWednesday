"""Telegram handlers for the Beer Wednesday bot."""
from __future__ import annotations

import logging
from io import BytesIO
from typing import Dict, Optional

from telegram import Message, Update
from telegram.constants import ChatAction, ChatType, MessageEntityType
from telegram.ext import ContextTypes

from .config import DEFAULT_POSTCARD_PROMPT
from .groq_client import GroqVisionClient
from .postcard_client import HuggingFacePostcardClient

LOGGER = logging.getLogger(__name__)

ATTENDANCE_GOING_OPTION_INDEX = 0
ATTENDANCE_THRESHOLD = 5
ATTENDANCE_STORAGE_KEY = "attendance_polls"
POSTCARD_SCENARIOS_KEY = "postcard_scenarios"
POSTCARD_SCENARIO_INDEX_KEY = "postcard_scenario_index"


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

    prompt_base: str = (
        context.application.bot_data.get("postcard_prompt") or DEFAULT_POSTCARD_PROMPT
    )
    extra = update.message.text.partition(" ")[2].strip() if update.message.text else ""

    prompt = _compose_postcard_prompt(context, prompt_base, extra)

    await update.message.reply_chat_action(action=ChatAction.UPLOAD_PHOTO)

    postcard_sent = await _send_postcard(
        chat_id=update.effective_chat.id,
        context=context,
        prompt=prompt,
        reply_to_message_id=update.message.message_id,
    )

    if postcard_sent:
        await _start_attendance_poll(chat_id=update.effective_chat.id, context=context)


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
    """Answer beer questions when the bot is addressed directly."""
    message = update.message
    if not message:
        return

    bot_username = getattr(context.bot, "username", None)
    bot_id = getattr(context.bot, "id", None)

    if not _is_direct_engagement(message, bot_username, bot_id):
        return

    await _respond_as_sommelier(message, context, bot_username)


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

    base_prompt = (
        prompt
        or context.application.bot_data.get("postcard_prompt")
        or DEFAULT_POSTCARD_PROMPT
    )

    postcard_prompt = _compose_postcard_prompt(context, base_prompt)

    postcard_sent = await _send_postcard(
        chat_id=context.job.chat_id,
        context=context,
        prompt=postcard_prompt,
    )

    if postcard_sent:
        await _start_attendance_poll(chat_id=context.job.chat_id, context=context)


def _is_direct_engagement(
    message: Message, bot_username: Optional[str], bot_id: Optional[int]
) -> bool:
    """Return whether the message is addressed to the bot directly."""

    if message.chat.type == ChatType.PRIVATE:
        return True

    if bot_username and _mentions_bot(message, bot_username):
        return True

    if (
        bot_id
        and message.reply_to_message
        and message.reply_to_message.from_user
        and message.reply_to_message.from_user.id == bot_id
    ):
        return True

    return False


def _mentions_bot(message: Message, bot_username: str) -> bool:
    """Check whether the message mentions the bot by username."""

    if not message.entities or not message.text:
        return False

    mention = f"@{bot_username.lower()}"
    for entity in message.entities:
        if entity.type != MessageEntityType.MENTION:
            continue
        entity_text = message.text[entity.offset : entity.offset + entity.length]
        if entity_text.lower() == mention:
            return True

    return False


def _extract_question_text(message: Message, bot_username: Optional[str]) -> str:
    """Remove the bot mention from the message text and trim it."""

    text = message.text or ""
    if not text:
        return ""

    if not bot_username or not message.entities:
        return text.strip()

    cleaned = text
    lower_username = bot_username.lower()
    for entity in message.entities:
        if entity.type != MessageEntityType.MENTION:
            continue
        entity_text = text[entity.offset : entity.offset + entity.length]
        if entity_text.lower() == f"@{lower_username}":
            cleaned = (text[: entity.offset] + text[entity.offset + entity.length :]).strip()
            break

    return cleaned.strip()


async def _respond_as_sommelier(
    message: Message, context: ContextTypes.DEFAULT_TYPE, bot_username: Optional[str]
) -> None:
    """Generate a sommelier-style answer about beer."""

    groq_client: Optional[GroqVisionClient] = context.application.bot_data.get(
        "groq_client"
    )
    if not groq_client:
        await message.reply_text("Groq клиент не настроен. Обратитесь к администратору.")
        return

    question = _extract_question_text(message, bot_username)
    if not question:
        await message.reply_text("Спроси меня о пиве — с радостью отвечу!")
        return

    await context.bot.send_chat_action(
        chat_id=message.chat_id,
        action=ChatAction.TYPING,
    )

    try:
        answer = await groq_client.answer_beer_question(question)
    except Exception:  # pragma: no cover - runtime guard
        LOGGER.exception("Failed to answer beer question")
        await message.reply_text("Не удалось обсудить пиво, попробуй позже.")
        return

    await message.reply_text(answer)


def _compose_postcard_prompt(
    context: ContextTypes.DEFAULT_TYPE, base_prompt: str, extra: str = ""
) -> str:
    """Combine base prompt with rotating scenarios and user extras."""

    parts = [base_prompt.strip()]

    scenario = _pop_next_postcard_scenario(context)
    if scenario:
        parts.append(f"Сценарий: {scenario}")

    if extra:
        parts.append(f"Дополнительные пожелания: {extra}")

    return "\n\n".join(parts)


def _pop_next_postcard_scenario(context: ContextTypes.DEFAULT_TYPE) -> str:
    """Return the next postcard scenario and advance the rotation."""

    application = getattr(context, "application", None)
    if not application:
        return ""

    scenarios = application.bot_data.get(POSTCARD_SCENARIOS_KEY)
    if not scenarios:
        return ""

    index = application.bot_data.get(POSTCARD_SCENARIO_INDEX_KEY, 0)
    scenario = scenarios[index % len(scenarios)]
    application.bot_data[POSTCARD_SCENARIO_INDEX_KEY] = (index + 1) % len(scenarios)
    return scenario


async def _send_postcard(
    *,
    chat_id: int,
    context: ContextTypes.DEFAULT_TYPE,
    prompt: str,
    reply_to_message_id: Optional[int] = None,
) -> bool:
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
        return False

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
        return False

    await context.bot.send_photo(
        chat_id=chat_id,
        photo=image_bytes,
        caption=caption,
        reply_to_message_id=reply_to_message_id,
    )

    return True


async def _start_attendance_poll(
    *, chat_id: int, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Send a poll asking who plans to join Beer Wednesday."""

    poll_message = await context.bot.send_poll(
        chat_id=chat_id,
        question="Кто идёт на пивную среду?",
        options=[
            "Я иду",
            "Ещё не решил",
            "Не смогу",
        ],
        is_anonymous=False,
        allows_multiple_answers=False,
    )

    if not poll_message.poll:
        LOGGER.warning("Attendance poll was sent without poll payload")
        return

    poll_state: Dict[str, Dict[str, object]] = context.application.bot_data.setdefault(
        ATTENDANCE_STORAGE_KEY, {}
    )

    poll_state[poll_message.poll.id] = {
        "chat_id": chat_id,
        "message_id": poll_message.message_id,
        "notified": False,
        "votes": {},
    }


async def handle_poll_answer(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Track poll answers and remind to reserve a table when needed."""

    if not update.poll_answer:
        return

    poll_data = context.bot_data.get(ATTENDANCE_STORAGE_KEY)
    if not poll_data:
        return

    poll_state = poll_data.get(update.poll_answer.poll_id)
    if not poll_state:
        return

    votes: Dict[int, list[int]] = poll_state.setdefault("votes", {})
    votes[update.poll_answer.user.id] = update.poll_answer.option_ids

    going_count = sum(
        1
        for selected in votes.values()
        if ATTENDANCE_GOING_OPTION_INDEX in selected
    )

    if going_count >= ATTENDANCE_THRESHOLD and not poll_state.get("notified"):
        poll_state["notified"] = True
        await context.bot.send_message(
            chat_id=poll_state["chat_id"],
            text=(
                f"Нас уже {going_count}! "
                "Пора бронировать стол 🍻"
            ),
        )
