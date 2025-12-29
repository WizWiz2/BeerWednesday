"""Telegram handlers for the Beer Wednesday bot."""
from __future__ import annotations

import logging
import re
from datetime import date, datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional

from telegram import Message, Update
from telegram.constants import ChatAction, ChatType, MessageEntityType
from telegram.ext import ContextTypes

from zoneinfo import ZoneInfo

from .config import DEFAULT_BARGHOPPING_PROMPT, DEFAULT_POSTCARD_PROMPT
from .groq_client import GroqVisionClient
from .postcard_client import (
    BARGHOPPING_POSTCARD_PLACEHOLDER_PATH,
    BEER_POSTCARD_PLACEHOLDER_PATH,
    HuggingFacePostcardClient,
)
from .memory import ConversationManager

LOGGER = logging.getLogger(__name__)

ATTENDANCE_GOING_OPTION_INDEX = 0
ATTENDANCE_THRESHOLD = 5
ATTENDANCE_STORAGE_KEY = "attendance_polls"
POSTCARD_SCENARIOS_KEY = "postcard_scenarios"
POSTCARD_SCENARIO_INDEX_KEY = "postcard_scenario_index"
DEBUG_POSTCARDS_JOB_KEY = "debug_postcards_job"
DEBUG_POSTCARDS_INTERVAL_SECONDS = 5 * 60

DEFAULT_ATTENDANCE_OPTIONS = [
    "Я иду",
    "Ещё не решил",
    "Не смогу",
]
DEFAULT_BEER_POLL_QUESTION = "Кто идёт на пивную среду?"
DEFAULT_BARGHOPPING_POLL_QUESTION = "Кто идёт на бархоппинг?"


def _load_placeholder_postcard(*, path: Path) -> Optional[bytes]:
    """Read the bundled placeholder postcard image from disk."""

    try:
        return path.read_bytes()
    except FileNotFoundError:
        LOGGER.error("Placeholder postcard file is missing at %s", path)
    except OSError:  # pragma: no cover - defensive branch
        LOGGER.exception("Failed to read placeholder postcard from %s", path)

    return None


def _debug_postcards_state_message(*, enabled: bool) -> str:
    """Return a short sentence describing the debug postcards state."""

    state = "включена" if enabled else "отключена"
    return f"Текущее состояние: {state}."


def _is_debug_postcards_enabled(context: ContextTypes.DEFAULT_TYPE, job_name: str) -> bool:
    """Best-effort detection of whether the debug postcards job is active."""

    if context.job_queue:
        jobs = context.job_queue.get_jobs_by_name(job_name)
        if any(not job.removed for job in jobs):
            return True

    stored_job_name = context.chat_data.get(DEBUG_POSTCARDS_JOB_KEY)
    return stored_job_name == job_name


def _chat_id_message(chat_id: int) -> str:
    """Return a human-friendly message that exposes the chat id."""

    return (
        "Привет! Пришли мне фото пива, и я попрошу сомелье оставить отзыв.\n\n"
        "ID этого чата: {chat_id}. Передай его админам, чтобы заполнить "
        "POSTCARD_CHAT_ID для еженедельных открыток."
    ).format(chat_id=chat_id)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a greeting message when the bot is started."""

    if not update.message or not update.effective_chat:
        return

    await update.message.reply_text(_chat_id_message(update.effective_chat.id))


async def chat_id_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Reply with the current chat id so admins can copy/paste it."""

    if not update.message or not update.effective_chat:
        return

    await update.message.reply_text(
        f"ID этого чата: {update.effective_chat.id}. "
        "Скопируй его в переменную окружения POSTCARD_CHAT_ID."
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
        placeholder_path=BEER_POSTCARD_PLACEHOLDER_PATH,
    )

    if postcard_sent:
        poll_question = (
            context.application.bot_data.get("beer_poll_question")
            or DEFAULT_BEER_POLL_QUESTION
        )
        await _start_attendance_poll(
            chat_id=update.effective_chat.id,
            context=context,
            question=poll_question,
        )


async def barhopping_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Generate a monthly barhopping invitation on demand."""

    if not update.message:
        return

    prompt_base: str = (
        context.application.bot_data.get("barhopping_prompt")
        or DEFAULT_BARGHOPPING_PROMPT
    )
    extra = update.message.text.partition(" ")[2].strip() if update.message.text else ""
    prompt = _compose_postcard_prompt(context, prompt_base, extra)

    await update.message.reply_chat_action(action=ChatAction.UPLOAD_PHOTO)

    negative_prompt = context.application.bot_data.get("barhopping_negative_prompt")
    caption = context.application.bot_data.get("barhopping_caption")

    postcard_sent = await _send_postcard(
        chat_id=update.effective_chat.id,
        context=context,
        prompt=prompt,
        negative_prompt=negative_prompt,
        caption=caption,
        reply_to_message_id=update.message.message_id,
        placeholder_path=BARGHOPPING_POSTCARD_PLACEHOLDER_PATH,
    )

    if postcard_sent:
        poll_question = (
            context.application.bot_data.get("barhopping_poll_question")
            or DEFAULT_BARGHOPPING_POLL_QUESTION
        )
        await _start_attendance_poll(
            chat_id=update.effective_chat.id,
            context=context,
            question=poll_question,
        )


async def debug_postcards_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Toggle a debug postcard broadcast that runs every five minutes."""

    if not update.message:
        return

    chat_id = update.effective_chat.id

    mode_raw = ""
    if context.args:
        mode_raw = context.args[0]
    elif update.message.text:
        parts = update.message.text.split(None, 1)
        if len(parts) > 1:
            mode_raw = parts[1]

    mode = mode_raw.strip().lower()

    job_name = f"{DEBUG_POSTCARDS_JOB_KEY}_{chat_id}"

    if not mode:
        state_message = _debug_postcards_state_message(
            enabled=_is_debug_postcards_enabled(context, job_name)
        )
        await update.message.reply_text(
            "Использование: /debug_postcards on|off. " + state_message
        )
        return

    if context.job_queue is None:
        state_message = _debug_postcards_state_message(
            enabled=_is_debug_postcards_enabled(context, job_name)
        )
        await update.message.reply_text(
            "Job queue недоступен — не могу управлять отладочной рассылкой. "
            + state_message
        )
        return

    if "postcard_client" not in context.application.bot_data:
        state_message = _debug_postcards_state_message(
            enabled=_is_debug_postcards_enabled(context, job_name)
        )
        await update.message.reply_text(
            "Генерация открыток недоступна: нет доступа к Hugging Face API. "
            + state_message
        )
        return

    if mode in {"on", "enable", "start"}:
        for job in context.job_queue.get_jobs_by_name(job_name):
            job.schedule_removal()

        context.job_queue.run_repeating(
            scheduled_postcard_job,
            interval=DEBUG_POSTCARDS_INTERVAL_SECONDS,
            first=0,
            chat_id=chat_id,
            name=job_name,
            data={"prompt": context.application.bot_data.get("postcard_prompt", "")},
        )
        context.chat_data[DEBUG_POSTCARDS_JOB_KEY] = job_name
        state_message = _debug_postcards_state_message(enabled=True)
        await update.message.reply_text(
            "Отладочная рассылка включена — открытки будут приходить каждые 5 минут. "
            + state_message
        )
        LOGGER.info("Debug postcards enabled for chat %s", chat_id)
        return

    if mode in {"off", "disable", "stop"}:
        jobs = context.job_queue.get_jobs_by_name(job_name)
        for job in jobs:
            job.schedule_removal()

        context.chat_data.pop(DEBUG_POSTCARDS_JOB_KEY, None)
        state_message = _debug_postcards_state_message(enabled=False)

        if jobs:
            await update.message.reply_text(
                "Отладочная рассылка отключена. " + state_message
            )
            LOGGER.info("Debug postcards disabled for chat %s", chat_id)
        else:
            await update.message.reply_text(
                "Отладочная рассылка и так была отключена. " + state_message
            )

        return

    state_message = _debug_postcards_state_message(
        enabled=_is_debug_postcards_enabled(context, job_name)
    )
    await update.message.reply_text(
        "Неизвестный режим. Используй /debug_postcards on или /debug_postcards off. "
        + state_message
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

    # Save context for potential follow-up questions
    conversation_manager: Optional[ConversationManager] = context.application.bot_data.get(
        "conversation_manager"
    )
    if conversation_manager:
        chat_id = update.effective_chat.id
        # We don't save the image itself to history to save tokens/memory,
        # but we save the user's caption (if any) and the bot's review.
        user_text = f"Фото пива. {caption}" if caption else "Фото пива."
        conversation_manager.add_message(chat_id, "user", user_text)
        conversation_manager.add_message(chat_id, "assistant", review)


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Answer beer questions when the bot is addressed directly or by keyword."""
    message = update.message
    if not message:
        return

    if message.entities and message.text:
        for entity in message.entities:
            if entity.type != MessageEntityType.BOT_COMMAND or entity.offset != 0:
                continue

            command = message.text[1 : entity.length]
            command_name = command.split("@", 1)[0].lower()

            if command_name == "debug_postcards":
                await debug_postcards_command(update, context)

            return

    contains_beer_keyword = False

    # VIP Defense Check
    if message.text:
        text_lower = message.text.lower()

        is_vip_targeted = False
        if "wizwiz0107" in text_lower or "барякин" in text_lower:
            is_vip_targeted = True
        elif message.reply_to_message and message.reply_to_message.from_user:
            replied_username = message.reply_to_message.from_user.username
            if replied_username and replied_username.lower() == "wizwiz0107":
                is_vip_targeted = True

        if is_vip_targeted:
            groq_client: Optional[GroqVisionClient] = context.application.bot_data.get("groq_client")
            if groq_client:
                defense_response = await groq_client.defend_vip(message.text)
                if defense_response:
                    await context.bot.send_chat_action(chat_id=message.chat_id, action=ChatAction.TYPING)
                    await message.reply_text(defense_response)
                    return

        trimmed_text = message.text.strip()
        if trimmed_text.startswith("/"):
            command = trimmed_text.split(None, 1)[0]
            command_name = command[1:].split("@", 1)[0].lower()

            if command_name == "debug_postcards":
                await debug_postcards_command(update, context)
                return

        contains_beer_keyword = _mentions_beer_keyword(trimmed_text)

    if contains_beer_keyword:
        await _respond_as_sommelier(message, context, getattr(context.bot, "username", None))
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

    job_data: Dict[str, object] = {}
    if context.job.data and isinstance(context.job.data, dict):
        job_data = context.job.data

    await context.bot.send_chat_action(
        chat_id=context.job.chat_id,
        action=ChatAction.UPLOAD_PHOTO,
    )

    base_prompt = (
        str(job_data.get("prompt", ""))
        or context.application.bot_data.get("postcard_prompt")
        or DEFAULT_POSTCARD_PROMPT
    )

    postcard_prompt = _compose_postcard_prompt(context, base_prompt)

    postcard_sent = await _send_postcard(
        chat_id=context.job.chat_id,
        context=context,
        prompt=postcard_prompt,
        negative_prompt=str(job_data.get("negative_prompt", ""))
        if job_data.get("negative_prompt")
        else None,
        caption=str(job_data.get("caption", ""))
        if job_data.get("caption")
        else None,
        placeholder_path=BEER_POSTCARD_PLACEHOLDER_PATH,
    )

    if postcard_sent:
        poll_question = (
            str(job_data.get("poll_question", ""))
            or context.application.bot_data.get("beer_poll_question")
            or DEFAULT_BEER_POLL_QUESTION
        )
        await _start_attendance_poll(
            chat_id=context.job.chat_id,
            context=context,
            question=poll_question,
        )


async def scheduled_barhopping_notification_job(
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Send the monthly barhopping reminder if tomorrow is penultimate Friday."""

    if not context.job or context.job.chat_id is None:
        LOGGER.warning("Barhopping job triggered without chat_id; skipping")
        return

    job_data: Dict[str, object] = {}
    if context.job.data and isinstance(context.job.data, dict):
        job_data = context.job.data

    timezone_name = (
        str(job_data.get("timezone", ""))
        or context.application.bot_data.get("barhopping_timezone")
        or "Asia/Almaty"
    )

    try:
        tzinfo = ZoneInfo(timezone_name)
    except Exception:  # pragma: no cover - defensive branch
        LOGGER.exception(
            "Не удалось определить таймзону '%s' для бархоппинга.", timezone_name
        )
        return

    today = datetime.now(tzinfo).date()
    tomorrow = today + timedelta(days=1)

    if not _is_penultimate_friday(tomorrow):
        LOGGER.debug(
            "Skipping barhopping reminder: %s is not the penultimate Friday of the month.",
            tomorrow,
        )
        return

    await context.bot.send_chat_action(
        chat_id=context.job.chat_id,
        action=ChatAction.UPLOAD_PHOTO,
    )

    base_prompt = (
        str(job_data.get("prompt", ""))
        or context.application.bot_data.get("barhopping_prompt")
        or DEFAULT_BARGHOPPING_PROMPT
    )

    postcard_prompt = _compose_postcard_prompt(context, base_prompt)

    negative_prompt = job_data.get("negative_prompt") or context.application.bot_data.get(
        "barhopping_negative_prompt"
    )
    caption = job_data.get("caption") or context.application.bot_data.get(
        "barhopping_caption"
    )
    poll_question = (
        job_data.get("poll_question")
        or context.application.bot_data.get("barhopping_poll_question")
        or DEFAULT_BARGHOPPING_POLL_QUESTION
    )

    postcard_sent = await _send_postcard(
        chat_id=context.job.chat_id,
        context=context,
        prompt=postcard_prompt,
        negative_prompt=str(negative_prompt) if negative_prompt else None,
        caption=str(caption) if caption else None,
        placeholder_path=BARGHOPPING_POSTCARD_PLACEHOLDER_PATH,
    )

    if postcard_sent:
        await _start_attendance_poll(
            chat_id=context.job.chat_id,
            context=context,
            question=str(poll_question),
        )


def _is_penultimate_friday(candidate: date) -> bool:
    """Return whether the given date is the penultimate Friday of its month."""

    if candidate.weekday() != 4:  # 0=Monday, 4=Friday
        return False

    next_friday = candidate + timedelta(days=7)
    two_weeks_later = candidate + timedelta(days=14)

    return next_friday.month == candidate.month and two_weeks_later.month != candidate.month


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

    conversation_manager: Optional[ConversationManager] = context.application.bot_data.get(
        "conversation_manager"
    )
    history = conversation_manager.get_history(message.chat_id) if conversation_manager else None

    try:
        answer = await groq_client.answer_beer_question(question, history=history)
    except Exception:  # pragma: no cover - runtime guard
        LOGGER.exception("Failed to answer beer question")
        await message.reply_text("Не удалось обсудить пиво, попробуй позже.")
        return

    await message.reply_text(answer)

    if conversation_manager:
        conversation_manager.add_message(message.chat_id, "user", question)
        conversation_manager.add_message(message.chat_id, "assistant", answer)


def _mentions_beer_keyword(text: str) -> bool:
    """Return True if the text contains the word 'пиво' or similar forms."""

    if not text:
        return False

    return re.search(r"\bпив[а-яё]*", text, flags=re.IGNORECASE) is not None


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
    negative_prompt: Optional[str] = None,
    caption: Optional[str] = None,
    reply_to_message_id: Optional[int] = None,
    placeholder_path: Path,
) -> bool:
    """Generate postcard and send it to the specified chat."""

    client: Optional[HuggingFacePostcardClient] = context.application.bot_data.get(
        "postcard_client"
    )

    caption_source = (
        caption
        if caption is not None
        else context.application.bot_data.get("postcard_caption", "")
    )
    caption_to_send = str(caption_source) if caption_source is not None else ""

    if not client:
        LOGGER.warning(
            "Postcard client is not configured — sending bundled placeholder postcard"
        )
        placeholder_bytes = _load_placeholder_postcard(path=placeholder_path)
        if placeholder_bytes is None:
            fail_text = "Генерация открыток недоступна: нет доступа к Hugging Face API."
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
            photo=placeholder_bytes,
            caption=caption_to_send,
            reply_to_message_id=reply_to_message_id,
        )
        return True

    if negative_prompt is None:
        negative_prompt = context.application.bot_data.get("postcard_negative_prompt")

    if negative_prompt is not None:
        negative_prompt = str(negative_prompt)

    try:
        image_bytes = await client.generate_postcard(
            prompt,
            negative_prompt=negative_prompt,
            placeholder_path=placeholder_path,
        )
    except Exception:  # pragma: no cover - runtime guard
        LOGGER.exception("Failed to generate postcard")
        placeholder_bytes = _load_placeholder_postcard(path=placeholder_path)

        if placeholder_bytes is not None:
            LOGGER.info("Sending placeholder postcard due to generation failure")
            await context.bot.send_photo(
                chat_id=chat_id,
                photo=placeholder_bytes,
                caption=caption_to_send,
                reply_to_message_id=reply_to_message_id,
            )
            return True

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
        caption=caption_to_send,
        reply_to_message_id=reply_to_message_id,
    )

    return True


async def _start_attendance_poll(
    *,
    chat_id: int,
    context: ContextTypes.DEFAULT_TYPE,
    question: str = DEFAULT_BEER_POLL_QUESTION,
    options: Optional[List[str]] = None,
) -> None:
    """Send a poll asking who plans to join the upcoming meetup."""

    poll_options = list(options or DEFAULT_ATTENDANCE_OPTIONS)

    poll_message = await context.bot.send_poll(
        chat_id=chat_id,
        question=question,
        options=poll_options,
        is_anonymous=False,
        allows_multiple_answers=False,
        api_kwargs={"allow_view_results_without_vote": True},
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
