"""Client helper for generating postcards via Hugging Face Serverless Inference."""
from __future__ import annotations

import asyncio
import logging
import textwrap
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
from PIL import Image, ImageDraw, ImageFont


LOGGER = logging.getLogger(__name__)


PLACEHOLDER_POSTCARD_PATH = Path(__file__).resolve().parent / "postcard_placeholder.jpg"


class HuggingFacePostcardClient:
    """Tiny wrapper around the Hugging Face text-to-image endpoint."""

    def __init__(
        self,
        *,
        api_token: str,
        model: str,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        self._api_token = api_token
        self._model = model
        self._base_url = base_url or f"https://api-inference.huggingface.co/models/{model}"
        self._timeout = timeout
        self._max_retries = max_retries

    async def generate_postcard(
        self,
        prompt: str,
        *,
        negative_prompt: Optional[str] = None,
        guidance_scale: float = 3.5,
        num_inference_steps: int = 28,
        width: int = 1024,
        height: int = 1024,
    ) -> bytes:
        """Generate postcard image bytes for the provided text prompt."""

        payload: Dict[str, Any] = {
            "inputs": prompt,
            "parameters": {
                "guidance_scale": guidance_scale,
                "num_inference_steps": num_inference_steps,
                "width": width,
                "height": height,
            },
        }

        if negative_prompt:
            payload["parameters"]["negative_prompt"] = negative_prompt

        headers = {
            "Authorization": f"Bearer {self._api_token}",
            "Content-Type": "application/json",
            "Accept": "image/png",
        }

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            for attempt in range(1, self._max_retries + 1):
                LOGGER.debug(
                    "Requesting postcard generation (attempt %s/%s) from %s",
                    attempt,
                    self._max_retries,
                    self._base_url,
                )
                response = await client.post(self._base_url, headers=headers, json=payload)

                content_type = response.headers.get("content-type", "")
                if response.status_code == httpx.codes.OK and content_type.startswith("image/"):
                    return response.content

                if response.status_code == httpx.codes.PAYMENT_REQUIRED:
                    LOGGER.warning(
                        "Hugging Face вернул 402 — используем запасной шаблон открытки."
                    )
                    return self._render_fallback_postcard(
                        prompt,
                        width=width,
                        height=height,
                    )

                if response.status_code == httpx.codes.ACCEPTED:
                    try:
                        wait_seconds = float(response.json().get("estimated_time", 3.0))
                    except Exception:  # pragma: no cover - best effort parsing
                        wait_seconds = 3.0
                    LOGGER.info(
                        "Model '%s' is loading, retrying in %.1f seconds", self._model, wait_seconds
                    )
                    await asyncio.sleep(min(wait_seconds, 10.0))
                    continue

                try:
                    error_detail = response.json()
                except ValueError:  # pragma: no cover - fallback to text body
                    error_detail = response.text

                LOGGER.error(
                    "Failed to generate postcard via Hugging Face (%s): %s",
                    response.status_code,
                    error_detail,
                )
                response.raise_for_status()

        raise RuntimeError("Не удалось получить открытку от Hugging Face API")

    def _render_fallback_postcard(
        self,
        prompt: str,
        *,
        width: int,
        height: int,
    ) -> bytes:
        """Generate a simple postcard using Pillow when API is unavailable."""

        try:
            with Image.open(PLACEHOLDER_POSTCARD_PATH) as placeholder:
                postcard = placeholder.convert("RGB")
                if postcard.size != (width, height):
                    postcard = postcard.resize((width, height), Image.LANCZOS)
        except FileNotFoundError:
            LOGGER.warning(
                "Фирменный постер не найден — используем резервный рендер",
            )
            return self._render_legacy_postcard(prompt, width=width, height=height)
        except OSError:
            LOGGER.exception(
                "Не удалось загрузить изображение-заглушку, используем резервный рендер",
            )
            return self._render_legacy_postcard(prompt, width=width, height=height)

        buffer = BytesIO()
        postcard.save(buffer, format="PNG")
        return buffer.getvalue()

    def _render_legacy_postcard(
        self,
        prompt: str,
        *,
        width: int,
        height: int,
    ) -> bytes:
        """Render the previous gradient-based postcard as an additional fallback."""

        gradient_top = (17, 24, 39)
        gradient_bottom = (67, 56, 202)

        gradient_strip = Image.new("RGB", (1, height))
        for y in range(height):
            factor = y / max(height - 1, 1)
            r = int(gradient_top[0] * (1 - factor) + gradient_bottom[0] * factor)
            g = int(gradient_top[1] * (1 - factor) + gradient_bottom[1] * factor)
            b = int(gradient_top[2] * (1 - factor) + gradient_bottom[2] * factor)
            gradient_strip.putpixel((0, y), (r, g, b))

        image = gradient_strip.resize((width, height))

        glow_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        glow = ImageDraw.Draw(glow_layer)
        glow_radius = int(width * 0.9)
        glow_center = (int(width * 0.8), int(height * 0.2))
        glow.ellipse(
            [
                (glow_center[0] - glow_radius, glow_center[1] - glow_radius),
                (glow_center[0] + glow_radius, glow_center[1] + glow_radius),
            ],
            fill=(251, 191, 36, 85),
        )

        image = Image.alpha_composite(image.convert("RGBA"), glow_layer).convert("RGB")
        draw = ImageDraw.Draw(image)

        accent_color = "#FBBF24"
        text_color = "#F8FAFC"
        panel_color = (15, 23, 42, 210)

        title_font = self._load_font(82, bold=True)
        subtitle_font = self._load_font(40, bold=True)
        body_font = self._load_font(32)
        caption_font = self._load_font(28)

        padding_x = int(width * 0.08)
        top_padding = int(height * 0.08)
        bottom_padding = int(height * 0.1)

        def measure_text(
            text: str, font: ImageFont.FreeTypeFont | ImageFont.ImageFont
        ) -> tuple[int, int]:
            try:
                bbox = draw.textbbox((0, 0), text, font=font)
            except AttributeError:  # Pillow < 8.0 fallback
                bbox = font.getbbox(text)
            return bbox[2] - bbox[0], bbox[3] - bbox[1]

        header_y = top_padding
        title_text = "Beer Wednesday"
        _, title_h = measure_text(title_text, title_font)
        draw.text(
            (padding_x, header_y),
            title_text,
            font=title_font,
            fill=accent_color,
        )

        date_text = datetime.now().strftime("%d %B %Y")
        _, date_h = measure_text(date_text, subtitle_font)
        draw.text(
            (padding_x, header_y + title_h + 12),
            date_text,
            font=subtitle_font,
            fill=text_color,
        )

        tagline_text = "Крафтовый четверг для своих" if datetime.now().weekday() == 3 else "Среда, когда собираются друзья"
        _, tagline_h = measure_text(tagline_text, caption_font)
        draw.text(
            (padding_x, header_y + title_h + date_h + 36),
            tagline_text,
            font=caption_font,
            fill=text_color,
        )

        panel_top = header_y + title_h + date_h + tagline_h + 76
        panel_left = padding_x
        panel_right = width - padding_x
        panel_bottom = height - bottom_padding

        card_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        card_draw = ImageDraw.Draw(card_layer)
        card_draw.rounded_rectangle(
            [(panel_left, panel_top), (panel_right, panel_bottom)],
            radius=int(width * 0.04),
            fill=panel_color,
        )

        highlight_y = panel_top + int((panel_bottom - panel_top) * 0.18)
        card_draw.line(
            [(panel_left + 60, highlight_y), (panel_right - 60, highlight_y)],
            fill=accent_color,
            width=4,
        )

        image = Image.alpha_composite(image.convert("RGBA"), card_layer).convert("RGB")
        draw = ImageDraw.Draw(image)

        body_margin = 70
        content_left = panel_left + body_margin
        content_top = panel_top + body_margin
        content_right = panel_right - body_margin
        wrap_width = max(28, int((content_right - content_left) / 20))
        formatted_prompt = prompt.strip() or "Поделись настроением вечера, а мы подготовим кружки!"
        body_text = textwrap.fill(formatted_prompt, width=wrap_width)

        draw.multiline_text(
            (content_left, content_top),
            body_text,
            font=body_font,
            fill=text_color,
            spacing=14,
        )

        footer_text = "Ждём тебя у стойки бара"
        footer_w, footer_h = measure_text(footer_text, subtitle_font)
        footer_y = panel_bottom - body_margin - footer_h
        draw.text(
            (panel_right - body_margin - footer_w, footer_y),
            footer_text,
            font=subtitle_font,
            fill=accent_color,
        )

        location_text = "Ламповый бар на углу"
        draw.text(
            (content_left, footer_y),
            location_text,
            font=caption_font,
            fill=text_color,
        )

        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    @staticmethod
    def _load_font(
        size: int, *, bold: bool = False
    ) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        """Try to load a TTF font with graceful fallback to the default bitmap font."""

        font_candidates = [
            "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else \
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]

        for path in font_candidates:
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue

        # Pillow's default font is small, but ensures we still render text.
        return ImageFont.load_default()
