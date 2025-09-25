"""Client helper for generating postcards via Hugging Face Serverless Inference."""
from __future__ import annotations

import asyncio
import logging
import textwrap
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, Optional

import httpx
from PIL import Image, ImageDraw, ImageFont


LOGGER = logging.getLogger(__name__)


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

        image = Image.new("RGB", (width, height), "#191919")
        draw = ImageDraw.Draw(image)

        accent_color = "#f9a602"
        secondary_color = "#353535"

        draw.rectangle([(0, 0), (width, int(height * 0.22))], fill=secondary_color)
        draw.rectangle([(0, height - int(height * 0.18)), (width, height)], fill=secondary_color)

        title_font = self._load_font(68, bold=True)
        subtitle_font = self._load_font(36, bold=True)
        body_font = self._load_font(30)

        def measure_text(text: str, font: ImageFont.FreeTypeFont | ImageFont.ImageFont) -> tuple[int, int]:
            """Return width/height for the provided text using Pillow's textbbox."""

            try:
                bbox = draw.textbbox((0, 0), text, font=font)
            except AttributeError:  # Pillow < 8.0 fallback
                bbox = font.getbbox(text)
            return bbox[2] - bbox[0], bbox[3] - bbox[1]

        title_text = "Beer Wednesday"
        title_w, _ = measure_text(title_text, title_font)
        draw.text(
            ((width - title_w) / 2, int(height * 0.05)),
            title_text,
            font=title_font,
            fill=accent_color,
        )

        date_text = datetime.now().strftime("%d %B %Y")
        date_w, _ = measure_text(date_text, subtitle_font)
        draw.text(
            ((width - date_w) / 2, int(height * 0.16)),
            date_text,
            font=subtitle_font,
            fill="white",
        )

        body_text = textwrap.fill(prompt.strip(), width=40)
        draw.multiline_text(
            (int(width * 0.08), int(height * 0.32)),
            body_text,
            font=body_font,
            fill="white",
            spacing=12,
        )

        footer_text = "Встретимся в баре!"
        footer_w, _ = measure_text(footer_text, subtitle_font)
        draw.text(
            ((width - footer_w) / 2, height - int(height * 0.12)),
            footer_text,
            font=subtitle_font,
            fill=accent_color,
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
