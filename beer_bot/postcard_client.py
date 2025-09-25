"""Client helper for generating postcards via Hugging Face Serverless Inference."""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

import httpx


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
