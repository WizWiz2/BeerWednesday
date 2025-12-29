import unittest
from unittest.mock import AsyncMock, MagicMock, patch
from beer_bot.groq_client import GroqVisionClient

class TestGroqClient(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.client = GroqVisionClient(
            api_key="fake",
            model="llama-3.2-11b-vision-preview",
            base_url="https://api.groq.com/openai/v1/chat/completions",
            temperature=0.5,
            max_tokens=100
        )

    async def test_request_completion_cleans_artifacts(self):
        """Test that Llama 3 header artifacts are removed from the response."""

        artifact_response = "<|header_start|>assistant<|header_end|>\nReal response"

        with patch("httpx.AsyncClient.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [
                    {
                        "message": {
                            "content": artifact_response
                        }
                    }
                ]
            }
            mock_post.return_value = mock_response

            # Access _request_completion directly for testing since it's "protected"
            # but we want to verify the cleaning logic specifically.
            response = await self.client._request_completion([])

            self.assertEqual(response, "Real response")

    async def test_defend_vip_returns_none_on_artifact_only(self):
        """Test that defend_vip returns None if the response only contains artifacts."""

        artifact_only_response = "<|header_start|>assistant<|header_end|>\n  \n"

        with patch("httpx.AsyncClient.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [
                    {
                        "message": {
                            "content": artifact_only_response
                        }
                    }
                ]
            }
            mock_post.return_value = mock_response

            response = await self.client.defend_vip("Hello")

            self.assertIsNone(response)

    async def test_defend_vip_returns_content(self):
        """Test that defend_vip returns content when it's real text."""

        real_response = "<|header_start|>assistant<|header_end|>\nI am watching you."

        with patch("httpx.AsyncClient.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [
                    {
                        "message": {
                            "content": real_response
                        }
                    }
                ]
            }
            mock_post.return_value = mock_response

            response = await self.client.defend_vip("You are bad")

            self.assertEqual(response, "I am watching you.")

if __name__ == "__main__":
    unittest.main()
