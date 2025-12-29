import unittest
import os
from unittest.mock import patch
from beer_bot.config import Settings

class TestConfigFallback(unittest.TestCase):
    @patch.dict(os.environ, {
        "TELEGRAM_BOT_TOKEN": "test_token",
        "GROQ_API_KEY": "test_key",
        "POSTCARD_CHAT_ID": "99999"
    }, clear=True)
    def test_barhopping_inherits_postcard_chat_id(self):
        """Test that barhopping_chat_id falls back to postcard_chat_id if not set."""
        settings = Settings.load()
        self.assertEqual(settings.postcard_chat_id, 99999)
        self.assertEqual(settings.barhopping_chat_id, 99999)
        self.assertEqual(settings.barhopping_timezone, "Asia/Almaty") # Default
        self.assertEqual(settings.barhopping_hour, 12) # Default

    @patch.dict(os.environ, {
        "TELEGRAM_BOT_TOKEN": "test_token",
        "GROQ_API_KEY": "test_key",
        "POSTCARD_CHAT_ID": "99999",
        "BARHOPPING_CHAT_ID": "88888"
    }, clear=True)
    def test_barhopping_explicit_overrides_fallback(self):
        """Test that explicit barhopping_chat_id overrides the fallback."""
        settings = Settings.load()
        self.assertEqual(settings.postcard_chat_id, 99999)
        self.assertEqual(settings.barhopping_chat_id, 88888)

if __name__ == '__main__':
    unittest.main()
