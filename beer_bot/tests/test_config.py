import unittest
import os
from unittest.mock import patch
from beer_bot.config import Settings

class TestConfig(unittest.TestCase):
    @patch.dict(os.environ, {
        "TELEGRAM_BOT_TOKEN": "test_token",
        "GROQ_API_KEY": "test_key",
        "BARHOPPING_CHAT_ID": "123456"  # Correct spelling
    }, clear=True)
    def test_load_settings_correct_spelling(self):
        # This should fail if the code only looks for BARGHOPPING_CHAT_ID
        settings = Settings.load()
        self.assertEqual(settings.barhopping_chat_id, 123456)

    @patch.dict(os.environ, {
        "TELEGRAM_BOT_TOKEN": "test_token",
        "GROQ_API_KEY": "test_key",
        "BARGHOPPING_CHAT_ID": "654321"  # Typo spelling
    }, clear=True)
    def test_load_settings_typo_spelling(self):
        settings = Settings.load()
        self.assertEqual(settings.barhopping_chat_id, 654321)

    @patch.dict(os.environ, {
        "TELEGRAM_BOT_TOKEN": "test_token",
        "GROQ_API_KEY": "test_key",
        "BARHOPPING_CHAT_ID": "111",
        "BARGHOPPING_CHAT_ID": "222"
    }, clear=True)
    def test_load_settings_priority(self):
        # If both are present, which one takes precedence?
        # Ideally the correct one should, or we define a behavior.
        # For now, let's see what happens.
        settings = Settings.load()
        # If I fix it, I'll likely prioritize the correct spelling.
        self.assertEqual(settings.barhopping_chat_id, 111)

if __name__ == '__main__':
    unittest.main()
