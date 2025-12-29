import unittest
from datetime import date
from beer_bot.handlers import _is_penultimate_friday

class TestBarhoppingLogic(unittest.TestCase):
    def test_is_penultimate_friday(self):
        # August 2024 Fridays: 2, 9, 16, 23, 30
        # Last: 30
        # Penultimate: 23

        # Test Friday Aug 23, 2024 (Should be True)
        self.assertTrue(_is_penultimate_friday(date(2024, 8, 23)))

        # Test Friday Aug 30, 2024 (Should be False - it's the last one)
        self.assertFalse(_is_penultimate_friday(date(2024, 8, 30)))

        # Test Friday Aug 16, 2024 (Should be False - it's the antepenultimate)
        self.assertFalse(_is_penultimate_friday(date(2024, 8, 16)))

    def test_is_penultimate_friday_feb_2024(self):
        # February 2024 (Leap year) Fridays: 2, 9, 16, 23
        # Last: 23
        # Penultimate: 16

        self.assertTrue(_is_penultimate_friday(date(2024, 2, 16)))
        self.assertFalse(_is_penultimate_friday(date(2024, 2, 23)))

if __name__ == '__main__':
    unittest.main()
