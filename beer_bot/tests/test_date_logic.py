import unittest
from datetime import date
from beer_bot.handlers import _is_penultimate_friday

class TestDateLogic(unittest.TestCase):
    def test_penultimate_friday_normal_month(self):
        # August 2024: 2, 9, 16, 23, 30. Penultimate: 23.
        self.assertTrue(_is_penultimate_friday(date(2024, 8, 23)))
        self.assertFalse(_is_penultimate_friday(date(2024, 8, 30)))
        self.assertFalse(_is_penultimate_friday(date(2024, 8, 16)))

    def test_penultimate_friday_leap_year_feb(self):
        # Feb 2024: 2, 9, 16, 23. Penultimate: 16.
        self.assertTrue(_is_penultimate_friday(date(2024, 2, 16)))
        self.assertFalse(_is_penultimate_friday(date(2024, 2, 23)))

    def test_penultimate_friday_4_fridays(self):
        # Feb 2023: 3, 10, 17, 24. Penultimate: 17.
        self.assertTrue(_is_penultimate_friday(date(2023, 2, 17)))
        self.assertFalse(_is_penultimate_friday(date(2023, 2, 24)))

    def test_not_a_friday(self):
        # Aug 22, 2024 is Thursday
        self.assertFalse(_is_penultimate_friday(date(2024, 8, 22)))

if __name__ == '__main__':
    unittest.main()
