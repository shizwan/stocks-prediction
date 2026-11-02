"""Regression tests for filtering against session data; no network calls."""
from datetime import datetime, timedelta
import unittest

import pytz

from src.data_models import Candle
from src.filters import FilterConfig, _find_first_two, apply_filters
from src.pipeline import classify_days_for_symbol


def bar(day, hour, minute, opening, closing, symbol="TEST"):
    opened = datetime(2026, 11, day, hour, minute, tzinfo=pytz.utc)
    return Candle(
        symbol=symbol, open_time_utc=opened,
        close_time_utc=opened + timedelta(minutes=15),
        open=opening, high=max(opening, closing),
        low=min(opening, closing), close=closing, volume=100,
    )


class FilteringTests(unittest.TestCase):
    def test_missing_second_opening_bar_does_not_use_later_bar(self):
        # 08:00 UTC == 09:00 Berlin in November.
        candles = [bar(9, 8, 0, 100, 101), bar(9, 8, 30, 101, 102)]
        self.assertEqual((None, None), _find_first_two(
            [c.to_cet() for c in candles], 9, 0,
        ))

    def test_two_consecutive_opening_bars_are_selected(self):
        candles = [bar(9, 8, 0, 100, 101), bar(9, 8, 15, 101, 102)]
        first, second = _find_first_two([c.to_cet() for c in candles], 9, 0)
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertEqual(15 * 60, (second.open_time_utc -
                                  first.open_time_utc).total_seconds())

    def test_monday_gap_uses_prior_friday_not_sunday(self):
        # Friday Nov 6 has a 100 closing price.
        # Monday Nov 9 opens at 102, yielding a +2% weekend gap.
        candles = [
            bar(6, 8, 0, 98, 99),
            bar(6, 8, 15, 99, 100),
            bar(9, 8, 0, 102, 103),
            bar(9, 8, 15, 103, 104),
        ]
        classes = classify_days_for_symbol(candles, open_hour=9, open_minute=0)
        filtered = apply_filters(classes, candles, FilterConfig(gap_direction="up"),
                                 open_hour=9, open_minute=0)
        self.assertEqual([datetime(2026, 11, 9).date()],
                         [entry.date for entry in filtered])

    def test_missing_prior_session_rejects_gap_filter(self):
        candles = [bar(9, 8, 0, 102, 103), bar(9, 8, 15, 103, 104)]
        classes = classify_days_for_symbol(candles, open_hour=9, open_minute=0)
        filtered = apply_filters(classes, candles, FilterConfig(gap_direction="up"),
                                 open_hour=9, open_minute=0)
        self.assertEqual([], filtered)

    def test_gap_does_not_mix_symbols(self):
        candles = [
            bar(6, 8, 0, 98, 100, "OTHER"),
            bar(6, 8, 15, 100, 100, "OTHER"),
            bar(9, 8, 0, 102, 103, "TEST"),
            bar(9, 8, 15, 103, 104, "TEST"),
        ]
        classes = classify_days_for_symbol(candles, open_hour=9, open_minute=0)
        filtered = apply_filters(classes, candles, FilterConfig(gap_direction="up"),
                                 open_hour=9, open_minute=0)
        self.assertEqual([], filtered)


if __name__ == "__main__":
    unittest.main()
