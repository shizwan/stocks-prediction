"""Regression tests for exact 15-minute candle alignment; no network needed."""
from datetime import datetime, timedelta
import unittest
import pytz
from src.data_models import Candle
from src.pipeline import _find_candle_at as find_pipeline, classify_days_for_symbol
from src.backtest import _find_candle_at as find_backtest, run_simple_backtest

UTC = pytz.utc

def candle(hour, minute, opening=100, closing=101, day=3):
    start = datetime(2026, 11, day, hour, minute, tzinfo=UTC)
    return Candle(symbol="TEST", open_time_utc=start,
                  close_time_utc=start + timedelta(minutes=15),
                  open=opening, high=max(opening, closing),
                  low=min(opening, closing), close=closing, volume=100)

class OpeningCandleAlignmentTests(unittest.TestCase):
    def test_missing_open_bar_is_not_replaced_by_nearest_bar(self):
        bars = [candle(8, 15), candle(8, 30)]
        for find in (find_pipeline, find_backtest):
            self.assertIsNone(find(bars, 9, 0))

    def test_two_different_expected_bars_are_required(self):
        bars = [candle(8, 0, 100, 101), candle(8, 30, 101, 102)]
        self.assertEqual([], classify_days_for_symbol(bars, open_hour=9, open_minute=0))

    def test_classification_with_exact_opening_bars(self):
        # 09:00 and 09:15 in Berlin are 08:00 and 08:15 UTC in November.
        bars = [candle(8, 0, 100, 101), candle(8, 15, 101, 102)]
        result = classify_days_for_symbol(bars, open_hour=9, open_minute=0)
        self.assertEqual(1, len(result))
        self.assertEqual("A", result[0].scenario)

    def test_missing_trade_entry_is_not_substituted(self):
        bars = [candle(8, 0, 100, 101), candle(8, 15, 101, 102),
                candle(8, 30, 102, 103), candle(16, 15, 104, 105)]
        classifications = classify_days_for_symbol(bars, open_hour=9, open_minute=0)
        self.assertEqual(1, len(classifications))
        self.assertEqual([], run_simple_backtest(bars, classifications,
                         open_hour=9, open_minute=45, close_hour=17, close_minute=15))

if __name__ == "__main__":
    unittest.main()
