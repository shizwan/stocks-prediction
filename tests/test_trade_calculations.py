"""Deterministic tests for basic trading math and chronological validity."""
from datetime import datetime, timedelta
import unittest

import pytz

from src.data_models import Candle
from src.pipeline import classify_days_for_symbol
from src.backtest import run_simple_backtest, summarize_results


def bar(day, hour, minute, opening, closing):
    opened = datetime(2026, 11, day, hour, minute, tzinfo=pytz.utc)
    return Candle(
        symbol="TEST", open_time_utc=opened,
        close_time_utc=opened + timedelta(minutes=15),
        open=opening, high=max(opening, closing),
        low=min(opening, closing), close=closing, volume=100,
    )


class TradeCalculationTests(unittest.TestCase):
    def test_long_return_deducts_round_trip_cost(self):
        candles = [bar(9, 8, 0, 100, 101), bar(9, 8, 15, 101, 102),
                   bar(9, 16, 15, 103, 104)]
        classes = classify_days_for_symbol(candles, 9, 0)
        trades = run_simple_backtest(candles, classes, 0.0004, 9, 15, 17, 15)
        self.assertEqual(1, len(trades))
        self.assertEqual("long", trades[0].direction)
        self.assertAlmostEqual((104 - 102) / 102 - 0.0004,
                               trades[0].net_return_pct)
        stats = summarize_results(trades)
        self.assertEqual(1, stats["TEST:A"]["trades"])
        self.assertEqual(1.0, stats["TEST:A"]["win_rate"])

    def test_short_return_deducts_round_trip_cost(self):
        candles = [bar(10, 8, 0, 100, 99), bar(10, 8, 15, 99, 98),
                   bar(10, 16, 15, 96, 95)]
        classes = classify_days_for_symbol(candles, 9, 0)
        trades = run_simple_backtest(candles, classes, 0.0004, 9, 15, 17, 15)
        self.assertEqual("short", trades[0].direction)
        self.assertAlmostEqual((98 - 95) / 98 - 0.0004,
                               trades[0].net_return_pct)

    def test_exit_before_entry_is_never_allowed(self):
        candles = [bar(9, 8, 0, 100, 101), bar(9, 8, 15, 101, 102),
                   bar(9, 16, 15, 103, 104)]
        classes = classify_days_for_symbol(candles, 9, 0)
        self.assertEqual([], run_simple_backtest(
            candles, classes, 0.0004, 17, 15, 9, 15,
        ))

    def test_invalid_entry_price_drops_trade(self):
        candles = [bar(9, 8, 0, 100, 101), bar(9, 8, 15, 101, 0),
                   bar(9, 16, 15, 103, 104)]
        classes = classify_days_for_symbol(candles, 9, 0)
        self.assertEqual([], run_simple_backtest(
            candles, classes, 0.0004, 9, 15, 17, 15,
        ))


if __name__ == "__main__":
    unittest.main()
