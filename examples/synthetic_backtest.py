"""Offline, reproducible OpenRange Backtest demonstration.

Run from repository root: python -m examples.synthetic_backtest
No real financial data or API credentials are used.
"""
from datetime import datetime, timedelta
import json
import pytz

from src.data_models import Candle
from src.pipeline import classify_days_for_symbol
from src.backtest import run_simple_backtest, summarize_results


def bar(day, hour, minute, start, finish):
    opened = datetime(2026, 11, day, hour, minute, tzinfo=pytz.utc)
    return Candle(symbol="SYNTHETIC", open_time_utc=opened,
                  close_time_utc=opened + timedelta(minutes=15),
                  open=start, high=max(start, finish),
                  low=min(start, finish), close=finish, volume=1000)


def main():
    # Nov 9 & 10: 09:00 Berlin is 08:00 UTC (standard time).
    candles = [
        bar(9, 8, 0, 100, 101), bar(9, 8, 15, 101, 102),
        bar(9, 16, 15, 103, 104),
        bar(10, 8, 0, 100, 99), bar(10, 8, 15, 99, 98),
        bar(10, 16, 15, 96, 95),
    ]
    classes = classify_days_for_symbol(candles, open_hour=9, open_minute=0)
    trades = run_simple_backtest(candles, classes,
                                 open_hour=9, open_minute=15,
                                 close_hour=17, close_minute=15)
    stats = summarize_results(trades)
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
