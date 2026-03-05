from __future__ import annotations

from collections import defaultdict
from datetime import date
from typing import Dict, Iterable, List, Tuple

import pytz

from .data_models import Candle, CandleColor, DayClassification, TradeResult


def run_simple_backtest(
    candles: List[Candle],
    classifications: List[DayClassification],
    round_trip_cost_pct: float = 0.0004,
    open_hour: int = 9,
    open_minute: int = 15,
    close_hour: int = 17,
    close_minute: int = 15,
) -> List[TradeResult]:
    """
    Phase 1 backtest:
    - Scenario A: long
    - Scenario B: short
    - Scenarios C/D: no trade for now (recorded only)
    """

    # Group candles by (symbol, date) in CET
    cet = pytz.timezone("Europe/Berlin")
    by_day: Dict[Tuple[str, date], List[Candle]] = defaultdict(list)

    for c in candles:
        cet_candle = c.to_cet()
        d = cet_candle.open_time_utc.date()
        by_day[(cet_candle.symbol, d)].append(cet_candle)

    classified_by_key: Dict[Tuple[str, date], DayClassification] = {
        (cls.symbol, cls.date): cls for cls in classifications
    }

    results: List[TradeResult] = []

    for key, day_candles in by_day.items():
        symbol, d = key
        cls = classified_by_key.get(key)
        if not cls:
            continue

        day_candles.sort(key=lambda c: c.open_time_utc)

        open_candle = _find_candle_at(
            day_candles,
            hour=open_hour,
            minute=open_minute,
        )
        close_candle = _find_candle_at(
            day_candles,
            hour=close_hour,
            minute=close_minute,
        )

        if not open_candle or not close_candle:
            continue

        scenario = cls.scenario or ""
        direction: str | None = None

        if scenario == "A":
            direction = "long"
        elif scenario == "B":
            direction = "short"
        else:
            # Scenarios C and D: record as no-trade; used only for classification statistics
            results.append(
                TradeResult(
                    date=d,
                    symbol=symbol,
                    scenario=scenario,
                    direction=None,
                    entry_price=None,
                    exit_price=None,
                    gross_return_pct=None,
                    net_return_pct=None,
                )
            )
            continue

        entry = open_candle.close
        exit_ = close_candle.close

        if entry <= 0 or exit_ <= 0:
            continue

        if direction == "long":
            gross_return = (exit_ - entry) / entry
        else:
            gross_return = (entry - exit_) / entry

        net_return = gross_return - round_trip_cost_pct

        results.append(
            TradeResult(
                date=d,
                symbol=symbol,
                scenario=scenario,
                direction=direction,
                entry_price=entry,
                exit_price=exit_,
                gross_return_pct=gross_return,
                net_return_pct=net_return,
            )
        )

    return results


def summarize_results(results: Iterable[TradeResult]) -> Dict[str, Dict[str, float]]:
    """
    Produce a simple summary: count, win rate, avg net return per trade
    by symbol and scenario.
    """

    stats: Dict[str, Dict[str, float]] = {}
    grouped: Dict[Tuple[str, str], List[TradeResult]] = defaultdict(list)

    for r in results:
        if r.direction is None or r.net_return_pct is None:
            continue
        key = (r.symbol, r.scenario or "Unknown")
        grouped[key].append(r)

    for (symbol, scenario), items in grouped.items():
        n = len(items)
        wins = sum(1 for r in items if r.net_return_pct > 0)
        avg_net = sum(r.net_return_pct for r in items) / n if n else 0.0

        stats_key = f"{symbol}:{scenario}"
        stats[stats_key] = {
            "trades": n,
            "win_rate": wins / n if n else 0.0,
            "avg_net_return_pct": avg_net,
        }

    return stats


def _find_candle_at(candles: List[Candle], hour: int, minute: int) -> Candle | None:
    target_minutes = hour * 60 + minute
    best: Candle | None = None
    best_diff = 9999

    for c in candles:
        m = c.open_time_utc.hour * 60 + c.open_time_utc.minute
        diff = abs(m - target_minutes)
        if diff < best_diff:
            best_diff = diff
            best = c

    if best is not None and best_diff <= 30:
        return best

    return None
