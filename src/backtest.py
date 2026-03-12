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
            # Scenarios C and D: record as no-trade; only used for classification stats
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
        net_returns = [r.net_return_pct for r in items if r.net_return_pct is not None]
        wins = sum(1 for r in net_returns if r > 0)
        avg_net = sum(net_returns) / n if n else 0.0

        # Basic significance diagnostics for the mean net return per trade.
        if n > 1:
            mean = avg_net
            var = sum((x - mean) ** 2 for x in net_returns) / (n - 1)
            std = var ** 0.5
            se = std / (n ** 0.5)
            t_stat = mean / se if se > 0 else 0.0
        else:
            std = 0.0
            t_stat = 0.0

        stats_key = f"{symbol}:{scenario}"
        stats[stats_key] = {
            "trades": n,
            "win_rate": wins / n if n else 0.0,
            "avg_net_return_pct": avg_net,
            "std_net_return_pct": std,
            "t_stat_mean_gt_0": t_stat,
        }

    return stats


def compute_scenario_prediction_stats(
    candles: List[Candle],
    classifications: List[DayClassification],
    open_hour: int,
    open_minute: int,
    close_hour: int,
    close_minute: int,
) -> Dict[str, Dict[str, float]]:
    """
    For each scenario (A–D), measure how often the expected direction
    of the day matches the realized direction from open to close.

    This is independent of any trading rules and is used to answer
    "how predictive is each scenario of the daily close direction?".
    """

    cet = pytz.timezone("Europe/Berlin")
    by_day: Dict[Tuple[str, date], List[Candle]] = defaultdict(list)

    for c in candles:
        cet_c = c.to_cet()
        d = cet_c.open_time_utc.date()
        by_day[(cet_c.symbol, d)].append(cet_c)

    classified_by_key: Dict[Tuple[str, date], DayClassification] = {
        (cls.symbol, cls.date): cls for cls in classifications if cls.scenario
    }

    # scenario -> list of daily returns where a direction could be determined
    grouped_returns: Dict[str, List[float]] = defaultdict(list)
    correct_counts: Dict[str, int] = defaultdict(int)

    for key, day_candles in by_day.items():
        symbol, d = key
        cls = classified_by_key.get(key)
        if not cls or not cls.scenario:
            continue

        day_candles.sort(key=lambda c: c.open_time_utc)

        open_candle = _find_candle_at(day_candles, hour=open_hour, minute=open_minute)
        close_candle = _find_candle_at(day_candles, hour=close_hour, minute=close_minute)

        if not open_candle or not close_candle:
            continue

        o = open_candle.open
        c = close_candle.close
        if o <= 0 or c <= 0:
            continue

        daily_ret = (c - o) / o
        if abs(daily_ret) < 1e-8:
            # Flat day; ignore for hit-rate calculations.
            continue

        scen = cls.scenario or ""
        grouped_returns[scen].append(daily_ret)

        expected_dir = 0
        if scen in ("A", "C"):
            expected_dir = 1
        elif scen in ("B", "D"):
            expected_dir = -1

        if expected_dir == 1 and daily_ret > 0:
            correct_counts[scen] += 1
        elif expected_dir == -1 and daily_ret < 0:
            correct_counts[scen] += 1

    stats: Dict[str, Dict[str, float]] = {}
    for scen, rets in grouped_returns.items():
        n = len(rets)
        correct = correct_counts.get(scen, 0)
        hit_rate = correct / n if n else 0.0
        avg_ret = sum(rets) / n if n else 0.0

        if n > 1:
            mean = avg_ret
            var = sum((x - mean) ** 2 for x in rets) / (n - 1)
            std = var ** 0.5
            se = std / (n ** 0.5)
            t_stat = mean / se if se > 0 else 0.0
        else:
            std = 0.0
            t_stat = 0.0

        stats[scen] = {
            "days": n,
            "hit_rate": hit_rate,
            "avg_daily_return_pct": avg_ret,
            "std_daily_return_pct": std,
            "t_stat_mean_gt_0": t_stat,
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
