from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable, List, Optional, Sequence, Tuple, Dict

import pytz

from src.data_models import Candle, DayClassification, CandleColor
from src.pipeline import classify_candle_color


@dataclass
class FilterConfig:
    """
    Represents the user-selectable filters from software filter options.
    All fields are optional; None / "any" means no filtering on that dimension.
    """

    # 2. Day of week (0=Mon .. 4=Fri)
    allowed_weekdays: Optional[Sequence[int]] = None

    # 3. Gap direction: "any", "up", "down"
    gap_direction: str = "any"

    # 4. Gap size: minimum absolute gap in decimal (e.g. 0.0025 for 0.25%).
    min_gap_abs_pct: float = 0.0

    # 5. Open vs previous day close/high/low: "any", "above_high", "below_low"
    open_vs_prev: str = "any"

    # 6–7. 1st & 2nd 15m bar sign: "any", "positive", "negative"
    first_bar_sign: str = "any"
    second_bar_sign: str = "any"

    # 8–9. Size comparison of 1st vs 2nd 15m bar: "any", "first_gt_second", "first_lt_second"
    bar_size_relation: str = "any"


def apply_filters(
    classifications: List[DayClassification],
    candles: List[Candle],
    filters: FilterConfig,
    open_hour: int,
    open_minute: int,
) -> List[DayClassification]:
    """
    Filter DayClassification entries using the configured filters.
    """

    if not classifications:
        return classifications

    cet = pytz.timezone("Europe/Berlin")

    # Group candles by (symbol, date) for intraday feature calculations.
    by_day: Dict[Tuple[str, date], List[Candle]] = {}
    for c in candles:
        c_cet = c.to_cet()
        d = c_cet.open_time_utc.date()
        key = (c_cet.symbol, d)
        by_day.setdefault(key, []).append(c_cet)

    # Compute daily OHLC per (symbol, date) for gap and prior-day filters.
    daily_ohlc: Dict[Tuple[str, date], Tuple[float, float, float, float]] = {}
    for key, day_candles in by_day.items():
        day_candles.sort(key=lambda x: x.open_time_utc)
        if not day_candles:
            continue
        o = day_candles[0].open
        c = day_candles[-1].close
        h = max(candle.high for candle in day_candles)
        l = min(candle.low for candle in day_candles)
        daily_ohlc[key] = (o, h, l, c)

    filtered: List[DayClassification] = []

    for cls in classifications:
        key = (cls.symbol, cls.date)
        day_candles = by_day.get(key)
        if not day_candles:
            continue

        if not _passes_weekday_filter(cls.date, filters):
            continue

        # Find first and second 15-minute candles around the open time.
        day_candles.sort(key=lambda x: x.open_time_utc)
        c1, c2 = _find_first_two(day_candles, open_hour, open_minute)
        if not c1 or not c2:
            continue

        # Get previous-day OHLC for gap and open-vs-prior filters.
        prev_key = (cls.symbol, cls.date.fromordinal(cls.date.toordinal() - 1))
        prev_ohlc = daily_ohlc.get(prev_key)

        if not _passes_gap_filters(c1, prev_ohlc, filters):
            continue

        if not _passes_open_vs_prior_filters(c1, prev_ohlc, filters):
            continue

        if not _passes_bar_sign_filters(cls, filters):
            continue

        if not _passes_bar_size_relation(c1, c2, filters):
            continue

        filtered.append(cls)

    return filtered


def _passes_weekday_filter(d: date, filters: FilterConfig) -> bool:
    if not filters.allowed_weekdays:
        return True
    return d.weekday() in filters.allowed_weekdays


def _passes_gap_filters(
    first_candle: Candle,
    prev_ohlc: Optional[Tuple[float, float, float, float]],
    filters: FilterConfig,
) -> bool:
    if filters.gap_direction == "any" and filters.min_gap_abs_pct <= 0.0:
        return True

    if not prev_ohlc:
        # Cannot evaluate gap without previous day close.
        return False

    _, _, _, prev_close = prev_ohlc
    if prev_close <= 0:
        return False

    gap_pct = (first_candle.open - prev_close) / prev_close

    # Direction filter.
    if filters.gap_direction == "up" and gap_pct <= 0:
        return False
    if filters.gap_direction == "down" and gap_pct >= 0:
        return False

    # Size filter (absolute).
    if abs(gap_pct) < filters.min_gap_abs_pct:
        return False

    return True


def _passes_open_vs_prior_filters(
    first_candle: Candle,
    prev_ohlc: Optional[Tuple[float, float, float, float]],
    filters: FilterConfig,
) -> bool:
    if filters.open_vs_prev == "any":
        return True

    if not prev_ohlc:
        return False

    _, prev_high, prev_low, _ = prev_ohlc
    o = first_candle.open

    if filters.open_vs_prev == "above_high":
        return o > prev_high
    if filters.open_vs_prev == "below_low":
        return o < prev_low

    return True


def _passes_bar_sign_filters(
    cls: DayClassification,
    filters: FilterConfig,
) -> bool:
    # First bar
    if filters.first_bar_sign == "positive" and cls.candle_1_color is not CandleColor.GREEN:
        return False
    if filters.first_bar_sign == "negative" and cls.candle_1_color is not CandleColor.RED:
        return False

    # Second bar
    if filters.second_bar_sign == "positive" and cls.candle_2_color is not CandleColor.GREEN:
        return False
    if filters.second_bar_sign == "negative" and cls.candle_2_color is not CandleColor.RED:
        return False

    return True


def _passes_bar_size_relation(
    c1: Candle,
    c2: Candle,
    filters: FilterConfig,
) -> bool:
    if filters.bar_size_relation == "any":
        return True

    body1 = abs(c1.close - c1.open)
    body2 = abs(c2.close - c2.open)

    if filters.bar_size_relation == "first_gt_second":
        return body1 > body2
    if filters.bar_size_relation == "first_lt_second":
        return body1 < body2

    return True


def _find_first_two(
    day_candles: List[Candle],
    open_hour: int,
    open_minute: int,
) -> Tuple[Optional[Candle], Optional[Candle]]:
    """
    Find the first two candles around the configured open time.
    """

    target_minutes = open_hour * 60 + open_minute
    # Find nearest candle to the open time.
    best_idx = -1
    best_diff = 9999
    for idx, c in enumerate(day_candles):
        m = c.open_time_utc.hour * 60 + c.open_time_utc.minute
        diff = abs(m - target_minutes)
        if diff < best_diff:
            best_diff = diff
            best_idx = idx

    if best_idx == -1 or best_diff > 30:
        return None, None

    c1 = day_candles[best_idx]
    c2 = day_candles[best_idx + 1] if best_idx + 1 < len(day_candles) else None
    return c1, c2

