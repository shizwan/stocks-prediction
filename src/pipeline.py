from __future__ import annotations

from datetime import datetime, timedelta, time, date
from typing import Dict, List, Tuple

import pytz

from .data_models import Candle, CandleColor, DayClassification
from .finnhub_client import FinnhubClient


def _epoch_seconds(dt: datetime) -> int:
    return int(dt.timestamp())


def fetch_intraday_for_range(
    client: FinnhubClient,
    symbol: str,
    start_date: date,
    end_date: date,
    resolution: str,
    minutes: int,
) -> List[Candle]:
    """
    Fetch intraday candles between two dates (inclusive of start, exclusive of end)
    for an arbitrary Finnhub resolution such as '5' or '15'.
    """

    utc = pytz.utc

    start_dt = utc.localize(datetime.combine(start_date, time(0, 0)))
    end_dt = utc.localize(datetime.combine(end_date + timedelta(days=1), time(0, 0)))

    raw = client.get_intraday_candles(
        symbol=symbol,
        resolution=resolution,
        from_unix=_epoch_seconds(start_dt),
        to_unix=_epoch_seconds(end_dt),
    )

    candles: List[Candle] = []
    t_list = raw.get("t") or []
    o_list = raw.get("o") or []
    h_list = raw.get("h") or []
    l_list = raw.get("l") or []
    c_list = raw.get("c") or []
    v_list = raw.get("v") or []

    for ts, o, h, l, c, v in zip(t_list, o_list, h_list, l_list, c_list, v_list):
        open_time_utc = datetime.fromtimestamp(ts, tz=utc)
        close_time_utc = open_time_utc + timedelta(minutes=minutes)
        candles.append(
            Candle(
                symbol=symbol,
                open_time_utc=open_time_utc,
                close_time_utc=close_time_utc,
                open=float(o),
                high=float(h),
                low=float(l),
                close=float(c),
                volume=float(v),
            )
        )

    return candles


def fetch_intraday_15m_for_range(
    client: FinnhubClient,
    symbol: str,
    start_date: date,
    end_date: date,
) -> List[Candle]:
    """
    Fetch 15-minute candles between two dates (inclusive of start, exclusive of end).
    """

    return fetch_intraday_for_range(
        client=client,
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        resolution="15",
        minutes=15,
    )


def fetch_intraday_5m_for_range(
    client: FinnhubClient,
    symbol: str,
    start_date: date,
    end_date: date,
) -> List[Candle]:
    """
    Fetch 5-minute candles between two dates (inclusive of start, exclusive of end).
    """

    return fetch_intraday_for_range(
        client=client,
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        resolution="5",
        minutes=5,
    )


def classify_candle_color(open_price: float, close_price: float, eps: float = 1e-8) -> CandleColor:
    if close_price - open_price > eps:
        return CandleColor.GREEN
    if open_price - close_price > eps:
        return CandleColor.RED
    return CandleColor.NEUTRAL


def classify_days_for_symbol(
    candles: List[Candle],
    open_hour: int = 9,
    open_minute: int = 0,
) -> List[DayClassification]:
    """
    Group candles by (symbol, date) in CET and classify opening scenarios.
    """

    cet = pytz.timezone("Europe/Berlin")
    by_day: Dict[Tuple[str, date], List[Candle]] = {}

    for c in candles:
        candle_cet = c.to_cet()
        d = candle_cet.open_time_utc.date()
        key = (candle_cet.symbol, d)
        by_day.setdefault(key, []).append(candle_cet)

    results: List[DayClassification] = []

    for (symbol, d), day_candles in by_day.items():
        day_candles.sort(key=lambda c: c.open_time_utc)

        # Opening window: first two 15-minute candles starting at the configured
        # market open time (e.g. 09:00/09:15 CET for DAX, or 15:30/15:45 CET for US).
        c1 = _find_candle_at(day_candles, hour=open_hour, minute=open_minute)

        # Second candle is 15 minutes after the first
        total_minutes = open_hour * 60 + open_minute + 15
        c2_hour, c2_minute = divmod(total_minutes, 60)
        c2 = _find_candle_at(day_candles, hour=c2_hour, minute=c2_minute)

        if not c1 or not c2:
            continue

        color1 = classify_candle_color(c1.open, c1.close)
        color2 = classify_candle_color(c2.open, c2.close)

        if color1 is CandleColor.NEUTRAL or color2 is CandleColor.NEUTRAL:
            scenario = None
        else:
            scenario = _scenario_from_colors(color1, color2)

        results.append(
            DayClassification(
                date=d,
                symbol=symbol,
                candle_1_color=color1,
                candle_2_color=color2,
                scenario=scenario,
            )
        )

    return results


def _find_candle_at(candles: List[Candle], hour: int, minute: int) -> Candle | None:
    """Return the candle opening at the exact requested local session time.

    Using a nearest-candle tolerance here can silently select the same candle
    twice when an opening bar is missing, or introduce a later bar into an
    earlier entry decision. Missing bars must be treated as missing data.
    """
    for candle in candles:
        opened = candle.open_time_utc
        if opened.hour == hour and opened.minute == minute:
            return candle
    return None

def _scenario_from_colors(c1: CandleColor, c2: CandleColor) -> str:
    if c1 is CandleColor.GREEN and c2 is CandleColor.GREEN:
        return "A"
    if c1 is CandleColor.RED and c2 is CandleColor.RED:
        return "B"
    if c1 is CandleColor.RED and c2 is CandleColor.GREEN:
        return "C"
    if c1 is CandleColor.GREEN and c2 is CandleColor.RED:
        return "D"
    return ""

