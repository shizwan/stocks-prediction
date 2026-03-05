from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import List, Tuple

import pytz
import yfinance as yf

from src.data_models import Candle


def fetch_intraday_15m_yahoo(
    symbol: str,
    start_date: date,
    end_date: date,
) -> List[Candle]:
    """
    Fetch 15-minute candles for a symbol from Yahoo Finance.

    This is used as an alternative to Finnhub for local testing when a paid
    intraday feed is not available.
    """

    # yfinance expects datetimes; include one extra day for the exclusive end bound
    start_dt = datetime.combine(start_date, time(0, 0))
    end_dt = datetime.combine(end_date + timedelta(days=1), time(0, 0))

    try:
        # Using Ticker().history tends to be more robust than the
        # top-level download() for single symbols.
        ticker = yf.Ticker(symbol)
        df = ticker.history(
            interval="15m",
            start=start_dt,
            end=end_dt,
            auto_adjust=False,
        )
    except Exception:
        # Any failure here is treated as "no data"
        return []

    if df.empty:
        return []

    candles: List[Candle] = []
    utc = pytz.utc
    cet = pytz.timezone("Europe/Berlin")

    for idx, row in df.iterrows():
        index_ts = idx
        # Yahoo typically returns exchange-local timestamps (for ^GDAXI this is CET/CEST).
        # Normalize to CET, then convert to UTC so the rest of the pipeline remains consistent.
        if index_ts.tzinfo is None:
            index_ts = cet.localize(index_ts)
        else:
            index_ts = index_ts.astimezone(cet)

        open_time_utc = index_ts.astimezone(utc)
        close_time_utc = open_time_utc + timedelta(minutes=15)

        candles.append(
            Candle(
                symbol=symbol,
                open_time_utc=open_time_utc,
                close_time_utc=close_time_utc,
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                volume=float(row.get("Volume", 0.0)),
            )
        )

    return candles


def infer_market_session_yahoo(symbol: str) -> Tuple[int, int, int, int, str]:
    """
    Infer a reasonable market session (open/close times) from Yahoo metadata.

    Returns (open_hour, open_minute, close_hour, close_minute, timezone_name).
    """

    ticker = yf.Ticker(symbol)
    tz = None

    # Prefer fast_info if available
    fast_info = getattr(ticker, "fast_info", None)
    if fast_info is not None:
        tz = getattr(fast_info, "timezone", None)

    if not tz:
        try:
            info = ticker.info or {}
            tz = info.get("exchangeTimezoneName")
        except Exception:
            tz = None

    # Simple mapping for now; can be extended as needed.
    if tz in {"America/New_York", "US/Eastern"}:
        # US equities (NYSE/Nasdaq): 09:30–16:00 ET -> approx 15:30–21:45 CET
        return 15, 30, 21, 45, tz or "America/New_York"

    if tz in {"Europe/Berlin", "Europe/Amsterdam", "Europe/Paris"}:
        # DAX-style European session
        return 9, 0, 17, 15, tz or "Europe/Berlin"

    # Fallback: treat as European-style if unknown
    return 9, 0, 17, 15, tz or "Unknown"

