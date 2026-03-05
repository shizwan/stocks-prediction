from __future__ import annotations

import logging
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path
from typing import List

import pandas as pd

from .data_models import Candle
from .finnhub_client import FinnhubClient
from .pipeline import fetch_intraday_15m_for_range

logger = logging.getLogger(__name__)


def _candles_to_dataframe(candles: List[Candle]) -> pd.DataFrame:
    rows = []
    for c in candles:
        rows.append(
            {
                "symbol": c.symbol,
                "open_time_utc": c.open_time_utc,
                "close_time_utc": c.close_time_utc,
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "volume": c.volume,
            }
        )
    return pd.DataFrame(rows)


def _data_dir() -> Path:
    base = Path(__file__).resolve().parent.parent
    path = base / "data" / "finnhub"
    path.mkdir(parents=True, exist_ok=True)
    return path


def download_intraday_15m_paged(
    client: FinnhubClient,
    symbol: str,
    start_date: date,
    end_date: date,
    *,
    chunk_days: int = 120,
) -> List[Candle]:
    """
    Fetch 15-minute intraday candles for a long range by paging in chunks.

    Finnhub's intraday endpoints work best over smaller windows; this helper
    breaks the requested period into `chunk_days` slices and reuses the
    existing fetch_intraday_15m_for_range pipeline function for each slice.
    """

    candles: List[Candle] = []
    current_start = start_date

    while current_start <= end_date:
        current_end = min(current_start + timedelta(days=chunk_days - 1), end_date)
        logger.info(
            "Downloading %s 15m candles from %s to %s",
            symbol,
            current_start,
            current_end,
        )
        slice_candles = fetch_intraday_15m_for_range(
            client,
            symbol=symbol,
            start_date=current_start,
            end_date=current_end,
        )
        candles.extend(slice_candles)
        current_start = current_end + timedelta(days=1)

    return candles


def download_and_cache_symbol_15m(
    client: FinnhubClient,
    symbol: str,
    start_date: date,
    end_date: date,
) -> Path:
    """
    Download 15m intraday candles for `symbol` over the full range and
    persist them to a Parquet file under data/finnhub/.
    """

    candles = download_intraday_15m_paged(client, symbol, start_date, end_date)
    df = _candles_to_dataframe(candles)

    out_dir = _data_dir()
    out_path = out_dir / f"{symbol.replace(':', '_')}_15m.parquet"
    logger.info("Writing %d candles for %s to %s", len(df), symbol, out_path)
    df.to_parquet(out_path, index=False)
    return out_path


def load_cached_symbol_15m(symbol: str) -> List[Candle]:
    """
    Load cached 15m candles for `symbol` from the Parquet file, if present.
    """

    path = _data_dir() / f"{symbol.replace(':', '_')}_15m.parquet"
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_parquet(path)
    candles: List[Candle] = []
    for _, row in df.iterrows():
        candles.append(
            Candle(
                symbol=row["symbol"],
                open_time_utc=row["open_time_utc"],
                close_time_utc=row["close_time_utc"],
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
            )
        )
    return candles

