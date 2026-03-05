from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, date
from enum import Enum
from typing import List, Optional

import pytz


class CandleColor(str, Enum):
    GREEN = "green"
    RED = "red"
    NEUTRAL = "neutral"


@dataclass
class Candle:
    symbol: str
    open_time_utc: datetime
    close_time_utc: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    def to_cet(self) -> "Candle":
        utc = pytz.utc
        cet = pytz.timezone("Europe/Berlin")
        return Candle(
            symbol=self.symbol,
            open_time_utc=self.open_time_utc.astimezone(cet),
            close_time_utc=self.close_time_utc.astimezone(cet),
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            volume=self.volume,
        )


@dataclass
class DayClassification:
    date: date
    symbol: str
    candle_1_color: CandleColor
    candle_2_color: CandleColor
    scenario: Optional[str]  # "A", "B", "C", "D" or None if unclassified


@dataclass
class TradeResult:
    date: date
    symbol: str
    scenario: Optional[str]
    direction: Optional[str]  # "long", "short", or None (no trade)
    entry_price: Optional[float]
    exit_price: Optional[float]
    gross_return_pct: Optional[float]
    net_return_pct: Optional[float]

