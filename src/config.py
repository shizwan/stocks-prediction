from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, timedelta


@dataclass
class Settings:
    """
    Central configuration object for the Phase 1 engine.
    """

    finnhub_api_key: str
    deu40_symbol: str
    deu40e_symbol: str
    days_back: int = 365

    @property
    def start_date(self) -> date:
        return date.today() - timedelta(days=self.days_back)

    @property
    def end_date(self) -> date:
        return date.today()


def load_settings() -> Settings:
    """
    Load settings from environment variables.

    This keeps secrets (API keys) out of source control.
    """

    api_key = os.getenv("FINNHUB_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "FINNHUB_API_KEY is not set. Please set it in your environment before running Phase 1."
        )

    deu40_symbol = os.getenv("DEU40_SYMBOL", "").strip() or "DEU40"
    deu40e_symbol = os.getenv("DEU40E_SYMBOL", "").strip() or "DEU40E"

    days_back_raw = os.getenv("PHASE1_DAYS_BACK", "").strip()
    days_back = 365
    if days_back_raw:
        try:
            days_back = max(30, int(days_back_raw))
        except ValueError:
            # Fallback to default
            pass

    return Settings(
        finnhub_api_key=api_key,
        deu40_symbol=deu40_symbol,
        deu40e_symbol=deu40e_symbol,
        days_back=days_back,
    )

