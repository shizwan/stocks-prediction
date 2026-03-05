from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List

import requests


logger = logging.getLogger(__name__)


@dataclass
class FinnhubClient:
    """
    Minimal Finnhub REST client for intraday candles.
    """

    api_key: str
    base_url: str = "https://finnhub.io/api/v1"
    timeout_seconds: int = 30

    def _get(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"
        all_params = {**params, "token": self.api_key}

        logger.debug("GET %s params=%s", url, all_params)
        response = requests.get(url, params=all_params, timeout=self.timeout_seconds)
        response.raise_for_status()
        data = response.json()

        if isinstance(data, dict) and data.get("error"):
            raise RuntimeError(f"Finnhub error: {data['error']}")

        return data

    def get_intraday_candles(
        self,
        symbol: str,
        resolution: str,
        from_unix: int,
        to_unix: int,
    ) -> Dict[str, Any]:
        """
        Fetch intraday candles for a symbol between two Unix timestamps.

        `resolution` is a Finnhub resolution code, e.g. '15' for 15-minute candles.
        """

        return self._get(
            "stock/candle",
            {
                "symbol": symbol,
                "resolution": resolution,
                "from": from_unix,
                "to": to_unix,
            },
        )

