from __future__ import annotations

import argparse
import json
import logging
from datetime import date

from .backtest import run_simple_backtest, summarize_results
from .config import load_settings
from .finnhub_client import FinnhubClient
from .finnhub_download import (
    download_and_cache_symbol_15m,
    load_cached_symbol_15m,
)
from .pipeline import fetch_intraday_15m_for_range, classify_days_for_symbol


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_phase1_finnhub() -> None:
    """
    Original Phase 1 prototype: 1-year Finnhub window driven by env config.
    """
    settings = load_settings()
    logger.info("Loaded settings: %s", settings)

    client = FinnhubClient(api_key=settings.finnhub_api_key)

    logger.info(
        "Fetching 15m candles for %s and %s from %s to %s",
        settings.deu40_symbol,
        settings.deu40e_symbol,
        settings.start_date,
        settings.end_date,
    )

    deu40_candles = fetch_intraday_15m_for_range(
        client,
        symbol=settings.deu40_symbol,
        start_date=settings.start_date,
        end_date=settings.end_date,
    )
    deu40e_candles = fetch_intraday_15m_for_range(
        client,
        symbol=settings.deu40e_symbol,
        start_date=settings.start_date,
        end_date=settings.end_date,
    )

    logger.info("Fetched %d candles for %s", len(deu40_candles), settings.deu40_symbol)
    logger.info("Fetched %d candles for %s", len(deu40e_candles), settings.deu40e_symbol)

    deu40_classes = classify_days_for_symbol(deu40_candles)
    deu40e_classes = classify_days_for_symbol(deu40e_candles)

    logger.info(
        "Classified %d days for %s and %d days for %s",
        len(deu40_classes),
        settings.deu40_symbol,
        len(deu40e_classes),
        settings.deu40e_symbol,
    )

    deu40_results = run_simple_backtest(deu40_candles, deu40_classes)
    deu40e_results = run_simple_backtest(deu40e_candles, deu40e_classes)

    summary = summarize_results(deu40_results + deu40e_results)

    print("\nPhase 1 summary (proto, 1-year window):")
    print(json.dumps(summary, indent=2))


def run_finnhub_30y() -> None:
    """
    Phase 2-style 30-year backtest using Finnhub and local Parquet caches.
    """
    settings = load_settings()
    logger.info("Loaded settings: %s", settings)

    client = FinnhubClient(api_key=settings.finnhub_api_key)

    earliest = date(1996, 1, 1)
    today = date.today()

    # Download or refresh caches
    for symbol in (settings.deu40_symbol, settings.deu40e_symbol):
        download_and_cache_symbol_15m(client, symbol, earliest, today)

    deu40_candles = load_cached_symbol_15m(settings.deu40_symbol)
    deu40e_candles = load_cached_symbol_15m(settings.deu40e_symbol)

    logger.info("Loaded %d cached candles for %s", len(deu40_candles), settings.deu40_symbol)
    logger.info("Loaded %d cached candles for %s", len(deu40e_candles), settings.deu40e_symbol)

    deu40_classes = classify_days_for_symbol(deu40_candles)
    deu40e_classes = classify_days_for_symbol(deu40e_candles)

    deu40_results = run_simple_backtest(deu40_candles, deu40_classes)
    deu40e_results = run_simple_backtest(deu40e_candles, deu40e_classes)

    summary = summarize_results(deu40_results + deu40e_results)

    print("\nFinnhub 30-year summary:")
    print(json.dumps(summary, indent=2))


def cli() -> None:
    parser = argparse.ArgumentParser(description="OpenRange backtest CLI")
    parser.add_argument(
        "--mode",
        choices=["phase1", "finnhub-30y"],
        default="phase1",
        help="Backtest mode to run.",
    )
    args = parser.parse_args()

    if args.mode == "phase1":
        run_phase1_finnhub()
    elif args.mode == "finnhub-30y":
        run_finnhub_30y()


if __name__ == "__main__":
    cli()

