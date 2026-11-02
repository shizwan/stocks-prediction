# OpenRange Backtest

An experimental Python/Streamlit research tool for inspecting intraday opening-range patterns, classifying two 15-minute candles into scenarios A–D, and running simple research backtests.

> **Research software only.** Trading results are hypothetical and should not be interpreted as predictions, investment advice, or guaranteed future returns. Data-provider limits, missing candles, timezone mapping and execution assumptions can materially affect results.

## Current capabilities

- Import intraday OHLCV data from Yahoo Finance using `yfinance` (availability depends on provider limits).
- Scaffold a Finnhub-based data source (requires appropriate API access).
- Infer approximate market sessions, group candles by market day and classify opening patterns.
- Evaluate prototype scenario A long / B short trades and report counts, win rates and average net returns.
- Explore results in a Streamlit UI.

The prototype does **not** implement the planned 30-year data pipeline, robust real-world execution modeling, or systematic validation against all exchange calendars. See [technical overview](BACKTESTING_OVERVIEW.md) for details.

## Run locally

Prerequisites: Python 3.11 or newer, pip, and internet access for live data fetching.

```bash
git clone https://github.com/shizwan/stocks-prediction.git
cd stocks-prediction
python -m venv .venv
```

Activate the environment:

- macOS/Linux: `source .venv/bin/activate`
- Windows PowerShell: `.venv\\Scripts\\Activate.ps1`

Install dependencies and start the app:

```bash
python -m pip install -r requirements.txt
python -m streamlit run src/ui_app.py
```

If using Finnhub, supply your API key through the UI or the `FINNHUB_API_KEY` environment setting as supported by the application; never commit real credentials. Data access is subject to provider restrictions.

## Offline reproducible example

Run a small, entirely synthetic example (no live market-data access needed):

```bash
python -m examples.synthetic_backtest
```

It simulates one bullish and one bearish session, deducts a configurable round-trip cost, and prints aggregated statistics. These numbers are demonstrative and not evidence of financial performance.

## Tests

Run deterministic unit tests without a market-data connection:

```bash
python -m unittest discover -s tests -v
python -m compileall -q src
```

The project includes a GitHub Actions workflow that runs these checks on pushes and pull requests.

## Code map

- `src/data_models.py`: candle, scenario and trade data structures
- `src/yahoo_data.py` and `src/finnhub_client.py`: provider integration
- `src/pipeline.py`: candle grouping and opening scenario classification
- `src/backtest.py`: prototype trade simulation and metrics
- `src/filters.py`: optional filters
- `src/ui_app.py`: Streamlit interface
- `BACKTESTING_OVERVIEW.md`: methodology and current system limitations

## Methodology cautions

- An opening scenario uses two distinct bars at the configured 15-minute timestamps; missing bars are not silently substituted by nearby data.
- The current session mapping is approximate, and US/EU daylight-saving differences need further validation before claims about accuracy.
- Gap filters use the previous available trading session per symbol rather than the previous calendar day; this is not yet a substitute for a validated exchange calendar.
- This project does not yet account comprehensively for slippage, data survivorship, market holidays, exchange-specific market hours, or realistic fills.
- No production-grade historical result or externally verified adoption claim is presented here.

## Contributing and releases

See [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md), and the [release checklist](docs/RELEASE_CHECKLIST.md).

## License

The original application code is available under the [MIT License](LICENSE), copyright (c) 2026 shizwan. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for bundled PDFs, the Streamline/TradingView SVG, external market data, and other materials that are **not** automatically licensed for reuse under MIT. This project is still an experimental research prototype.
