# OpenRange Backtest

OpenRange Backtest is a configurable opening-range analysis and backtesting tool.
It tests whether the first 30 minutes of a trading session (two 15-minute candles) –
optionally combined with filters like gaps, weekday, and bar-size relationships –
can statistically predict the rest-of-day direction for any symbol your data source supports.

This repository is organized into phases. **Phase 1** focuses on a 1-year intraday prototype
using Yahoo Finance for quick experimentation, with Finnhub support scaffolded for later
30-year DEU40/DEU40E studies.

## 1. Project Structure

- `requirements.txt` – Python dependencies.
- `src/` – Application code.
  - `config.py` – Configuration and environment handling (e.g., Finnhub API key, symbols, date ranges).
  - `finnhub_client.py` – Lightweight wrapper for the Finnhub REST API.
  - `data_models.py` – Typed containers for candles and trading days.
  - `pipeline.py` – High-level functions to fetch, normalize, and classify candles into scenarios A–D.
  - `backtest.py` – Core backtest engine for Phase 1 (1-year intraday test).
  - `main.py` – Simple CLI entrypoint to run Phase 1 checks from the command line.

## 2. Environment Setup

It is recommended to use a virtual environment.

```bash
cd "d:\stock prediction"
python -m venv .venv
.venv\Scripts\activate  # PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 3. Configuration

For Finnhub usage (future 30-year study), the engine reads configuration from
environment variables:

- `FINNHUB_API_KEY` – Your Finnhub API key.
- `DEU40_SYMBOL` – Exact Finnhub symbol for DEU40.
- `DEU40E_SYMBOL` – Exact Finnhub symbol for DEU40E.

Defaults and validation logic are implemented in `config.py`.

## 4. Running the Streamlit UI

Once the environment is set up:

```bash
cd "d:\stock prediction"
python -m streamlit run src/ui_app.py
```

The UI will:

1. Let you choose a data source (Yahoo or Finnhub).
2. Let you select symbols and a date window.
3. Infer market session and timezone automatically (for Yahoo symbols).
4. Apply optional filters (weekday, gaps, open vs prior range, 1st/2nd bar sign and size).
5. Classify days into A–D scenarios, run the simple backtest, and display summary statistics
   plus diagnostics about candles, classified days, and total trades.

Later phases will:

- Scale from 1-year to 30-years of DEU40/DEU40E data.
- Persist cleaned datasets to disk.
- Produce full statistical reports and visualizations.
