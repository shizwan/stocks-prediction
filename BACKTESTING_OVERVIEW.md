## OpenRange Backtest – System Overview

This document explains how the prototype backtesting system works end-to-end: what data it pulls, how it classifies scenarios, how trades are simulated, and how the UI numbers are computed.

---

### 1. Overall Flow

High-level pipeline:

1. **Fetch candles** for a symbol over a date range (15-minute OHLCV).
2. **Normalize timezones** and infer the market session (open/close times).
3. **Group candles by trading day** in local market time.
4. **Classify the first 30 minutes** (two candles) into scenarios **A–D**.
5. **Simulate trades** for some scenarios (currently A and B).
6. **Aggregate performance statistics** (trades, win rate, average return).
7. Show results and diagnostics in the **Streamlit UI**.

The core logic lives in:

- `yahoo_data.py` – fetching candles and inferring the market session for Yahoo symbols.
- `pipeline.py` – grouping candles by day and classifying scenarios.
- `backtest.py` – simulating trades and aggregating results.
- `ui_app.py` – Streamlit app that wires everything together.

---

### 2. Data Sources and Candle Model

#### 2.1 Yahoo Finance (via `yfinance`)

For now, Yahoo is the default data source:

- We use `yfinance.Ticker(symbol).history(interval="15m", start=..., end=...)`.
- This returns a DataFrame with columns like:
  - `Open`, `High`, `Low`, `Close`, `Volume`.
  - A timestamp index for each 15-minute bar.

In `yahoo_data.py`, each row is converted into a `Candle` object:

- `symbol`: the ticker string you entered (e.g. `^GDAXI`, `AAPL`).
- `open_time_utc`: bar open time in UTC.
- `close_time_utc`: 15 minutes after `open_time_utc`.
- `open`, `high`, `low`, `close`, `volume`: numeric OHLCV values.

The system converts the raw timestamps into **UTC** and later into **Europe/Berlin (CET/CEST)** for consistent date grouping.

#### 2.2 Finnhub (planned / partial)

Finnhub support is scaffolded but requires a paid key with intraday access:

- `FinnhubClient.get_intraday_candles()` calls the `/stock/candle` endpoint.
- The same `Candle` model is used so the rest of the pipeline does not care which provider supplied the data.

---

### 3. Inferring Market Session and Timezones

Different symbols trade in different timezones and with different sessions (e.g. US vs European markets). The system needs to know:

- When the **market opens** (first candle to inspect).
- When the **market closes** (final candle to use for exits).

#### 3.1 Yahoo: Automatic Session Inference

For Yahoo symbols we call:

- `infer_market_session_yahoo(symbol)` in `yahoo_data.py`.

It looks up the ticker’s metadata:

- Tries `ticker.fast_info.timezone`.
- Falls back to `ticker.info["exchangeTimezoneName"]` if needed.

From that timezone, we choose reasonable default sessions:

- **US equities** (`America/New_York`, `US/Eastern`):
  - Local session: ~09:30–16:00 ET.
  - Mapped into Europe/Berlin time for classification.
- **European equities / indices** (`Europe/Berlin`, `Europe/Amsterdam`, `Europe/Paris`):
  - Session: ~09:00–17:15 CET/CEST.
- **Unknown timezone**:
  - Fallback: treat as a European-style session (09:00–17:15) and label timezone as `"Unknown"`.

The inferred open/close times are passed into the classification and backtest functions.

#### 3.2 Finnhub: Fixed Session (for now)

For Finnhub, until a richer metadata layer is added:

- We use a fixed **DAX-style** session:
  - Open: 09:00 CET/CEST.
  - Close: 17:15 CET/CEST.

---

### 4. Grouping Candles by Trading Day

The function `classify_days_for_symbol` in `pipeline.py` handles this:

1. Convert each `Candle` to CET/CEST using `pytz` (`Europe/Berlin`).
2. Group by `(symbol, calendar_date)` using the local open time.
3. For each `(symbol, date)`:
   - Sort the day’s candles by `open_time_utc`.
   - Find the **first 15-minute candle** nearest the configured open time (e.g. 09:00).
   - Find the **second 15-minute candle** 15 minutes after the open (e.g. 09:15).
   - If either is missing or too far in time, skip this day (no scenario).

The lookup allows a small tolerance window (up to 30 minutes difference) to handle provider quirks and daylight-saving edges.

---

### 5. Scenario Classification (A–D)

Once we have the first two 15-minute candles for a given day:

1. Determine the **color** of each candle:
   - **GREEN**: `close > open` by a small epsilon.
   - **RED**: `close < open` by a small epsilon.
   - **NEUTRAL**: `close` approximately equals `open` (doji).
2. If either candle is **NEUTRAL**, the day is excluded from scenario analysis.
3. Otherwise we classify:
   - **Scenario A** – Strong Bullish:
     - Candle 1: GREEN, Candle 2: GREEN.
   - **Scenario B** – Strong Bearish:
     - Candle 1: RED, Candle 2: RED.
   - **Scenario C** – Bullish Reversal / Indecision:
     - Candle 1: RED, Candle 2: GREEN.
   - **Scenario D** – Bearish Reversal / Indecision:
     - Candle 1: GREEN, Candle 2: RED.

Each day becomes a `DayClassification`:

- `date`, `symbol`, `candle_1_color`, `candle_2_color`, `scenario`.

---

### 6. Trade Simulation Logic

The simple Phase 1 backtest in `backtest.py` simulates at most **one trade per day per symbol**.

#### 6.1 When a Trade is Opened

For each classified day:

- **Scenario A**:
  - Open a **long** position.
- **Scenario B**:
  - Open a **short** position.
- **Scenarios C and D**:
  - Currently **no trade** (they are logged only for classification statistics).

This behavior can be extended later to include C/D-based strategies.

#### 6.2 Entry and Exit Prices

Given the inferred session:

- `open_hour`, `open_minute` – market open.
- `close_hour`, `close_minute` – near market close.

The engine:

1. Finds the **entry candle**:
   - Candle aligned to `open_time + 15 minutes`.
   - Uses that candle’s **close** price as `entry_price`.
2. Finds the **exit candle**:
   - Candle aligned to `close_time`.
   - Uses that candle’s **close** price as `exit_price`.

If either candle is missing or invalid (e.g. non-positive prices), the day is skipped for trading.

#### 6.3 Return Calculations

For each trade:

- **Long trade** (Scenario A):
  - Gross return:
    \[
    \text{gross\_return} = \frac{\text{exit\_price} - \text{entry\_price}}{\text{entry\_price}}
    \]
- **Short trade** (Scenario B):
  - Gross return:
    \[
    \text{gross\_return} = \frac{\text{entry\_price} - \text{exit\_price}}{\text{entry\_price}}
    \]
- **Transaction costs**:
  - A fixed **round-trip cost** is subtracted:
    \[
    \text{net\_return} = \text{gross\_return} - 0.0004
    \]
  - 0.0004 = 0.04% total cost (0.02% per side).

Position sizing is expressed in % returns; a later phase can convert this into dollar P&L for a fixed notional (e.g. 1,000 USD per trade).

Each trade is stored as a `TradeResult`:

- `date`, `symbol`, `scenario`, `direction` (long/short), `entry_price`, `exit_price`, `gross_return_pct`, `net_return_pct`.

---

### 7. Aggregated Statistics

The function `summarize_results` groups all trades by:

- `(symbol, scenario)` → e.g. `^GDAXI:A`, `^GDAXI:B`.

For each group it calculates:

- **trades**:
  - Total number of trades in that bucket.
- **win_rate**:
  - Fraction of trades where `net_return_pct > 0`.
- **avg_net_return_pct**:
  - Arithmetic mean of `net_return_pct` across all trades in the group.

This is the structure printed in the UI as JSON and rendered into a small table for readability.

---

### 8. Streamlit UI and Diagnostics

The Streamlit app (`ui_app.py`) provides an interface over the engine.

#### 8.1 Inputs (Sidebar)

- **Data source**:
  - `Yahoo Finance (15m)` or `Finnhub (15m)` (when available).
- **Primary symbol**:
  - Ticker string (e.g. `^GDAXI`, `AAPL`).
- **Comparison symbol**:
  - Optional second ticker; backtest runs on both and merges results.
- **Days of history**:
  - Look-back window (e.g. 30–365 days).
- **Finnhub API key**:
  - Only required when the Finnhub source is selected.

For Yahoo:

- The app automatically infers the timezone and session for the **primary symbol**.

#### 8.2 Outputs

After clicking **Run backtest**, the app shows:

- **Phase 1 summary**:
  - JSON of aggregated stats by `(symbol, scenario)`.
  - A table with:
    - Symbol
    - Scenario
    - Trades
    - Win rate (%)
    - Average net return (%)
- **Data diagnostics**:
  - `source` – `yahoo` or `finnhub`.
  - `symbol_1`, `symbol_2` – what was requested.
  - `candles_symbol_1/2` – how many 15m candles were fetched.
  - `days_classified_symbol_1/2` – how many days had valid A–D scenarios.
  - `total_trades` – number of simulated trades across all scenarios.
  - `inferred_timezone` – timezone detected for the primary symbol (for Yahoo).

These diagnostics confirm that:

- Data was pulled successfully.
- There were enough valid days for classification.
- The trade counts and win rates are based on non-empty samples.

---

### 9. Next Steps Toward the Full Plan

This prototype currently:

- Implements the **scenario logic** (A–D) on 15-minute data.
- Handles **timezones** and session inference for Yahoo symbols.
- Simulates basic **A/B trades** with transaction costs.
- Provides a **UI** to experiment with symbols and windows.

To fully match the 30-year DEU40/DEU40E plan:

1. Add a Finnhub Professional-based data acquisition module with rate-limit handling and 30-year storage.
2. Extend the backtest to run on the full historical dataset and produce:
   - Year-by-year win rates.
   - Regime-specific performance (e.g. 2000, 2008, 2020).
   - Equity curves, drawdowns, and full profitability simulations.
3. Enhance the UI with richer visualizations (heatmaps, yearly bar charts, drawdown plots) and exportable reports.

