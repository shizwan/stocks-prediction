from __future__ import annotations

import json
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import altair as alt
import pandas as pd
import streamlit as st

# Ensure the project root is on sys.path so `src` is importable as a package
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_models import Candle, CandleColor
from src.backtest import (
    run_simple_backtest,
    summarize_results,
    compute_scenario_prediction_stats,
)
from src.config import Settings
from src.finnhub_client import FinnhubClient
from src.pipeline import (
    fetch_intraday_15m_for_range,
    fetch_intraday_5m_for_range,
    classify_days_for_symbol,
    classify_candle_color,
)
from src.yahoo_data import (
    fetch_intraday_15m_yahoo,
    fetch_intraday_5m_yahoo,
    infer_market_session_yahoo,
)
from src.filters import FilterConfig, apply_filters


def _default_dates(days_back: int = 365) -> tuple[date, date]:
    end = date.today()
    start = end - timedelta(days=days_back)
    return start, end


def _build_tradingview_url(symbol: str) -> str | None:
    """
    Build a TradingView chart URL for a given symbol.

    If the symbol already contains an exchange prefix like 'NASDAQ:AAPL',
    it is used as-is. Otherwise we let TradingView resolve the symbol
    without an explicit exchange.
    """
    cleaned = (symbol or "").strip()
    if not cleaned:
        return None

    tv_symbol = cleaned
    return f"https://www.tradingview.com/chart/?symbol={tv_symbol}"


def run_backtest_finnhub(
    api_key: str,
    symbol1: str,
    symbol2: str,
    days_back: int,
    open_hour: int,
    open_minute: int,
    close_hour: int,
    close_minute: int,
) -> tuple[dict, dict]:
    settings = Settings(
        finnhub_api_key=api_key,
        deu40_symbol=symbol1,
        deu40e_symbol=symbol2,
        days_back=days_back,
    )

    client = FinnhubClient(api_key=settings.finnhub_api_key)

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

    return _run_backtest_on_candles(
        deu40_candles,
        deu40e_candles,
        open_hour=open_hour,
        open_minute=open_minute,
        close_hour=close_hour,
        close_minute=close_minute,
        source="finnhub",
        symbol1=symbol1,
        symbol2=symbol2,
    )


def run_backtest_yahoo(
    symbol1: str,
    symbol2: str,
    days_back: int,
    open_hour: int,
    open_minute: int,
    close_hour: int,
    close_minute: int,
) -> tuple[dict, dict]:
    start, end = _default_dates(days_back)

    deu40_candles = fetch_intraday_15m_yahoo(symbol1, start, end)
    deu40e_candles = fetch_intraday_15m_yahoo(symbol2, start, end)

    return _run_backtest_on_candles(
        deu40_candles,
        deu40e_candles,
        open_hour=open_hour,
        open_minute=open_minute,
        close_hour=close_hour,
        close_minute=close_minute,
        source="yahoo",
        symbol1=symbol1,
        symbol2=symbol2,
    )


def _run_backtest_on_candles(
    deu40_candles,
    deu40e_candles,
    *,
    open_hour: int,
    open_minute: int,
    close_hour: int,
    close_minute: int,
    source: str,
    symbol1: str,
    symbol2: str,
) -> tuple[dict, dict]:
    # Classify days based on 15-minute opening behavior.
    deu40_classes = classify_days_for_symbol(
        deu40_candles,
        open_hour=open_hour,
        open_minute=open_minute,
    )
    deu40e_classes = classify_days_for_symbol(
        deu40e_candles,
        open_hour=open_hour,
        open_minute=open_minute,
    )

    # Build filter configuration from current UI choices.
    filter_cfg = _build_filter_config()

    if filter_cfg is not None:
        deu40_classes = apply_filters(
            deu40_classes,
            deu40_candles,
            filter_cfg,
            open_hour=open_hour,
            open_minute=open_minute,
        )
        deu40e_classes = apply_filters(
            deu40e_classes,
            deu40e_candles,
            filter_cfg,
            open_hour=open_hour,
            open_minute=open_minute,
        )

    deu40_results = run_simple_backtest(
        deu40_candles,
        deu40_classes,
        open_hour=open_hour,
        open_minute=open_minute + 15,
        close_hour=close_hour,
        close_minute=close_minute,
    )
    deu40e_results = run_simple_backtest(
        deu40e_candles,
        deu40e_classes,
        open_hour=open_hour,
        open_minute=open_minute + 15,
        close_hour=close_hour,
        close_minute=close_minute,
    )

    summary = summarize_results(deu40_results + deu40e_results)

    # Scenario-level directional prediction stats (A–D), independent of trading rules.
    pred_stats_1 = compute_scenario_prediction_stats(
        deu40_candles,
        deu40_classes,
        open_hour=open_hour,
        open_minute=open_minute,
        close_hour=close_hour,
        close_minute=close_minute,
    )
    pred_stats_2 = compute_scenario_prediction_stats(
        deu40e_candles,
        deu40e_classes,
        open_hour=open_hour,
        open_minute=open_minute,
        close_hour=close_hour,
        close_minute=close_minute,
    )

    debug = {
        "source": source,
        "symbol_1": symbol1,
        "symbol_2": symbol2,
        "candles_symbol_1": len(deu40_candles),
        "candles_symbol_2": len(deu40e_candles),
        "days_classified_symbol_1": len(deu40_classes),
        "days_classified_symbol_2": len(deu40e_classes),
        "total_trades": sum(v.get("trades", 0) for v in summary.values()),
        "scenario_prediction_symbol_1": pred_stats_1,
        "scenario_prediction_symbol_2": pred_stats_2,
    }

    return summary, debug


def _group_candles_by_day(candles: List[Candle]) -> Dict[Tuple[str, date], List[Candle]]:
    grouped: Dict[Tuple[str, date], List[Candle]] = defaultdict(list)
    for c in candles:
        c_cet = c.to_cet()
        d = c_cet.open_time_utc.date()
        key = (c_cet.symbol, d)
        grouped[key].append(c_cet)
    for key in grouped:
        grouped[key].sort(key=lambda x: x.open_time_utc)
    return grouped


def _find_first_two_candles(
    day_candles: List[Candle],
    open_hour: int,
    open_minute: int,
) -> tuple[Candle | None, Candle | None]:
    """
    Find the first two candles around the configured open time.
    """
    target_minutes = open_hour * 60 + open_minute
    best_idx = -1
    best_diff = 9999
    for idx, c in enumerate(day_candles):
        m = c.open_time_utc.hour * 60 + c.open_time_utc.minute
        diff = abs(m - target_minutes)
        if diff < best_diff:
            best_diff = diff
            best_idx = idx
    if best_idx == -1 or best_diff > 30:
        return None, None
    c1 = day_candles[best_idx]
    c2 = day_candles[best_idx + 1] if best_idx + 1 < len(day_candles) else None
    return c1, c2


def _find_first_n_candles(
    day_candles: List[Candle],
    open_hour: int,
    open_minute: int,
    n: int,
) -> List[Candle]:
    """
    Find up to `n` candles around the configured open time.
    """
    if n <= 0:
        return []

    target_minutes = open_hour * 60 + open_minute
    best_idx = -1
    best_diff = 9999
    for idx, c in enumerate(day_candles):
        m = c.open_time_utc.hour * 60 + c.open_time_utc.minute
        diff = abs(m - target_minutes)
        if diff < best_diff:
            best_diff = diff
            best_idx = idx

    if best_idx == -1 or best_diff > 30:
        return []

    start = best_idx
    end = min(best_idx + n, len(day_candles))
    return day_candles[start:end]


def _collect_gap_opening_samples(
    candles: List[Candle],
    open_hour: int,
    open_minute: int,
    open_vs_prev_choice: str,
    candles_to_use: int,
) -> tuple[pd.DataFrame, Dict[str, int]]:
    """
    Build a tabular dataset of days where the session open is above the prior
    day high or below the prior day low, based on the user's choice.
    """
    by_day = _group_candles_by_day(candles)

    # Daily OHLC used to derive previous-day highs/lows and closes.
    daily_ohlc: Dict[Tuple[str, date], Tuple[float, float, float, float]] = {}
    for key, day_candles in by_day.items():
        if not day_candles:
            continue
        o = day_candles[0].open
        c = day_candles[-1].close
        h = max(candle.high for candle in day_candles)
        l = min(candle.low for candle in day_candles)
        daily_ohlc[key] = (o, h, l, c)

    records: list[dict] = []
    total_days = len(by_day)
    kept_days = 0

    # Funnel counters to mirror backtest filters and reveal where
    # configuration choices remove most days.
    dropped_weekday = 0
    dropped_no_first_candles = 0
    dropped_no_prev_ohlc = 0
    dropped_gap = 0
    dropped_open_vs = 0
    dropped_bar_sign = 0
    dropped_size_relation = 0

    # Optional weekday filter based on the current UI selection.
    weekdays_selected = st.session_state.get("filter_weekdays", [])
    weekday_map = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4}
    allowed_weekdays = {weekday_map[d] for d in weekdays_selected if d in weekday_map}

    # Optional gap direction / size filters based on the current UI selection.
    gap_dir_ui = st.session_state.get("filter_gap_direction", "Any")
    gap_direction = {
        "Any": "any",
        "Gap up": "up",
        "Gap down": "down",
    }.get(gap_dir_ui, "any")

    gap_size_ui = st.session_state.get("filter_gap_size", "Any")
    gap_min_abs_pct = {
        "Any": 0.0,
        "≥ 0.25%": 0.0025,
        "≥ 0.5%": 0.005,
        "≥ 1%": 0.01,
        "≥ 1.5%": 0.015,
    }.get(gap_size_ui, 0.0)

    # Optional first/second/third bar sign filters.
    first_sign_ui = st.session_state.get("filter_first_bar_sign", "Any")
    second_sign_ui = st.session_state.get("filter_second_bar_sign", "Any")
    third_sign_ui = st.session_state.get("filter_third_bar_sign", "Any")
    sign_map = {
        "Any": "any",
        "Positive": "positive",
        "Negative": "negative",
    }
    first_bar_sign = sign_map.get(first_sign_ui, "any")
    second_bar_sign = sign_map.get(second_sign_ui, "any")
    third_bar_sign = sign_map.get(third_sign_ui, "any")

    # Optional body-size relation filter.
    size_rel_ui = st.session_state.get("filter_bar_size_relation", "Any")
    bar_size_relation = {
        "Any": "any",
        "1st bar larger than 2nd": "first_gt_second",
        "1st bar smaller than 2nd": "first_lt_second",
    }.get(size_rel_ui, "any")

    for (symbol, d), day_candles in by_day.items():
        if allowed_weekdays and d.weekday() not in allowed_weekdays:
            dropped_weekday += 1
            continue
        first_candles = _find_first_n_candles(
            day_candles,
            open_hour=open_hour,
            open_minute=open_minute,
            n=candles_to_use,
        )
        if not first_candles:
            dropped_no_first_candles += 1
            continue
        c1 = first_candles[0]
        c2 = first_candles[1] if len(first_candles) >= 2 else None
        c3 = first_candles[2] if len(first_candles) >= 3 else None

        prev_key = (symbol, d - timedelta(days=1))
        prev_ohlc = daily_ohlc.get(prev_key)
        if not prev_ohlc:
            dropped_no_prev_ohlc += 1
            continue

        _, prev_high, prev_low, prev_close = prev_ohlc
        if prev_close <= 0:
            dropped_gap += 1
            continue

        o = c1.open
        gap_frac = (o - prev_close) / prev_close
        gap_pct = gap_frac * 100.0

        # Apply gap direction/size filters (same semantics as filters.py).
        if gap_direction == "up" and gap_frac <= 0:
            dropped_gap += 1
            continue
        if gap_direction == "down" and gap_frac >= 0:
            dropped_gap += 1
            continue
        if abs(gap_frac) < gap_min_abs_pct:
            dropped_gap += 1
            continue

        relation = "inside"
        if o > prev_high:
            relation = "above_high"
        elif o < prev_low:
            relation = "below_low"

        if open_vs_prev_choice == "Above prior day high" and relation != "above_high":
            dropped_open_vs += 1
            continue
        if open_vs_prev_choice == "Below prior day low" and relation != "below_low":
            dropped_open_vs += 1
            continue
        if open_vs_prev_choice not in ("Any", "Above prior day high", "Below prior day low"):
            # Treat any unknown choice as "Any"
            pass

        # First/second bar sign filters (applied on the chosen timeframe).
        if c2 is not None:
            color1 = classify_candle_color(c1.open, c1.close)
            color2 = classify_candle_color(c2.open, c2.close)
            color3 = classify_candle_color(c3.open, c3.close) if c3 is not None else None

            if first_bar_sign == "positive" and color1 is not CandleColor.GREEN:
                dropped_bar_sign += 1
                continue
            if first_bar_sign == "negative" and color1 is not CandleColor.RED:
                dropped_bar_sign += 1
                continue

            if second_bar_sign == "positive" and color2 is not CandleColor.GREEN:
                dropped_bar_sign += 1
                continue
            if second_bar_sign == "negative" and color2 is not CandleColor.RED:
                dropped_bar_sign += 1
                continue

            if third_bar_sign != "any":
                # Require a 3rd candle if user filters on it.
                if color3 is None:
                    dropped_bar_sign += 1
                    continue
                if third_bar_sign == "positive" and color3 is not CandleColor.GREEN:
                    dropped_bar_sign += 1
                    continue
                if third_bar_sign == "negative" and color3 is not CandleColor.RED:
                    dropped_bar_sign += 1
                    continue

            # Body-size relation filter.
            if bar_size_relation != "any":
                body1 = abs(c1.close - c1.open)
                body2 = abs(c2.close - c2.open)
                if bar_size_relation == "first_gt_second" and not (body1 > body2):
                    dropped_size_relation += 1
                    continue
                if bar_size_relation == "first_lt_second" and not (body1 < body2):
                    dropped_size_relation += 1
                    continue

        kept_days += 1

        rec = {
            "Symbol": symbol,
            "Date": d,
            "Open relation vs prior": relation,
            "Gap vs prior close (%)": round(gap_pct, 3),
            "First candle open": c1.open,
            "First candle close": c1.close,
        }
        if c2:
            rec["Second candle open"] = c2.open
            rec["Second candle close"] = c2.close
        if c3 and candles_to_use >= 3:
            rec["Third candle open"] = c3.open
            rec["Third candle close"] = c3.close
        records.append(rec)

    df = pd.DataFrame(records)
    debug = {
        "total_days": total_days,
        "matched_days": kept_days,
        "dropped_weekday": dropped_weekday,
        "dropped_no_first_candles": dropped_no_first_candles,
        "dropped_no_prev_ohlc": dropped_no_prev_ohlc,
        "dropped_gap": dropped_gap,
        "dropped_open_vs": dropped_open_vs,
        "dropped_bar_sign": dropped_bar_sign,
        "dropped_size_relation": dropped_size_relation,
    }
    return df, debug


def _render_opening_candle_charts(
    df: pd.DataFrame,
    candles: List[Candle],
    open_hour: int,
    open_minute: int,
    max_charts: int = 50,
    focus_symbol: str | None = None,
    focus_date: date | None = None,
    candles_per_chart: int = 2,
) -> None:
    if df.empty or not candles:
        return

    by_day = _group_candles_by_day(candles)

    st.markdown("#### Sample charts (first candles around the open)")

    for _, row in df.head(max_charts).iterrows():
        symbol = row["Symbol"]
        d = row["Date"]
        key = (symbol, d)
        day_candles = by_day.get(key)
        if not day_candles:
            continue

        candles_to_plot = _find_first_n_candles(
            day_candles,
            open_hour=open_hour,
            open_minute=open_minute,
            n=candles_per_chart,
        )
        if not candles_to_plot:
            continue

        chart_df = pd.DataFrame(
            {
                "idx": list(range(1, len(candles_to_plot) + 1)),
                "time": [c.open_time_utc for c in candles_to_plot],
                "open": [c.open for c in candles_to_plot],
                "high": [c.high for c in candles_to_plot],
                "low": [c.low for c in candles_to_plot],
                "close": [c.close for c in candles_to_plot],
            }
        )

        # Tighten the Y scale around the actual candle range so that
        # the bars are clearly visible instead of flattened at the top.
        y_min = float(chart_df["low"].min())
        y_max = float(chart_df["high"].max())
        padding = (y_max - y_min) * 0.1 if y_max > y_min else max(y_max * 0.01, 1.0)

        base = (
            alt.Chart(chart_df)
            .encode(
                x=alt.X(
                    "idx:O",
                    title="Candle",
                    axis=alt.Axis(
                        values=list(chart_df["idx"]),
                        labelExpr="datum.value == 1 ? '1st' : (datum.value == 2 ? '2nd' : '3rd')",
                    ),
                )
            )
        )
        rule = base.mark_rule().encode(
            y=alt.Y(
                "low:Q",
                title="Price",
                scale=alt.Scale(domain=[y_min - padding, y_max + padding]),
            ),
            y2="high:Q",
        )
        bar = base.mark_bar(size=24).encode(
            y="open:Q",
            y2="close:Q",
            color=alt.condition(
                "datum.close >= datum.open",
                alt.value("#22c55e"),
                alt.value("#ef4444"),
            ),
        )
        chart = (rule + bar).properties(height=220)

        label = f"{d} – {symbol} ({row['Open relation vs prior']}, gap {row['Gap vs prior close (%)']:.3f}%)"
        expanded = focus_symbol == symbol and focus_date == d
        with st.expander(label, expanded=expanded):
            st.altair_chart(chart, use_container_width=True)


def _build_filter_config() -> FilterConfig | None:
    """
    Read the current Streamlit widget values from session state and build
    a FilterConfig. If all options are effectively "any", returns None.
    """

    # Day of week
    weekdays_selected = st.session_state.get("filter_weekdays", [])
    weekday_map = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4}
    allowed_weekdays = [weekday_map[d] for d in weekdays_selected if d in weekday_map]

    # Gap direction
    gap_dir_ui = st.session_state.get("filter_gap_direction", "Any")
    gap_direction = {
        "Any": "any",
        "Gap up": "up",
        "Gap down": "down",
    }.get(gap_dir_ui, "any")

    # Gap size
    gap_size_ui = st.session_state.get("filter_gap_size", "Any")
    gap_min_abs_pct = {
        "Any": 0.0,
        "≥ 0.25%": 0.0025,
        "≥ 0.5%": 0.005,
        "≥ 1%": 0.01,
        "≥ 1.5%": 0.015,
    }.get(gap_size_ui, 0.0)

    # Open vs prior
    open_vs_ui = st.session_state.get("filter_open_vs_prev", "Any")
    open_vs_prev = {
        "Any": "any",
        "Above prior day high": "above_high",
        "Below prior day low": "below_low",
    }.get(open_vs_ui, "any")

    # First/second bar signs
    first_sign_ui = st.session_state.get("filter_first_bar_sign", "Any")
    second_sign_ui = st.session_state.get("filter_second_bar_sign", "Any")
    sign_map = {
        "Any": "any",
        "Positive": "positive",
        "Negative": "negative",
    }
    first_bar_sign = sign_map.get(first_sign_ui, "any")
    second_bar_sign = sign_map.get(second_sign_ui, "any")

    # Bar size relation
    size_rel_ui = st.session_state.get("filter_bar_size_relation", "Any")
    bar_size_relation = {
        "Any": "any",
        "1st bar larger than 2nd": "first_gt_second",
        "1st bar smaller than 2nd": "first_lt_second",
    }.get(size_rel_ui, "any")

    cfg = FilterConfig(
        allowed_weekdays=allowed_weekdays or None,
        gap_direction=gap_direction,
        min_gap_abs_pct=gap_min_abs_pct,
        open_vs_prev=open_vs_prev,
        first_bar_sign=first_bar_sign,
        second_bar_sign=second_bar_sign,
        bar_size_relation=bar_size_relation,
    )

    if (
        cfg.allowed_weekdays is None
        and cfg.gap_direction == "any"
        and cfg.min_gap_abs_pct <= 0.0
        and cfg.open_vs_prev == "any"
        and cfg.first_bar_sign == "any"
        and cfg.second_bar_sign == "any"
        and cfg.bar_size_relation == "any"
    ):
        return None

    return cfg


def main() -> None:
    st.set_page_config(
        page_title="OpenRange Backtest",
        page_icon="📈",
        layout="wide",
    )

    # --- Global theme CSS ---
    st.markdown("""<style>
.stApp {
    background:
        radial-gradient(circle at top left, rgba(56,189,248,0.12), transparent 55%),
        radial-gradient(circle at bottom right, rgba(244,114,182,0.10), transparent 55%),
        #020617;
    color: #e5e7eb;
}
[data-testid="stSidebar"] {
    background-color: rgba(15,23,42,0.98);
    border-right: 1px solid rgba(148,163,184,0.28);
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] label {
    color: #e5e7eb !important;
}
.block-container {
    padding-top: 2.5rem;
    padding-bottom: 3rem;
    max-width: 1120px;
}
.or-card-row {
    display: flex;
    gap: 1.5rem;
    margin-top: 1.5rem;
}
.or-card {
    flex: 1;
    min-height: 110px;
    border-radius: 0.9rem;
    border: 1px solid rgba(148,163,184,0.35);
    background-color: rgba(15,23,42,0.96);
    padding: 1.1rem 1.4rem;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.or-card--wide {
    flex: 2;
}
.or-card--narrow {
    flex: 1;
}
.or-card-title {
    font-weight: 600;
    font-size: 0.95rem;
    margin-bottom: 0.25rem;
}
.or-card-text {
    font-size: 0.85rem;
    color: #9ca3af;
}
.stSlider > div > div > div {
    background: #374151;
}
.stButton > button[kind="primary"] {
    background: #e5e7eb;
    color: #020617;
    font-weight: 600;
    border-radius: 999px;
    border: none;
}
.stButton > button[kind="primary"]:hover {
    background: #ffffff;
    color: #020617;
}
.or-tv-link-container {
    margin-top: 0.75rem;
    margin-bottom: 0.75rem;
    display: flex;
    justify-content: flex-start;
}
.or-tv-link {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.45rem 0.95rem;
    border-radius: 999px;
    border: 1px solid rgba(148,163,184,0.4);
    background: radial-gradient(circle at top left, rgba(56,189,248,0.22), rgba(15,23,42,0.98));
    color: #e5e7eb !important;
    font-size: 0.85rem;
    text-decoration: none !important;
    box-shadow: 0 8px 18px rgba(15,23,42,0.7);
    transition: border-color 0.18s ease, box-shadow 0.18s ease, transform 0.12s ease, background 0.18s ease;
}
.or-tv-link:link,
.or-tv-link:visited,
.or-tv-link:hover,
.or-tv-link:active {
    text-decoration: none !important;
}
.or-tv-link:hover {
    border-color: rgba(248,250,252,0.9);
    box-shadow: 0 14px 28px rgba(15,23,42,0.9);
    transform: translateY(-1px);
    background: radial-gradient(circle at top left, rgba(56,189,248,0.35), rgba(15,23,42,0.98));
}
.or-tv-link-icon {
    width: 20px;
    height: 20px;
    display: inline-block;
}
</style>""", unsafe_allow_html=True)

    # --- Hero header ---
    st.markdown("# OpenRange Data Collection")
    st.caption(
        "Collect structured datasets of opening gaps and first candles to study how "
        "the first minutes of a session behave across many days."
    )

    # Read URL query parameters (for deep-linking to a specific symbol/date chart).
    # Support both older and newer Streamlit APIs.
    if hasattr(st, "experimental_get_query_params"):
        query_params = st.experimental_get_query_params()
    else:
        try:
            query_params = dict(st.query_params)  # type ignore[attr-defined]
        except Exception:
            query_params = {}

    qp_symbol = (query_params.get("symbol") or [None])[0]
    qp_date_raw = (query_params.get("date") or [None])[0]
    qp_date: date | None = None
    if qp_date_raw:
        try:
            qp_date = date.fromisoformat(qp_date_raw)
        except ValueError:
            qp_date = None

    with st.sidebar:
        st.header("Configuration")

        mode = st.radio(
            "Mode",
            options=["Data collection"],
            index=0,
        )

        data_source = st.radio(
            "Data source",
            options=["Yahoo Finance (intraday)", "Finnhub (intraday)"],
            index=0,
            help="Yahoo is convenient but typically limited to ~60 days of 5m/15m history. "
            "Finnhub can usually provide up to ~3 years of intraday data.",
        )

        api_key: str = ""
        if data_source == "Finnhub (intraday)":
            api_key = st.text_input(
                "Finnhub API key",
                type="password",
                help="Your Finnhub API token. Required to fetch intraday candles.",
            )

        col1, col2 = st.columns(2)
        with col1:
            default_primary = "^GDAXI" if data_source.startswith("Yahoo") else "DEU40"
            symbol1 = st.text_input("Primary symbol", value=default_primary)
        with col2:
            default_secondary = "^GDAXI" if data_source.startswith("Yahoo") else "DEU40E"
            symbol2 = st.text_input("Comparison symbol", value=default_secondary)

        # For data collection, we want to support up to ~3 years of history when
        # using a provider that allows it (e.g. Finnhub). Yahoo will internally
        # truncate to its own intraday limits.
        days_back = st.slider(
            "Days of history",
            min_value=30,
            max_value=1095,
            value=365,
            step=15,
            help=(
                "Maximum lookback is 3 years. "
                "Yahoo Finance typically only returns ~60 days of intraday data; "
                "Finnhub can usually provide the full range."
            ),
        )

        st.markdown("---")
        filters_header_col, filters_button_col = st.columns([1, 1])
        with filters_header_col:
            st.subheader("Filters")
        with filters_button_col:
            if st.button("Reset", help="Reset all filters back to 'Any'"):
                st.session_state["filter_weekdays"] = []
                st.session_state["filter_gap_direction"] = "Any"
                st.session_state["filter_gap_size"] = "Any"
                st.session_state["filter_open_vs_prev"] = "Any"
                st.session_state["filter_first_bar_sign"] = "Any"
                st.session_state["filter_second_bar_sign"] = "Any"
                st.session_state["filter_third_bar_sign"] = "Any"
                st.session_state["filter_bar_size_relation"] = "Any"

        st.multiselect(
            "Day of week",
            options=["Mon", "Tue", "Wed", "Thu", "Fri"],
            default=[],
            help="If empty, all weekdays are allowed.",
            key="filter_weekdays",
        )

        st.selectbox(
            "Gap direction",
            options=["Any", "Gap up", "Gap down"],
            index=0,
            key="filter_gap_direction",
        )

        st.selectbox(
            "Gap size",
            options=["Any", "≥ 0.25%", "≥ 0.5%", "≥ 1%", "≥ 1.5%"],
            index=0,
            key="filter_gap_size",
        )

        st.selectbox(
            "Open vs previous day range",
            options=["Any", "Above prior day high", "Below prior day low"],
            index=0,
            key="filter_open_vs_prev",
        )

        timeframe = st.selectbox(
            "Opening timeframe",
            options=["15m", "5m"],
            index=0,
            help="Controls the candle size used when evaluating the session open. "
            "For 15m, we use the first 3 x 5-minute candles (5+5+5). "
            "For 5m, we use the first 2 x 5-minute candles.",
        )

        st.selectbox(
            "1st 5m bar sign",
            options=["Any", "Positive", "Negative"],
            index=0,
            key="filter_first_bar_sign",
        )

        st.selectbox(
            "2nd 5m bar sign",
            options=["Any", "Positive", "Negative"],
            index=0,
            key="filter_second_bar_sign",
        )

        # 3rd candle filter only applies to the 15m opening window (3 x 5m candles).
        if timeframe == "15m":
            st.selectbox(
                "3rd 5m bar sign",
                options=["Any", "Positive", "Negative"],
                index=0,
                key="filter_third_bar_sign",
            )
        else:
            # Prevent stale third-candle filters from affecting 5m mode.
            st.session_state["filter_third_bar_sign"] = "Any"

        st.selectbox(
            "Body size relation (1st vs 2nd 5m)",
            options=["Any", "1st bar larger than 2nd", "1st bar smaller than 2nd"],
            index=0,
            key="filter_bar_size_relation",
        )

        run_label = "Fetch dataset" if mode == "Data collection" else "Run backtest"
        run_button = st.button(run_label, type="primary")

    # --- Single TradingView link for the primary symbol (top of main UI) ---
    tv_url = _build_tradingview_url(symbol1)
    if tv_url:
        st.markdown(
            f"""
<div class="or-tv-link-container">
  <a class="or-tv-link" href="{tv_url}" target="_blank" rel="noopener noreferrer">
    <span class="or-tv-link-icon">
      <svg role="img" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" aria-hidden="true" focusable="false">
        <path d="M15.8654 8.2789c0 1.3541 -1.0978 2.4519 -2.452 2.4519 -1.354 0 -2.4519 -1.0978 -2.4519 -2.452 0 -1.354 1.0978 -2.4518 2.452 -2.4518 1.3541 0 2.4519 1.0977 2.4519 2.4519zM9.75 6H0v4.9038h4.8462v7.2692H9.75Zm8.5962 0H24l-5.1058 12.173h-5.6538z" fill="#e5e7eb"></path>
      </svg>
    </span>
    <span>TradingView · {(symbol1 or '').strip().upper() or 'PRIMARY'}</span>
  </a>
</div>
            """,
            unsafe_allow_html=True,
        )

    # Determine default market session. For Yahoo, infer from the primary
    # symbol's metadata. For Finnhub, fall back to the DAX-style session.
    if data_source.startswith("Yahoo"):
        open_hour, open_minute, close_hour, close_minute, inferred_tz = infer_market_session_yahoo(
            symbol1.strip() or "^GDAXI"
        )
    else:
        open_hour, open_minute = 9, 0
        close_hour, close_minute = 17, 15
        inferred_tz = "Europe/Berlin"

    summary = None
    debug = None
    data_df = None
    data_debug = None

    if run_button:
        with st.spinner("Fetching data..."):
            try:
                if data_source == "Finnhub (intraday)":
                    if not api_key.strip():
                        st.error("Finnhub API key is required for the Finnhub data source.")
                        return
                    if mode == "Backtest (Phase 1)":
                        summary, debug = run_backtest_finnhub(
                            api_key,
                            symbol1.strip(),
                            symbol2.strip(),
                            days_back,
                            open_hour=open_hour,
                            open_minute=open_minute,
                            close_hour=close_hour,
                            close_minute=close_minute,
                        )
                    else:
                        client = FinnhubClient(api_key=api_key)
                        start_date = date.today() - timedelta(days=days_back)
                        end_date = date.today()
                        # For data collection we always use 5m candles so that:
                        # - "5m" opening timeframe uses the first 2x5m candles
                        # - "15m" opening timeframe uses the first 3x5m candles (5+5+5)
                        candles = fetch_intraday_5m_for_range(
                            client,
                            symbol=symbol1.strip(),
                            start_date=start_date,
                            end_date=end_date,
                        )
                        open_vs_choice = st.session_state.get("filter_open_vs_prev", "Any")
                        candles_to_use = 3 if timeframe == "15m" else 2
                        data_df, data_debug = _collect_gap_opening_samples(
                            candles,
                            open_hour=open_hour,
                            open_minute=open_minute,
                            open_vs_prev_choice=open_vs_choice,
                            candles_to_use=candles_to_use,
                        )
                else:
                    if mode == "Backtest (Phase 1)":
                        summary, debug = run_backtest_yahoo(
                            symbol1.strip(),
                            symbol2.strip(),
                            days_back,
                            open_hour=open_hour,
                            open_minute=open_minute,
                            close_hour=close_hour,
                            close_minute=close_minute,
                        )
                    else:
                        start, end = _default_dates(days_back)
                        # For data collection we always use 5m candles; opening timeframe
                        # controls how many of the first 5m bars we consider.
                        candles = fetch_intraday_5m_yahoo(symbol1.strip(), start, end)
                        open_vs_choice = st.session_state.get("filter_open_vs_prev", "Any")
                        candles_to_use = 3 if timeframe == "15m" else 2
                        data_df, data_debug = _collect_gap_opening_samples(
                            candles,
                            open_hour=open_hour,
                            open_minute=open_minute,
                            open_vs_prev_choice=open_vs_choice,
                            candles_to_use=candles_to_use,
                        )
            except Exception as exc:
                st.error(
                    "Data request failed. This is often due to API access limits (403/401), "
                    "unsupported symbols, or temporary data source issues."
                )
                st.exception(exc)
                return

    # --- Today's signal (Yahoo-only, backtest mode only) ---
    today_signal = None
    if mode == "Backtest (Phase 1)" and summary and data_source.startswith("Yahoo"):
        primary_symbol = symbol1.strip() or "^GDAXI"
        today = date.today()
        todays_candles = fetch_intraday_15m_yahoo(primary_symbol, today, today)
        if todays_candles:
            todays_classes = classify_days_for_symbol(
                todays_candles,
                open_hour=open_hour,
                open_minute=open_minute,
            )
            today_class = next((c for c in todays_classes if c.date == today), None)
            if today_class and today_class.scenario:
                scen = today_class.scenario
                key = f"{primary_symbol}:{scen}"
                hist = summary.get(key)
                if scen == "A":
                    direction = "long"
                elif scen == "B":
                    direction = "short"
                else:
                    direction = None

                should_trade = False
                hist_trades = 0
                hist_wr = 0.0
                hist_edge = 0.0
                if hist and direction:
                    hist_trades = hist.get("trades", 0)
                    hist_wr = hist.get("win_rate", 0.0) * 100.0
                    hist_edge = hist.get("avg_net_return_pct", 0.0) * 100.0
                    should_trade = hist_trades >= 20 and hist_edge > 0.0

                entry_minutes = open_hour * 60 + open_minute + 15
                entry_h, entry_m = divmod(entry_minutes, 60)
                exit_h, exit_m = close_hour, close_minute

                today_signal = {
                    "scenario": scen,
                    "direction": direction,
                    "should_trade": should_trade,
                    "hist_trades": hist_trades,
                    "hist_wr": hist_wr,
                    "hist_edge": hist_edge,
                    "entry_time": f"{entry_h:02d}:{entry_m:02d}",
                    "exit_time": f"{exit_h:02d}:{exit_m:02d}",
                }

    # --- Overview cards (top row) ---
    if mode == "Backtest (Phase 1)" and summary:
        total_trades = sum(v.get("trades", 0) for v in summary.values())
        win_rates = [v.get("win_rate", 0) for v in summary.values() if v.get("trades", 0) > 0]
        avg_wr = sum(win_rates) / len(win_rates) * 100 if win_rates else 0.0
        net_returns = [v.get("avg_net_return_pct", 0) for v in summary.values() if v.get("trades", 0) > 0]
        avg_nr = sum(net_returns) / len(net_returns) * 100 if net_returns else 0.0

        left_text = f"{days_back} days tested, {len(summary)} scenarios, {total_trades} trades executed."
        right_text = f"Avg win rate {avg_wr:.1f}% and avg net return {avg_nr:.3f}% across all tradable scenarios."
    elif mode == "Data collection" and data_df is not None:
        total_samples = len(data_df)
        unique_days = data_df["Date"].nunique() if not data_df.empty else 0
        left_text = (
            f"{days_back} days scanned for {symbol1.strip() or 'primary symbol'}, "
            f"found {unique_days} matching sessions."
        )
        right_text = (
            "Each matching day is listed below with gap statistics and charts "
            "for the first two opening candles."
        )
    else:
        left_text = "Configure your symbols, history window, and filters in the sidebar, then run."
        right_text = (
            "A dataset of opening-gap sessions will appear here after you run."
        )

    st.markdown(
        f"""
<div class="or-card-row">
  <div class="or-card or-card--wide">
    <div class="or-card-title">Overview</div>
    <div class="or-card-text">
      {left_text}
    </div>
  </div>
  <div class="or-card or-card--narrow">
    <div class="or-card-title">Key metrics</div>
    <div class="or-card-text">
      {right_text}
    </div>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )

    # --- Today's signal card (if available, backtest mode only) ---
    if today_signal and mode == "Backtest (Phase 1)":
        st.markdown("### Today's signal")
        with st.container(border=True):
            scen = today_signal["scenario"]
            direction = today_signal["direction"]
            entry_time = today_signal["entry_time"]
            exit_time = today_signal["exit_time"]
            hist_trades = today_signal["hist_trades"]
            hist_wr = today_signal["hist_wr"]
            hist_edge = today_signal["hist_edge"]
            should_trade = today_signal["should_trade"]

            label = "Trade" if should_trade else "No trade"
            st.markdown(f"**Scenario {scen} – {direction.upper() if direction else 'no direction'}**")
            st.write(
                f"{label} today between **{entry_time}–{exit_time}** "
                f"(based on {hist_trades} historical trades, "
                f"win rate {hist_wr:.1f}%, avg net {hist_edge:.3f}%)."
            )

    # --- Results section ---
    if mode == "Backtest (Phase 1)":
        if summary is None:
            st.markdown("<div style='height: 1.75rem;'></div>", unsafe_allow_html=True)
            st.info("Configure your settings in the sidebar, then click **Run backtest**.")
            return

        st.divider()
        st.markdown("### Phase 1 results")

        if not summary:
            st.warning("No trades were generated. Check your symbol choices, market session, and date window.")
        else:
            # Equity curve preview based on cumulative net return across scenarios
            equity_points: list[dict[str, float | str]] = []
            cumulative_pct = 0.0

            for key in sorted(summary.keys()):
                symbol, scenario = key.split(":", 1)
                stats = summary[key]
                trades = stats.get("trades", 0)
                avg_net = stats.get("avg_net_return_pct", 0.0)  # fraction
                cumulative_pct += avg_net * trades * 100.0
                equity_points.append(
                    {"Step": f"{symbol} {scenario}", "Equity (%)": round(cumulative_pct, 3)}
                )

            if equity_points:
                st.markdown("#### Equity curve preview")
                with st.container(border=True):
                    df_equity = pd.DataFrame(equity_points)
                    st.line_chart(df_equity, x="Step", y="Equity (%)")

            rows = []
            for key, stats in summary.items():
                symbol, scenario = key.split(":", 1)
                rows.append(
                    {
                        "Symbol": symbol,
                        "Scenario": scenario,
                        "Trades": stats.get("trades", 0),
                        "Win rate (%)": round(stats.get("win_rate", 0.0) * 100, 2),
                        "Avg net return (%)": round(stats.get("avg_net_return_pct", 0.0) * 100, 3),
                    }
                )

            if rows:
                st.dataframe(rows, use_container_width=True)

            with st.expander("Raw JSON output"):
                st.code(json.dumps(summary, indent=2), language="json")

        st.divider()
        st.markdown("### Data diagnostics")
        debug_with_tz = {**debug, "inferred_timezone": inferred_tz}
        st.json(debug_with_tz)
    else:
        st.divider()
        st.markdown("### Data collection results")

        if data_df is None:
            st.markdown("<div style='height: 1.75rem;'></div>", unsafe_allow_html=True)
            st.info(
                "Configure your settings in the sidebar, then click **Fetch dataset** "
                "to list days where the open gaps above/below the previous day's range."
            )
            return

        if data_df.empty:
            st.warning(
                "No sessions matched the current filters. Try relaxing the open-vs-prior filter "
                "or shortening the history window."
            )
        else:
            st.dataframe(data_df, use_container_width=True)

            candles_per_chart = 3 if timeframe == "15m" else 2
            _render_opening_candle_charts(
                data_df,
                candles,
                open_hour=open_hour,
                open_minute=open_minute,
                focus_symbol=qp_symbol,
                focus_date=qp_date,
                candles_per_chart=candles_per_chart,
            )

        st.divider()
        st.markdown("### Data diagnostics")
        diag = data_debug or {}
        diag["inferred_timezone"] = inferred_tz
        st.json(diag)


if __name__ == "__main__":
    main()