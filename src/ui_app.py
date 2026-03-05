from __future__ import annotations

import json
from datetime import date, timedelta

import pandas as pd
import streamlit as st

try:
    # When run from the project root (e.g. `streamlit run src/ui_app.py`)
    from src.backtest import run_simple_backtest, summarize_results
    from src.config import Settings
    from src.finnhub_client import FinnhubClient
    from src.pipeline import fetch_intraday_15m_for_range, classify_days_for_symbol
    from src.yahoo_data import fetch_intraday_15m_yahoo, infer_market_session_yahoo
    from src.filters import FilterConfig, apply_filters
except ModuleNotFoundError:
    # When Streamlit sets the working directory to `src/`
    from backtest import run_simple_backtest, summarize_results
    from config import Settings
    from finnhub_client import FinnhubClient
    from pipeline import fetch_intraday_15m_for_range, classify_days_for_symbol
    from yahoo_data import fetch_intraday_15m_yahoo, infer_market_session_yahoo
    from filters import FilterConfig, apply_filters


def _default_dates(days_back: int = 365) -> tuple[date, date]:
    end = date.today()
    start = end - timedelta(days=days_back)
    return start, end


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

    debug = {
        "source": source,
        "symbol_1": symbol1,
        "symbol_2": symbol2,
        "candles_symbol_1": len(deu40_candles),
        "candles_symbol_2": len(deu40e_candles),
        "days_classified_symbol_1": len(deu40_classes),
        "days_classified_symbol_2": len(deu40e_classes),
        "total_trades": sum(v.get("trades", 0) for v in summary.values()),
    }

    return summary, debug


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
</style>""", unsafe_allow_html=True)

    # --- Hero header ---
    st.markdown("# OpenRange Backtest")
    st.caption(
        "Analyze how the first 30 minutes of any market session relate to the close, "
        "using 15-minute opening patterns, gap filters, and scenario-based backtesting."
    )

    with st.sidebar:
        st.header("Configuration")

        data_source = st.radio(
            "Data source",
            options=["Yahoo Finance (15m)", "Finnhub (15m)"],
            index=0,
            help="Use Yahoo for quick local tests without a paid Finnhub plan.",
        )

        api_key: str = ""
        if data_source == "Finnhub (15m)":
            api_key = st.text_input(
                "Finnhub API key",
                type="password",
                help="Your Finnhub API token. Required to fetch intraday candles.",
            )

        col1, col2 = st.columns(2)
        with col1:
            default_primary = "^GDAXI" if data_source == "Yahoo Finance (15m)" else "DEU40"
            symbol1 = st.text_input("Primary symbol", value=default_primary)
        with col2:
            default_secondary = "^GDAXI" if data_source == "Yahoo Finance (15m)" else "DEU40E"
            symbol2 = st.text_input("Comparison symbol", value=default_secondary)

        # Yahoo Finance limits intraday 15m history to ~60 days.
        # Keep the slider's maximum aligned with Yahoo even if Finnhub can go further.
        days_back = st.slider(
            "Days of history",
            min_value=30,
            max_value=60,
            value=60,
            step=5,
            help="Limited by Yahoo Finance 15m intraday history (max ~60 days).",
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

        st.selectbox(
            "1st 15m bar sign",
            options=["Any", "Positive", "Negative"],
            index=0,
            key="filter_first_bar_sign",
        )

        st.selectbox(
            "2nd 15m bar sign",
            options=["Any", "Positive", "Negative"],
            index=0,
            key="filter_second_bar_sign",
        )

        st.selectbox(
            "Body size relation (15m)",
            options=["Any", "1st bar larger than 2nd", "1st bar smaller than 2nd"],
            index=0,
            key="filter_bar_size_relation",
        )

        run_button = st.button("Run backtest", type="primary")

    # Determine default market session. For Yahoo, infer from the primary
    # symbol's metadata. For Finnhub, fall back to the DAX-style session.
    if data_source == "Yahoo Finance (15m)":
        open_hour, open_minute, close_hour, close_minute, inferred_tz = infer_market_session_yahoo(
            symbol1.strip() or "^GDAXI"
        )
    else:
        open_hour, open_minute = 9, 0
        close_hour, close_minute = 17, 15
        inferred_tz = "Europe/Berlin"

    summary = None
    debug = None

    if run_button:
        with st.spinner("Fetching data and running backtest..."):
            try:
                if data_source == "Finnhub (15m)":
                    if not api_key.strip():
                        st.error("Finnhub API key is required for the Finnhub data source.")
                        return
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
                    summary, debug = run_backtest_yahoo(
                        symbol1.strip(),
                        symbol2.strip(),
                        days_back,
                        open_hour=open_hour,
                        open_minute=open_minute,
                        close_hour=close_hour,
                        close_minute=close_minute,
                    )
            except Exception as exc:
                st.error(
                    "Backtest failed. This is often due to API access limits (403/401), "
                    "unsupported symbols, or temporary data source issues."
                )
                st.exception(exc)
                return

    # --- Today's signal (Yahoo-only) ---
    today_signal = None
    if summary and data_source == "Yahoo Finance (15m)":
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
    if summary:
        total_trades = sum(v.get("trades", 0) for v in summary.values())
        win_rates = [v.get("win_rate", 0) for v in summary.values() if v.get("trades", 0) > 0]
        avg_wr = sum(win_rates) / len(win_rates) * 100 if win_rates else 0.0
        net_returns = [v.get("avg_net_return_pct", 0) for v in summary.values() if v.get("trades", 0) > 0]
        avg_nr = sum(net_returns) / len(net_returns) * 100 if net_returns else 0.0

        left_text = f"{days_back} days tested, {len(summary)} scenarios, {total_trades} trades executed."
        right_text = f"Avg win rate {avg_wr:.1f}% and avg net return {avg_nr:.3f}% across all tradable scenarios."
    else:
        left_text = "Configure your symbols, history window, and filters in the sidebar, then run a backtest."
        right_text = "Key stats like win rate and average return will appear here after running a backtest."

    st.markdown(
        f"""
<div class="or-card-row">
  <div class="or-card or-card--wide">
    <div class="or-card-title">Backtest overview</div>
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

    # --- Today's signal card (if available) ---
    if today_signal:
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

    # --- Results section (only after backtest) ---
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


if __name__ == "__main__":
    main()