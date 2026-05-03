from __future__ import annotations

from dataclasses import asdict
from datetime import date, datetime, timedelta
from pathlib import Path
import csv
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import portfolio_engine as pe


AlphaConfig = pe.AlphaConfig
ConstructionConfig = pe.ConstructionConfig
find_sustainable_monthly_withdrawal = pe.find_sustainable_monthly_withdrawal
find_sustainable_monthly_withdrawal_monte_carlo = pe.find_sustainable_monthly_withdrawal_monte_carlo
GovernanceConfig = pe.GovernanceConfig
ImplementationConfig = pe.ImplementationConfig
PRESET_UNIVERSES = pe.PRESET_UNIVERSES
RiskConfig = pe.RiskConfig
download_market_data = pe.download_market_data
fetch_fundamentals = pe.fetch_fundamentals
parse_ticker_text = pe.parse_ticker_text
run_forward_test = pe.run_forward_test
run_one_shot_optimization = pe.run_one_shot_optimization
sanitize_tickers = pe.sanitize_tickers
simulate_retirement_paths = pe.simulate_retirement_paths
simulate_retirement_paths_monte_carlo = pe.simulate_retirement_paths_monte_carlo


def infer_market_reference(selected_groups: List[str], selected_tickers: List[str]) -> Dict[str, str]:
    infer_fn = getattr(pe, "infer_market_reference", None)
    if callable(infer_fn):
        return infer_fn(selected_groups, selected_tickers)
    return {"benchmark": "SPY", "vol_proxy": "^VIX"}


st.set_page_config(
    page_title="Portfolio Optimization Studio",
    page_icon=":material/trending_up:",
    layout="wide",
)


CUSTOM_CSS = """
<style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(19, 78, 94, 0.18), transparent 24%),
            radial-gradient(circle at top right, rgba(249, 115, 22, 0.14), transparent 28%),
            linear-gradient(180deg, #f6fbfb 0%, #eef5f4 52%, #f7f7f2 100%);
    }
    .hero {
        padding: 1.5rem 1.7rem;
        border-radius: 20px;
        background: linear-gradient(135deg, rgba(17, 94, 89, 0.98), rgba(15, 23, 42, 0.95));
        color: #f8fafc;
        box-shadow: 0 18px 48px rgba(15, 23, 42, 0.18);
        margin-bottom: 1rem;
    }
    .hero h1 {
        font-size: 2.2rem;
        margin: 0 0 0.3rem 0;
        letter-spacing: -0.03em;
    }
    .hero p {
        font-size: 1rem;
        margin: 0;
        color: rgba(248, 250, 252, 0.85);
    }
    .layer-card {
        border-radius: 18px;
        padding: 1rem 1rem 0.85rem 1rem;
        background: rgba(255, 255, 255, 0.78);
        border: 1px solid rgba(15, 23, 42, 0.08);
        box-shadow: 0 10px 26px rgba(15, 23, 42, 0.06);
        min-height: 190px;
    }
    .layer-card h4 {
        margin-top: 0;
        margin-bottom: 0.45rem;
    }
    .small-note {
        font-size: 0.92rem;
        color: #475569;
    }
</style>
"""


DEFAULT_GROUPS = ["US Liquid Leaders", "Gold-Silver Diversified", "Thailand SET100"]
LEGACY_DEFAULT_GROUPS = ["US Liquid Leaders", "Gold-Silver Diversified"]
DATA_DIR = Path(__file__).resolve().parent / "data"
BACKTEST_RECORDS_FILE = DATA_DIR / "backtest_records.csv"
BACKTEST_RECORD_COLUMNS = [
    "run_id",
    "saved_at",
    "selected_groups",
    "ticker_count",
    "tickers",
    "start_date",
    "end_date",
    "benchmark_symbol",
    "vol_proxy_symbol",
    "construction_method",
    "liquidity_cut",
    "alpha_candidate_cap",
    "target_holdings",
    "max_weight",
    "min_weight",
    "risk_free_rate",
    "target_volatility",
    "lookback_months",
    "rebalance_months",
    "transaction_cost_bps",
    "slippage_bps",
    "use_fundamental_factors",
    "min_avg_dollar_volume_millions",
    "use_historical_eligibility",
    "min_listing_days",
    "use_trend_filter",
    "use_regime_filter",
    "max_drawdown_stop",
    "assumed_aum_usd",
    "rebalances",
    "first_rebalance",
    "last_rebalance",
    "final_portfolio_value",
    "total_return",
    "cagr",
    "annual_volatility",
    "sharpe",
    "sortino",
    "max_drawdown",
    "hit_rate",
]
LEGACY_BACKTEST_RECORD_COLUMNS = [
    column
    for column in BACKTEST_RECORD_COLUMNS
    if column not in {"benchmark_symbol", "vol_proxy_symbol"}
]
BACKTEST_NUMERIC_COLUMNS = [
    "ticker_count",
    "liquidity_cut",
    "alpha_candidate_cap",
    "target_holdings",
    "max_weight",
    "min_weight",
    "risk_free_rate",
    "target_volatility",
    "lookback_months",
    "rebalance_months",
    "transaction_cost_bps",
    "slippage_bps",
    "min_avg_dollar_volume_millions",
    "min_listing_days",
    "max_drawdown_stop",
    "assumed_aum_usd",
    "rebalances",
    "final_portfolio_value",
    "total_return",
    "cagr",
    "annual_volatility",
    "sharpe",
    "sortino",
    "max_drawdown",
    "hit_rate",
]
BACKTEST_DATETIME_COLUMNS = [
    "saved_at",
    "first_rebalance",
    "last_rebalance",
]
BACKTEST_BOOLEAN_COLUMNS = [
    "use_fundamental_factors",
    "use_historical_eligibility",
    "use_trend_filter",
    "use_regime_filter",
]


def ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def metric_value(metrics: pd.DataFrame, metric_name: str) -> float:
    if metrics.empty:
        return float("nan")
    match = metrics.loc[metrics["Metric"] == metric_name, "Value"]
    if match.empty:
        return float("nan")
    return float(match.iloc[0])


def load_backtest_records() -> pd.DataFrame:
    if not BACKTEST_RECORDS_FILE.exists():
        return pd.DataFrame()
    try:
        records = read_backtest_records_file()
    except (pd.errors.EmptyDataError, csv.Error):
        return pd.DataFrame()
    for column in BACKTEST_DATETIME_COLUMNS:
        if column in records.columns:
            records[column] = pd.to_datetime(records[column], errors="coerce")
    for column in BACKTEST_NUMERIC_COLUMNS:
        if column in records.columns:
            records[column] = pd.to_numeric(records[column], errors="coerce")
    for column in BACKTEST_BOOLEAN_COLUMNS:
        if column in records.columns:
            normalized = (
                records[column]
                .astype(str)
                .str.strip()
                .str.lower()
            )
            records[column] = normalized.map(
                {
                    "true": True,
                    "false": False,
                    "1": True,
                    "0": False,
                }
            )
    return records


def read_backtest_records_file() -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    with BACKTEST_RECORDS_FILE.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            next(reader)
        except StopIteration:
            return pd.DataFrame(columns=BACKTEST_RECORD_COLUMNS)

        for row in reader:
            if not row:
                continue
            if len(row) == len(BACKTEST_RECORD_COLUMNS):
                record = dict(zip(BACKTEST_RECORD_COLUMNS, row))
            elif len(row) == len(LEGACY_BACKTEST_RECORD_COLUMNS):
                record = dict(zip(LEGACY_BACKTEST_RECORD_COLUMNS, row))
                record["benchmark_symbol"] = ""
                record["vol_proxy_symbol"] = ""
            else:
                # Skip malformed rows that do not match any known schema version.
                continue
            rows.append(record)

    if not rows:
        return pd.DataFrame(columns=BACKTEST_RECORD_COLUMNS)
    return pd.DataFrame(rows).reindex(columns=BACKTEST_RECORD_COLUMNS)


def save_backtest_records(records: pd.DataFrame) -> None:
    ensure_data_dir()
    output = records.copy().reindex(columns=BACKTEST_RECORD_COLUMNS)
    output.to_csv(BACKTEST_RECORDS_FILE, index=False)


def build_backtest_record(
    controls: Dict[str, object],
    result: Dict[str, object],
    selected_groups: List[str],
) -> Dict[str, object]:
    curve = result.get("curve", pd.DataFrame())
    rebalance_report = result.get("rebalance_report", pd.DataFrame())
    metrics = result.get("metrics", pd.DataFrame())

    final_portfolio_value = float(curve["PortValue"].iloc[-1]) if not curve.empty else np.nan
    first_rebalance = rebalance_report["Date"].min() if not rebalance_report.empty else pd.NaT
    last_rebalance = rebalance_report["Date"].max() if not rebalance_report.empty else pd.NaT

    return {
        "run_id": datetime.now().strftime("bt_%Y%m%d_%H%M%S_%f"),
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "selected_groups": " | ".join(selected_groups),
        "ticker_count": len(controls["selected"]),
        "tickers": ", ".join(controls["selected"]),
        "start_date": controls["start_date"],
        "end_date": controls["end_date"],
        "benchmark_symbol": controls["benchmark_symbol"],
        "vol_proxy_symbol": controls["vol_proxy_symbol"],
        "construction_method": controls["construction_cfg"].method,
        "liquidity_cut": controls["alpha_cfg"].liquidity_cut,
        "alpha_candidate_cap": controls["alpha_cfg"].top_n,
        "target_holdings": controls["alpha_cfg"].target_holdings,
        "max_weight": controls["construction_cfg"].max_weight,
        "min_weight": controls["construction_cfg"].min_weight,
        "risk_free_rate": controls["construction_cfg"].risk_free_rate,
        "target_volatility": controls["construction_cfg"].target_volatility,
        "lookback_months": controls["implementation_cfg"].lookback_months,
        "rebalance_months": controls["implementation_cfg"].rebalance_months,
        "transaction_cost_bps": controls["implementation_cfg"].transaction_cost_bps,
        "slippage_bps": controls["implementation_cfg"].slippage_bps,
        "use_fundamental_factors": controls["alpha_cfg"].use_fundamental_factors,
        "min_avg_dollar_volume_millions": controls["alpha_cfg"].min_avg_dollar_volume_millions,
        "use_historical_eligibility": controls["alpha_cfg"].use_historical_eligibility,
        "min_listing_days": controls["alpha_cfg"].min_listing_days,
        "use_trend_filter": controls["risk_cfg"].use_trend_filter,
        "use_regime_filter": controls["risk_cfg"].use_regime_filter,
        "max_drawdown_stop": controls["risk_cfg"].max_drawdown_stop,
        "assumed_aum_usd": controls["governance_cfg"].assumed_aum_usd,
        "rebalances": int(len(rebalance_report)),
        "first_rebalance": first_rebalance,
        "last_rebalance": last_rebalance,
        "final_portfolio_value": final_portfolio_value,
        "total_return": metric_value(metrics, "Total Return"),
        "cagr": metric_value(metrics, "CAGR"),
        "annual_volatility": metric_value(metrics, "Annual Volatility"),
        "sharpe": metric_value(metrics, "Sharpe"),
        "sortino": metric_value(metrics, "Sortino"),
        "max_drawdown": metric_value(metrics, "Max Drawdown"),
        "hit_rate": metric_value(metrics, "Hit Rate"),
    }


def append_backtest_record(
    controls: Dict[str, object],
    result: Dict[str, object],
    selected_groups: List[str],
) -> str:
    record = pd.DataFrame([build_backtest_record(controls, result, selected_groups)]).reindex(columns=BACKTEST_RECORD_COLUMNS)
    existing = load_backtest_records()
    combined = pd.concat([existing, record], ignore_index=True)
    save_backtest_records(combined)
    return str(record.loc[0, "run_id"])


def default_group_tickers(groups: List[str]) -> List[str]:
    return sanitize_tickers(
        ticker
        for group in groups
        for ticker in PRESET_UNIVERSES.get(group, [])
    )


def init_state() -> None:
    initial_groups = DEFAULT_GROUPS
    initial_tickers = default_group_tickers(initial_groups)
    defaults = {
        "market_bundle": None,
        "fundamentals": None,
        "one_shot_result": None,
        "forward_test_result": None,
        "download_signature": None,
        "one_shot_signature": None,
        "forward_test_signature": None,
        "selected_groups": initial_groups,
        "selected_tickers": initial_tickers,
        "app_page": "Optimization Studio",
        "last_saved_backtest_id": None,
        "last_saved_backtest_signature": None,
        "forward_test_controls_snapshot": None,
        "forward_test_groups_snapshot": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    existing_groups = list(st.session_state.get("selected_groups", []))
    if "US Mega Cap" in existing_groups:
        st.session_state["selected_groups"] = [
            "US Liquid Leaders" if group == "US Mega Cap" else group
            for group in existing_groups
        ]

    # Migrate older sessions that still carry the previous default groups.
    if set(st.session_state.get("selected_groups", [])) == set(LEGACY_DEFAULT_GROUPS):
        st.session_state["selected_groups"] = list(DEFAULT_GROUPS)
        st.session_state["selected_tickers"] = default_group_tickers(DEFAULT_GROUPS)
    elif set(st.session_state.get("selected_groups", [])) == set(DEFAULT_GROUPS):
        current_tickers = sanitize_tickers(st.session_state.get("selected_tickers", []))
        if len(current_tickers) < len(initial_tickers):
            if set(current_tickers).issubset(set(initial_tickers)):
                st.session_state["selected_tickers"] = initial_tickers


def build_download_signature(controls: Dict[str, object]) -> tuple:
    return (
        tuple(controls["selected"]),
        controls["start_date"],
        controls["end_date"],
        controls["benchmark_symbol"],
        controls["vol_proxy_symbol"],
    )


def build_run_signature(controls: Dict[str, object]) -> tuple:
    return (
        build_download_signature(controls),
        tuple(controls["selected"]),
        tuple(sorted(asdict(controls["alpha_cfg"]).items())),
        tuple(sorted(asdict(controls["construction_cfg"]).items())),
        tuple(sorted(asdict(controls["risk_cfg"]).items())),
        tuple(sorted(asdict(controls["implementation_cfg"]).items())),
        tuple(sorted(asdict(controls["governance_cfg"]).items())),
    )


@st.cache_data(ttl=3600, show_spinner=False)
def cached_market_download(
    tickers: tuple[str, ...],
    start_date: str,
    end_date: str,
    benchmark_symbol: str,
    vol_proxy_symbol: str,
) -> Dict[str, pd.DataFrame]:
    return download_market_data(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        benchmark=benchmark_symbol,
        vol_proxy=vol_proxy_symbol,
    )


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def cached_fundamentals(tickers: tuple[str, ...]) -> pd.DataFrame:
    return fetch_fundamentals(tickers)


def render_header() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown(
        """
        <div class="hero">
            <h1>Portfolio Optimization Studio</h1>
            <p>
                Build a portfolio from a defined universe, rank assets with alpha signals,
                optimize weights, apply risk overlays, and forward-test the process with rolling rebalances.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar_controls() -> Dict[str, object]:
    st.sidebar.header("Portfolio Setup")
    previous_groups = st.session_state.selected_groups
    selected_groups = st.sidebar.multiselect(
        "Universe groups",
        options=list(PRESET_UNIVERSES.keys()),
        default=st.session_state.selected_groups,
    )
    if not selected_groups:
        selected_groups = ["US Liquid Leaders"]

    group_universe = sanitize_tickers(
        ticker
        for group in selected_groups
        for ticker in PRESET_UNIVERSES.get(group, [])
    )
    groups_changed = set(selected_groups) != set(previous_groups)
    if groups_changed:
        default_tickers = group_universe
    else:
        default_tickers = [ticker for ticker in st.session_state.selected_tickers if ticker in group_universe]
    if not default_tickers:
        default_tickers = group_universe[: min(len(group_universe), 12)]

    selected_from_groups = st.sidebar.multiselect(
        "Select tickers from chosen groups",
        options=group_universe,
        default=default_tickers,
    )
    st.sidebar.caption(f"{len(group_universe)} tickers available from {len(selected_groups)} selected groups")
    custom_text = st.sidebar.text_area(
        "Add extra tickers",
        value="",
        placeholder="Example: TSLA, AMD, NFLX, GLD",
    )
    history_preset = st.sidebar.selectbox(
        "History preset",
        [
            "Since 2005 (GFC + COVID + 2022)",
            "20 years",
            "15 years",
            "10 years",
            "5 years",
            "Custom",
        ],
        index=4,
    )
    if history_preset == "Since 2005 (GFC + COVID + 2022)":
        default_start_date = date(2005, 1, 1)
    elif history_preset == "20 years":
        default_start_date = date.today() - timedelta(days=365 * 20)
    elif history_preset == "15 years":
        default_start_date = date.today() - timedelta(days=365 * 15)
    elif history_preset == "10 years":
        default_start_date = date.today() - timedelta(days=365 * 10)
    elif history_preset == "5 years":
        default_start_date = date.today() - timedelta(days=365 * 5)
    else:
        default_start_date = date(2005, 1, 1)

    start_date = st.sidebar.date_input("Start date", value=default_start_date)
    end_date = st.sidebar.date_input("End date", value=date.today())

    st.sidebar.header("Optimization")
    method = st.sidebar.selectbox(
        "Construction method",
        ["Max Sharpe", "Min Volatility", "Risk Parity", "Equal Weight"],
        index=0,
    )
    liquidity_cut = st.sidebar.slider("Liquidity cut", min_value=5, max_value=100, value=100)
    default_alpha_candidate_cap = min(30, liquidity_cut)
    if "alpha_candidate_cap" not in st.session_state:
        st.session_state["alpha_candidate_cap"] = default_alpha_candidate_cap
    st.session_state["alpha_candidate_cap"] = min(int(st.session_state["alpha_candidate_cap"]), liquidity_cut)
    top_n = st.sidebar.slider(
        "Alpha candidate cap",
        min_value=3,
        max_value=liquidity_cut,
        key="alpha_candidate_cap",
    )

    target_holdings_max = min(50, top_n)
    if "target_holdings" not in st.session_state:
        st.session_state["target_holdings"] = min(20, target_holdings_max)
    st.session_state["target_holdings"] = min(int(st.session_state["target_holdings"]), target_holdings_max)
    if target_holdings_max <= 3:
        target_holdings = target_holdings_max
        st.session_state["target_holdings"] = target_holdings
        st.sidebar.caption(f"Target holdings fixed at {target_holdings} because alpha candidate cap is {top_n}.")
    else:
        target_holdings = st.sidebar.slider(
            "Target holdings",
            min_value=3,
            max_value=target_holdings_max,
            key="target_holdings",
        )
    st.sidebar.caption("Hierarchy: liquidity cut -> alpha candidate cap -> target holdings")
    max_weight = st.sidebar.slider("Max single-name weight", min_value=0.05, max_value=0.50, value=0.22, step=0.01)
    min_weight = st.sidebar.slider("Min single-name weight", min_value=0.00, max_value=0.10, value=0.01, step=0.01)
    risk_free_rate = st.sidebar.slider("Risk-free rate", min_value=0.00, max_value=0.10, value=0.03, step=0.005)
    target_vol = st.sidebar.slider("Target volatility", min_value=0.05, max_value=0.35, value=0.18, step=0.01)

    st.sidebar.header("Forward Test")
    lookback_months = st.sidebar.slider("Lookback window (months)", min_value=3, max_value=60, value=12)
    rebalance_months = st.sidebar.slider("Recalculate every Y months", min_value=1, max_value=6, value=1)
    transaction_cost_bps = st.sidebar.slider("Transaction cost (bps)", min_value=0, max_value=100, value=10)
    slippage_bps = st.sidebar.slider("Slippage (bps)", min_value=0, max_value=50, value=2)

    st.sidebar.header("Advanced Framework")
    use_fundamentals = st.sidebar.checkbox("Use Quality / Value / Growth factors", value=False)
    min_adv = st.sidebar.number_input("Min avg dollar volume (USD mn)", min_value=0.0, value=0.0, step=5.0)
    use_historical_eligibility = st.sidebar.checkbox("Historical listing filter", value=True)
    min_listing_months = st.sidebar.slider("Min listing age (months)", min_value=0, max_value=60, value=24, step=3)
    use_trend_filter = st.sidebar.checkbox("Trend confirm overlay", value=True)
    use_regime_filter = st.sidebar.checkbox("Composite regime filter", value=True)
    max_drawdown_stop = st.sidebar.slider("Drawdown stop", min_value=0.00, max_value=0.30, value=0.12, step=0.01)
    assumed_aum = st.sidebar.number_input("Assumed AUM for capacity check", min_value=100_000.0, value=1_000_000.0, step=100_000.0)

    selected = sanitize_tickers(selected_from_groups + parse_ticker_text(custom_text))
    market_reference = infer_market_reference(selected_groups, selected)
    if "benchmark_symbol" not in st.session_state or groups_changed:
        st.session_state["benchmark_symbol"] = market_reference["benchmark"]
    if "vol_proxy_symbol" not in st.session_state or groups_changed:
        st.session_state["vol_proxy_symbol"] = market_reference["vol_proxy"]

    st.sidebar.header("Market References")
    benchmark_symbol = st.sidebar.text_input(
        "Benchmark ticker",
        key="benchmark_symbol",
        help="Examples: SPY for US equities or ^SET.BK for Thailand.",
    ).strip().upper()
    vol_proxy_symbol = st.sidebar.text_input(
        "Volatility proxy ticker",
        key="vol_proxy_symbol",
        help="Examples: ^VIX for US equities. Leave blank to disable the volatility proxy component.",
    ).strip().upper()
    st.session_state.selected_groups = selected_groups
    st.session_state.selected_tickers = selected
    return {
        "selected": selected,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "benchmark_symbol": benchmark_symbol,
        "vol_proxy_symbol": vol_proxy_symbol,
        "alpha_cfg": AlphaConfig(
            top_n=top_n,
            target_holdings=target_holdings,
            liquidity_cut=liquidity_cut,
            min_avg_dollar_volume_millions=min_adv,
            use_fundamental_factors=use_fundamentals,
            use_historical_eligibility=use_historical_eligibility,
            min_listing_days=int(min_listing_months * 21),
        ),
        "construction_cfg": ConstructionConfig(
            method=method,
            min_weight=min_weight,
            max_weight=max_weight,
            risk_free_rate=risk_free_rate,
            target_volatility=target_vol,
        ),
        "risk_cfg": RiskConfig(
            use_trend_filter=use_trend_filter,
            use_regime_filter=use_regime_filter,
            max_drawdown_stop=max_drawdown_stop,
        ),
        "implementation_cfg": ImplementationConfig(
            lookback_months=lookback_months,
            rebalance_months=rebalance_months,
            transaction_cost_bps=float(transaction_cost_bps),
            slippage_bps=float(slippage_bps),
        ),
        "governance_cfg": GovernanceConfig(assumed_aum_usd=float(assumed_aum)),
    }


def render_market_overview(bundle: Dict[str, pd.DataFrame]) -> None:
    prices = bundle["prices"]
    benchmark = bundle["benchmark"]
    missing = bundle["missing"]
    benchmark_symbol = str(bundle.get("benchmark_symbol", "")).strip().upper()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Tickers loaded", f"{prices.shape[1]}")
    col2.metric("Rows", f"{prices.shape[0]}")
    col3.metric("Start", str(prices.index.min().date()))
    col4.metric("End", str(prices.index.max().date()))
    st.caption("Normalized price chart is hidden for large universes to keep the app responsive.")

    if not benchmark.empty:
        regime_fig = go.Figure()
        regime_fig.add_trace(go.Scatter(x=benchmark.index, y=benchmark, mode="lines", name=benchmark_symbol or "Benchmark"))
        regime_fig.update_layout(
            title=f"Benchmark monitor ({benchmark_symbol or 'Benchmark'})",
            xaxis_title="Date",
            yaxis_title="Price",
        )
        st.plotly_chart(regime_fig, use_container_width=True)

    if not missing.empty:
        st.warning(f"Missing tickers from yfinance: {', '.join(missing['Ticker'].tolist())}")


def render_metrics_table(metrics: pd.DataFrame, percent_rows: List[str]) -> None:
    if metrics.empty:
        st.info("No metrics available yet.")
        return
    display = metrics.copy()
    display["Formatted"] = display.apply(
        lambda row: f"{row['Value']:.2%}" if row["Metric"] in percent_rows else f"{row['Value']:.2f}",
        axis=1,
    )
    st.dataframe(display[["Metric", "Formatted"]], use_container_width=True, hide_index=True)


def build_annual_return_summary(
    curve: pd.DataFrame,
    benchmark_curve: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if curve.empty or "PortValue" not in curve.columns:
        return pd.DataFrame(), pd.DataFrame()

    values = curve["PortValue"].dropna().astype(float)
    if len(values) < 2:
        return pd.DataFrame(), pd.DataFrame()

    yearly_end = values.resample("YE").last()
    annual_returns = yearly_end.pct_change().dropna()
    annual_df = annual_returns.rename("Portfolio").to_frame()

    benchmark_annual = pd.Series(dtype=float)
    if benchmark_curve is not None and not benchmark_curve.empty and "PortValue" in benchmark_curve.columns:
        benchmark_values = benchmark_curve["PortValue"].dropna().astype(float)
        if len(benchmark_values) >= 2:
            benchmark_annual = benchmark_values.resample("YE").last().pct_change().dropna()
            annual_df = annual_df.join(benchmark_annual.rename("Benchmark"), how="left")

    annual_df["Active"] = annual_df["Portfolio"] - annual_df.get("Benchmark", pd.Series(index=annual_df.index, dtype=float))
    annual_df.index = annual_df.index.year
    annual_df.index.name = "Year"
    annual_df = annual_df.reset_index()

    total_return = values.iloc[-1] / values.iloc[0] - 1.0
    years = len(values) / 252
    cagr = (values.iloc[-1] / values.iloc[0]) ** (1 / years) - 1 if years > 0 else 0.0
    best_year_idx = annual_returns.idxmax() if not annual_returns.empty else None
    worst_year_idx = annual_returns.idxmin() if not annual_returns.empty else None

    if annual_returns.empty:
        summary = pd.DataFrame(
            [
                {"Statistic": "Min Annual Return", "Value": np.nan},
                {"Statistic": "Max Annual Return", "Value": np.nan},
                {"Statistic": "Best Year", "Value": np.nan},
                {"Statistic": "Worst Year", "Value": np.nan},
                {"Statistic": "CAGR", "Value": cagr},
                {"Statistic": "Total Return", "Value": total_return},
            ]
        )
    else:
        summary = pd.DataFrame(
            [
                {"Statistic": "Min Annual Return", "Value": annual_returns.min()},
                {"Statistic": "Max Annual Return", "Value": annual_returns.max()},
                {"Statistic": "Best Year", "Value": annual_returns.max(), "Label": str(best_year_idx.year)},
                {"Statistic": "Worst Year", "Value": annual_returns.min(), "Label": str(worst_year_idx.year)},
                {"Statistic": "CAGR", "Value": cagr},
                {"Statistic": "Total Return", "Value": total_return},
            ]
        )

    if not benchmark_annual.empty:
        benchmark_total_return = benchmark_curve["PortValue"].iloc[-1] / benchmark_curve["PortValue"].iloc[0] - 1.0
        benchmark_years = len(benchmark_curve) / 252
        benchmark_cagr = (
            (benchmark_curve["PortValue"].iloc[-1] / benchmark_curve["PortValue"].iloc[0]) ** (1 / benchmark_years) - 1
            if benchmark_years > 0
            else 0.0
        )
        summary = pd.concat(
            [
                summary,
                pd.DataFrame(
                    [
                        {"Statistic": "Benchmark CAGR", "Value": benchmark_cagr},
                        {"Statistic": "Benchmark Total Return", "Value": benchmark_total_return},
                    ]
                ),
            ],
            ignore_index=True,
        )
    return annual_df, summary


def render_one_shot(result: Dict[str, object]) -> None:
    st.subheader("One-Shot Optimization")
    weights = result["weights"].reset_index().rename(columns={"index": "Ticker"})
    regime = result["regime"]
    curve = result["portfolio_curve"]
    candidate_count = len(result.get("candidate_table", pd.DataFrame()))
    liquidity_universe_size = int(result.get("liquidity_universe_size", candidate_count))
    target_holdings_count = len(result["selected_assets"])

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Liquidity universe", str(liquidity_universe_size))
    col2.metric("Alpha candidates", str(candidate_count))
    col3.metric("Target holdings", str(target_holdings_count))
    col4.metric("Regime", regime["regime"])
    col5.metric("Effective exposure", f"{result['effective_exposure']:.0%}")
    st.caption("Flow: liquidity cut -> alpha candidates -> target holdings -> optimize")

    left, right = st.columns([1.1, 1.2])
    with left:
        st.dataframe(weights, use_container_width=True, hide_index=True)
        pie_weights = weights[weights["Invested Weight"] > 0]
        pie = px.pie(pie_weights, names="Ticker", values="Invested Weight", title="Invested weights")
        st.plotly_chart(pie, use_container_width=True)
    with right:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=curve.index, y=curve["PortValue"], mode="lines", name="Portfolio"))
        fig.update_layout(title="In-sample portfolio value", xaxis_title="Date", yaxis_title="Portfolio value")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(result["factor_exposure"], use_container_width=True)

    render_metrics_table(
        result["metrics"],
        percent_rows=["Total Return", "CAGR", "Annual Volatility", "Max Drawdown", "Hit Rate"],
    )
    st.caption("Alpha ranking below shows the full universe score before portfolio construction. `in_candidate_set` and `selected_for_portfolio` mark the two-stage selection flow.")
    st.dataframe(result["alpha_table"], use_container_width=True)


def render_forward_test(result: Dict[str, object]) -> None:
    st.subheader("Rolling Forward Test")
    curve = result["curve"]

    if curve.empty:
        st.info("Forward test returned no investable windows for the chosen settings.")
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curve.index, y=curve["PortValue"], mode="lines", name="Forward test"))
    fig.update_layout(title="Portfolio value through rolling re-optimization", xaxis_title="Date", yaxis_title="Portfolio value")
    st.plotly_chart(fig, use_container_width=True)

    render_metrics_table(
        result["metrics"],
        percent_rows=["Total Return", "CAGR", "Annual Volatility", "Max Drawdown", "Hit Rate"],
    )

    annual_df, annual_summary = build_annual_return_summary(curve, result.get("benchmark_curve"))

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Rebalances", "Weights", "Annual Returns", "Stress Tests", "Governance"])
    with tab1:
        st.dataframe(result["rebalance_report"], use_container_width=True)
    with tab2:
        if result["weight_history"].empty:
            st.info("No weight history captured.")
        else:
            weight_history = result["weight_history"].copy()
            weight_pivot = (
                weight_history.pivot_table(
                    index="Date",
                    columns="Asset",
                    values="Weight",
                    aggfunc="sum",
                    fill_value=0.0,
                )
                .sort_index()
            )

            stacked = go.Figure()
            for asset in weight_pivot.columns:
                stacked.add_trace(
                    go.Bar(
                        x=weight_pivot.index,
                        y=weight_pivot[asset],
                        name=asset,
                        hovertemplate="%{x|%Y-%m-%d}<br>%{fullData.name}: %{y:.2%}<extra></extra>",
                    )
                )
            stacked.update_layout(
                title="Weight evolution by rebalance date",
                xaxis_title="Date",
                yaxis_title="Weight (%)",
                barmode="stack",
                yaxis_tickformat=".0%",
            )
            st.plotly_chart(stacked, use_container_width=True)

            display_weights = weight_history.copy()
            display_weights["Weight"] = display_weights["Weight"].map(lambda x: f"{x:.2%}")
            st.dataframe(display_weights, use_container_width=True, hide_index=True)
    with tab3:
        if annual_df.empty:
            st.info("ยังมีข้อมูลไม่พอสำหรับสรุป annual return รายปี")
        else:
            chart_df = annual_df.copy()
            long_annual = chart_df.melt(
                id_vars="Year",
                value_vars=[column for column in ["Portfolio", "Benchmark"] if column in chart_df.columns],
                var_name="Series",
                value_name="Annual Return",
            )
            annual_chart = px.bar(
                long_annual,
                x="Year",
                y="Annual Return",
                color="Series",
                barmode="group",
                text=long_annual["Annual Return"].map(lambda x: f"{x:.1%}"),
                title="Annual returns by calendar year",
            )
            annual_chart.update_traces(textposition="outside", hovertemplate="Year %{x}<br>%{fullData.name}: %{y:.2%}<extra></extra>")
            annual_chart.update_layout(yaxis_title="Return (%)", yaxis_tickformat=".0%")
            st.plotly_chart(annual_chart, use_container_width=True)

            if "Benchmark" in annual_df.columns:
                active_chart = px.bar(
                    annual_df,
                    x="Year",
                    y="Active",
                    text=annual_df["Active"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "-"),
                    title="Active return vs benchmark",
                )
                active_chart.update_traces(
                    marker_color=["#115e59" if (pd.notna(value) and value >= 0) else "#dc2626" for value in annual_df["Active"]],
                    textposition="outside",
                    hovertemplate="Year %{x}<br>Active return %{y:.2%}<extra></extra>",
                )
                active_chart.update_layout(yaxis_title="Active Return (%)", yaxis_tickformat=".0%")
                st.plotly_chart(active_chart, use_container_width=True)

            best_row = annual_df.loc[annual_df["Portfolio"].idxmax()]
            worst_row = annual_df.loc[annual_df["Portfolio"].idxmin()]
            col1, col2, col3 = st.columns(3)
            col1.metric("Best year", str(int(best_row["Year"])), f"{best_row['Portfolio']:.2%}")
            col2.metric("Worst year", str(int(worst_row["Year"])), f"{worst_row['Portfolio']:.2%}")
            cagr_value = annual_summary.loc[annual_summary["Statistic"] == "CAGR", "Value"].iloc[0]
            col3.metric("CAGR", f"{cagr_value:.2%}")

            summary_display = annual_summary.copy()
            if "Label" not in summary_display.columns:
                summary_display["Label"] = ""
            summary_display["Value"] = summary_display["Value"].map(lambda x: "-" if pd.isna(x) else f"{x:.2%}")
            summary_display["Statistic"] = summary_display.apply(
                lambda row: f"{row['Statistic']} ({row['Label']})" if str(row["Label"]).strip() else row["Statistic"],
                axis=1,
            )
            st.dataframe(summary_display, use_container_width=True, hide_index=True)

            annual_table = annual_df.copy()
            for column in ["Portfolio", "Benchmark", "Active"]:
                if column in annual_table.columns:
                    annual_table[column] = annual_table[column].map(lambda x: "-" if pd.isna(x) else f"{x:.2%}")
            st.dataframe(annual_table, use_container_width=True, hide_index=True)

            st.download_button(
                "Download annual returns CSV",
                data=annual_df.to_csv(index=False),
                file_name="annual_returns_summary.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.download_button(
                "Download annual stats CSV",
                data=annual_summary.to_csv(index=False),
                file_name="annual_return_stats.csv",
                mime="text/csv",
                use_container_width=True,
            )
    with tab4:
        stress = result["stress_tests"]
        if stress.empty:
            st.info("The backtest window does not overlap the configured stress scenarios.")
        else:
            st.dataframe(stress, use_container_width=True)
    with tab5:
        governance = result["governance"]
        if governance.empty:
            st.info("Governance checks are not available.")
        else:
            st.dataframe(governance, use_container_width=True, hide_index=True)


def render_framework_notes() -> None:
    st.subheader("Advanced Framework")
    cols = st.columns(5)
    cards = [
        (
            "1. Alpha",
            "Define the universe first, then rank assets with a composite signal engine. This app blends momentum, optional quality, value, growth, and low-volatility signals before selecting Top-N names.",
        ),
        (
            "2. Construction",
            "Turn signals into a real portfolio. Choose equal weight, max Sharpe, min volatility, or risk parity with concentration constraints and target volatility controls.",
        ),
        (
            "3. Risk",
            "Protect the system, not just the picks. The app applies trend confirmation, a composite regime filter using your chosen benchmark, breadth, drawdown, and an optional volatility proxy, plus an optional drawdown stop.",
        ),
        (
            "4. Implementation",
            "Model real-world frictions. Forward tests use rolling lookback windows, fixed rebalance intervals, turnover, transaction cost, and slippage assumptions.",
        ),
        (
            "5. Governance",
            "Check survivability. Capacity proxy, turnover monitoring, concentration checks, and stress windows are surfaced so the framework is more than a stock screener.",
        ),
    ]
    for col, (title, body) in zip(cols, cards):
        with col:
            st.markdown(
                f"""
                <div class="layer-card">
                    <h4>{title}</h4>
                    <div class="small-note">{body}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_backtest_records_page() -> None:
    st.subheader("Backtest Records")
    st.caption(f"Stored in `{BACKTEST_RECORDS_FILE}`")
    records = load_backtest_records()
    if records.empty:
        st.info("No backtest records saved yet. Run a rolling forward test to create the first record.")
        return

    records = records.sort_values("saved_at", ascending=False).reset_index(drop=True)

    filter_col1, filter_col2 = st.columns([1.1, 0.9])
    with filter_col1:
        methods = sorted(records["construction_method"].dropna().unique().tolist())
        selected_methods = st.multiselect("Filter by method", methods, default=methods)
    with filter_col2:
        rows_cap = min(200, len(records))
        if rows_cap <= 5:
            max_rows = rows_cap
            st.caption(f"Showing all {rows_cap} saved run(s).")
        else:
            max_rows = st.slider("Rows to display", min_value=5, max_value=rows_cap, value=min(25, rows_cap))

    filtered = records.copy()
    if selected_methods:
        filtered = filtered[filtered["construction_method"].isin(selected_methods)]
    filtered = filtered.head(max_rows).copy()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Saved runs", str(len(records)))
    col2.metric("Best CAGR", f"{records['cagr'].max():.2%}" if records["cagr"].notna().any() else "-")
    col3.metric("Best Sharpe", f"{records['sharpe'].max():.2f}" if records["sharpe"].notna().any() else "-")
    col4.metric("Worst drawdown", f"{records['max_drawdown'].min():.2%}" if records["max_drawdown"].notna().any() else "-")

    scatter = px.scatter(
        filtered,
        x="max_drawdown",
        y="cagr",
        color="construction_method",
        size="target_holdings",
        hover_data=[
            "saved_at",
            "selected_groups",
            "liquidity_cut",
            "alpha_candidate_cap",
            "target_holdings",
            "lookback_months",
            "rebalance_months",
        ],
        title="CAGR vs Max Drawdown",
    )
    scatter.update_layout(xaxis_tickformat=".0%", yaxis_tickformat=".0%")
    st.plotly_chart(scatter, use_container_width=True)

    sharpe_chart = filtered.copy()
    sharpe_chart = sharpe_chart.loc[sharpe_chart["sharpe"].notna()].copy()
    if not sharpe_chart.empty:
        sharpe_chart["saved_at_label"] = sharpe_chart["saved_at"].apply(
            lambda value: pd.Timestamp(value).strftime("%Y-%m-%d %H:%M")
            if pd.notna(value)
            else "Unknown time"
        )
        sharpe_chart["run_label"] = sharpe_chart.apply(
            lambda row: f"{row['construction_method']} | {row['saved_at_label']}",
            axis=1,
        )
        sharpe_chart = sharpe_chart.sort_values("sharpe", ascending=False)
        sharpe_fig = px.bar(
            sharpe_chart,
            x="run_label",
            y="sharpe",
            color="construction_method",
            hover_data=[
                "saved_at",
                "selected_groups",
                "cagr",
                "max_drawdown",
                "target_holdings",
                "lookback_months",
                "rebalance_months",
            ],
            title="Sharpe by saved run",
        )
        sharpe_fig.update_layout(
            xaxis_title="Saved run",
            yaxis_title="Sharpe",
            xaxis={"categoryorder": "array", "categoryarray": sharpe_chart["run_label"].tolist()},
        )
        st.plotly_chart(sharpe_fig, use_container_width=True)

    display = filtered.copy()
    preferred_order = [
        "construction_method",
        "selected_groups",
        "start_date",
        "end_date",
        "first_rebalance",
        "last_rebalance",
        "liquidity_cut",
        "alpha_candidate_cap",
        "target_holdings",
        "lookback_months",
        "rebalance_months",
        "final_portfolio_value",
        "total_return",
        "cagr",
        "annual_volatility",
        "sharpe",
        "sortino",
        "max_drawdown",
        "hit_rate",
        "saved_at",
        "run_id",
    ]
    ordered_columns = preferred_order + [column for column in display.columns if column not in preferred_order]
    display = display.loc[:, [column for column in ordered_columns if column in display.columns]]
    for column in ["total_return", "cagr", "annual_volatility", "max_drawdown", "hit_rate"]:
        if column in display.columns:
            display[column] = display[column].map(lambda x: "-" if pd.isna(x) else f"{x:.2%}")
    for column in ["sharpe", "sortino"]:
        if column in display.columns:
            display[column] = display[column].map(lambda x: "-" if pd.isna(x) else f"{x:.2f}")
    st.dataframe(display, use_container_width=True, hide_index=True)
    st.download_button(
        "Download backtest records CSV",
        data=records.to_csv(index=False),
        file_name="backtest_records.csv",
        mime="text/csv",
        use_container_width=True,
    )


def render_retirement_page(forward_test_result: Dict[str, object] | None) -> None:
    st.subheader("Retirement")
    st.caption("Estimate how much can be withdrawn each month without exhausting the portfolio under simulated retirement paths.")

    if not forward_test_result or not isinstance(forward_test_result, dict):
        st.info("Run `Run rolling forward test` first to create a return history for the retirement page.")
        return

    curve = forward_test_result.get("curve", pd.DataFrame())
    metrics = forward_test_result.get("metrics", pd.DataFrame())
    if curve.empty or "PortValue" not in curve.columns:
        st.info("Forward test curve is not available yet. Run a rolling forward test first.")
        return

    monthly_values = curve["PortValue"].resample("ME").last().dropna()
    monthly_returns = monthly_values.pct_change().dropna()
    if len(monthly_returns) < 24:
        st.warning("Retirement simulation needs more monthly history. Try a longer backtest window first.")
        return

    source_cagr = metric_value(metrics, "CAGR")
    source_vol = metric_value(metrics, "Annual Volatility")
    source_total_return = metric_value(metrics, "Total Return")
    source_max_drawdown = metric_value(metrics, "Max Drawdown")

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Source CAGR", "-" if pd.isna(source_cagr) else f"{source_cagr:.2%}")
    s2.metric("Source Volatility", "-" if pd.isna(source_vol) else f"{source_vol:.2%}")
    s3.metric("Source Total Return", "-" if pd.isna(source_total_return) else f"{source_total_return:.2%}")
    s4.metric("Source Max Drawdown", "-" if pd.isna(source_max_drawdown) else f"{source_max_drawdown:.2%}")
    st.caption(
        "Retirement simulation source: the latest rolling forward test in this session. "
        "Monte Carlo draws monthly returns from the latest forward test CAGR and annual volatility."
    )

    col1, col2, col3 = st.columns(3)
    initial_portfolio = col1.number_input("Initial portfolio", min_value=1000.0, value=1_000_000.0, step=50_000.0)
    retirement_years = col2.slider("Retirement horizon (years)", min_value=5, max_value=50, value=30)
    target_success_rate = col3.slider("Target survival rate", min_value=0.50, max_value=1.00, value=0.90, step=0.01)

    col4, col5, col6 = st.columns(3)
    annual_inflation = col4.slider("Annual inflation", min_value=0.00, max_value=0.10, value=0.03, step=0.005)
    monthly_income = col5.number_input("Monthly pension / other income", min_value=0.0, value=0.0, step=1000.0)
    num_scenarios = col6.slider("Simulation scenarios", min_value=200, max_value=5000, value=1000, step=100)

    col7, col8 = st.columns(2)
    block_size = col7.slider("Bootstrap block size (months)", min_value=1, max_value=24, value=12)
    custom_withdrawal = col8.number_input("Test custom monthly withdrawal", min_value=0.0, value=40_000.0, step=1_000.0)

    if st.button("Run retirement test", use_container_width=True):
        with st.spinner("Running retirement survival simulation..."):
            bootstrap_sustainable = find_sustainable_monthly_withdrawal(
                monthly_returns=monthly_returns,
                initial_portfolio=initial_portfolio,
                years=retirement_years,
                annual_inflation=annual_inflation,
                monthly_income=monthly_income,
                target_success_rate=target_success_rate,
                num_scenarios=num_scenarios,
                block_size=block_size,
            )
            monte_carlo_sustainable = find_sustainable_monthly_withdrawal_monte_carlo(
                annual_cagr=0.0 if pd.isna(source_cagr) else float(source_cagr),
                annual_volatility=0.0 if pd.isna(source_vol) else float(source_vol),
                initial_portfolio=initial_portfolio,
                years=retirement_years,
                annual_inflation=annual_inflation,
                monthly_income=monthly_income,
                target_success_rate=target_success_rate,
                num_scenarios=num_scenarios,
            )
            custom_result = simulate_retirement_paths_monte_carlo(
                annual_cagr=0.0 if pd.isna(source_cagr) else float(source_cagr),
                annual_volatility=0.0 if pd.isna(source_vol) else float(source_vol),
                initial_portfolio=initial_portfolio,
                monthly_withdrawal=custom_withdrawal,
                years=retirement_years,
                annual_inflation=annual_inflation,
                monthly_income=monthly_income,
                num_scenarios=num_scenarios,
            )
            bootstrap_custom_result = simulate_retirement_paths(
                monthly_returns=monthly_returns,
                initial_portfolio=initial_portfolio,
                monthly_withdrawal=custom_withdrawal,
                years=retirement_years,
                annual_inflation=annual_inflation,
                monthly_income=monthly_income,
                num_scenarios=num_scenarios,
                block_size=block_size,
            )

        monte_carlo_result = monte_carlo_sustainable["result"]
        bootstrap_result = bootstrap_sustainable["result"]
        safe_monthly = float(monte_carlo_sustainable["monthly_withdrawal"])
        bootstrap_safe_monthly = float(bootstrap_sustainable["monthly_withdrawal"])
        safe_annual = safe_monthly * 12.0
        bootstrap_safe_annual = bootstrap_safe_monthly * 12.0
        initial_withdrawal_rate = safe_annual / initial_portfolio if initial_portfolio > 0 else 0.0
        custom_survival = float(custom_result["survival_rate"])
        bootstrap_custom_survival = float(bootstrap_custom_result["survival_rate"])

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("MC safe monthly withdrawal", f"{safe_monthly:,.0f}")
        m2.metric("MC safe annual withdrawal", f"{safe_annual:,.0f}")
        m3.metric("Initial withdrawal rate", f"{initial_withdrawal_rate:.2%}")
        m4.metric("MC custom withdrawal survival", f"{custom_survival:.2%}")

        final_values = monte_carlo_result["final_values"]
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("MC safe withdrawal survival", f"{float(monte_carlo_result['survival_rate']):.2%}")
        b2.metric("Bootstrap safe monthly", f"{bootstrap_safe_monthly:,.0f}")
        b3.metric("Bootstrap safe survival", f"{float(bootstrap_result['survival_rate']):.2%}")
        b4.metric("Bootstrap custom survival", f"{bootstrap_custom_survival:.2%}")

        s1, s2, s3 = st.columns(3)
        s1.metric("MC median ending value", f"{float(final_values.quantile(0.5)):,.0f}")
        s2.metric("MC 10th pct ending value", f"{float(final_values.quantile(0.1)):,.0f}")
        s3.metric("Bootstrap safe annual", f"{bootstrap_safe_annual:,.0f}")

        safe_paths = monte_carlo_result["simulated_paths"]
        percentile_df = pd.DataFrame(
            {
                "Month": np.arange(len(safe_paths)),
                "P10": safe_paths.quantile(0.10, axis=1).values,
                "Median": safe_paths.quantile(0.50, axis=1).values,
                "P90": safe_paths.quantile(0.90, axis=1).values,
            }
        )
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=percentile_df["Month"], y=percentile_df["P10"], mode="lines", name="P10"))
        fig.add_trace(go.Scatter(x=percentile_df["Month"], y=percentile_df["Median"], mode="lines", name="Median"))
        fig.add_trace(go.Scatter(x=percentile_df["Month"], y=percentile_df["P90"], mode="lines", name="P90"))
        fig.update_layout(
            title="Monte Carlo retirement portfolio path percentiles",
            xaxis_title="Month",
            yaxis_title="Portfolio value",
        )
        st.plotly_chart(fig, use_container_width=True)

        withdrawal_schedule = monte_carlo_result["gross_withdrawals"].quantile(0.5, axis=1).rename("Median Withdrawal").reset_index()
        withdrawal_schedule.columns = ["Month", "Median Withdrawal"]
        st.dataframe(withdrawal_schedule.head(24), use_container_width=True, hide_index=True)
        st.caption("Table above shows the first 24 months of the inflation-adjusted median withdrawal schedule from the Monte Carlo retirement test.")


def main() -> None:
    init_state()
    st.session_state["app_page"] = st.sidebar.radio(
        "Page",
        ["Optimization Studio", "Backtest Records", "Retirement"],
        index=["Optimization Studio", "Backtest Records", "Retirement"].index(
            st.session_state.get("app_page", "Optimization Studio")
        )
        if st.session_state.get("app_page", "Optimization Studio") in ["Optimization Studio", "Backtest Records", "Retirement"]
        else 0,
    )
    render_header()
    if st.session_state["app_page"] == "Backtest Records":
        render_backtest_records_page()
        return
    if st.session_state["app_page"] == "Retirement":
        render_retirement_page(st.session_state.get("forward_test_result"))
        return

    controls = sidebar_controls()
    render_framework_notes()

    if "US Liquid Leaders" in st.session_state.selected_groups:
        st.info(
            "US Liquid Leaders now uses local S&P 500 point-in-time membership from the sibling `sp500` repo, then ranks "
            "members by average dollar volume and keeps the top liquidity names for each rebalance period. Historical listing "
            "filter still helps reduce bias when free price history is incomplete."
        )
    if {"Thailand SET100", "ThaiSET100"} & set(st.session_state.selected_groups):
        st.info(
            "Thailand SET100 / ThaiSET100 now loads a historical superset from local semiannual SET100 documents, then applies point-in-time "
            "membership at each rebalance date before ranking liquidity. Older 2005-2013 source files are less structured, so "
            "some half-year snapshots may contain fewer than 100 parsed names, but this still reduces survivorship bias materially "
            "versus using today's SET100 members for the full history."
        )

    if not controls["selected"]:
        st.info("Select at least one ticker to begin.")
        return
    if controls["start_date"] >= controls["end_date"]:
        st.error("Start date must be earlier than end date.")
        return
    if controls["alpha_cfg"].top_n > controls["alpha_cfg"].liquidity_cut:
        st.error("Alpha candidate cap cannot be greater than the liquidity cut.")
        return
    if controls["alpha_cfg"].target_holdings > controls["alpha_cfg"].top_n:
        st.error("Target holdings cannot be greater than the alpha candidate cap.")
        return
    effective_target_holdings = min(controls["alpha_cfg"].target_holdings, len(controls["selected"]))
    min_weight_budget = controls["construction_cfg"].min_weight * effective_target_holdings
    if min_weight_budget > 1.0:
        st.error("Min weight times effective target holdings exceeds 100%, so the optimization would be infeasible.")
        return
    if np.isclose(min_weight_budget, 1.0):
        st.error(
            "Min weight times effective target holdings is exactly 100%, which forces the optimizer into equal weights. "
            "Reduce Top-N or lower the minimum weight."
        )
        return
    if controls["construction_cfg"].max_weight * effective_target_holdings < 1.0:
        st.error("Max weight times effective target holdings is below 100%, so the portfolio cannot be fully invested.")
        return

    action_col1, action_col2, action_col3 = st.columns([1, 1, 2])
    with action_col1:
        download_clicked = st.button("Download market data", use_container_width=True)
    with action_col2:
        clear_clicked = st.button("Clear results", use_container_width=True)
    with action_col3:
        st.caption("Workflow: download -> optional fundamental refresh -> one-shot optimization -> rolling forward test")

    if clear_clicked:
        st.session_state.market_bundle = None
        st.session_state.fundamentals = None
        st.session_state.one_shot_result = None
        st.session_state.forward_test_result = None
        st.session_state.download_signature = None
        st.session_state.one_shot_signature = None
        st.session_state.forward_test_signature = None
        st.session_state.last_saved_backtest_id = None
        st.session_state.last_saved_backtest_signature = None
        st.session_state.forward_test_controls_snapshot = None
        st.session_state.forward_test_groups_snapshot = None
        st.rerun()

    if download_clicked:
        with st.spinner("Downloading market data from yfinance..."):
            st.session_state.market_bundle = cached_market_download(
                tuple(controls["selected"]),
                controls["start_date"],
                controls["end_date"],
                controls["benchmark_symbol"],
                controls["vol_proxy_symbol"],
            )
        st.session_state.download_signature = build_download_signature(controls)
        st.session_state.fundamentals = None
        st.session_state.one_shot_result = None
        st.session_state.forward_test_result = None
        st.session_state.one_shot_signature = None
        st.session_state.forward_test_signature = None
        st.session_state.last_saved_backtest_id = None
        st.session_state.last_saved_backtest_signature = None
        st.session_state.forward_test_controls_snapshot = None
        st.session_state.forward_test_groups_snapshot = None

    bundle = st.session_state.market_bundle
    if not bundle:
        st.info("Download data to start the optimization workflow.")
        return

    current_download_signature = build_download_signature(controls)
    current_run_signature = build_run_signature(controls)
    if st.session_state.download_signature != current_download_signature:
        st.warning("Market data on screen is from older portfolio setup inputs. Adjust as needed, then click `Download market data` to refresh.")

    render_market_overview(bundle)

    run_col1, run_col2 = st.columns(2)
    with run_col1:
        if st.button("Run one-shot optimization", use_container_width=True):
            if controls["alpha_cfg"].use_fundamental_factors:
                with st.spinner("Fetching fundamental metrics from yfinance..."):
                    st.session_state.fundamentals = cached_fundamentals(tuple(controls["selected"]))
            with st.spinner("Running one-shot optimization..."):
                st.session_state.one_shot_result = run_one_shot_optimization(
                    prices=bundle["prices"],
                    volumes=bundle["volumes"],
                    benchmark=bundle["benchmark"],
                    vol_proxy=bundle["vol_proxy"],
                    alpha_cfg=controls["alpha_cfg"],
                    construction_cfg=controls["construction_cfg"],
                    risk_cfg=controls["risk_cfg"],
                    fundamentals=st.session_state.fundamentals,
                )
                st.session_state.one_shot_signature = current_run_signature
    with run_col2:
        if st.button("Run rolling forward test", use_container_width=True):
            if controls["alpha_cfg"].use_fundamental_factors:
                with st.spinner("Fetching fundamental metrics from yfinance..."):
                    st.session_state.fundamentals = cached_fundamentals(tuple(controls["selected"]))
            progress_text = st.empty()
            progress_bar = st.progress(0)

            def update_forward_progress(step: int, total: int, rebalance_date: pd.Timestamp) -> None:
                total = max(total, 1)
                progress_bar.progress(int(step / total * 100))
                progress_text.caption(
                    f"Forward test progress: {step}/{total} rebalances processed "
                    f"(current window ending {pd.Timestamp(rebalance_date).date()})"
                )

            with st.spinner("Running rolling forward test..."):
                st.session_state.forward_test_result = run_forward_test(
                    prices=bundle["prices"],
                    volumes=bundle["volumes"],
                    benchmark=bundle["benchmark"],
                    vol_proxy=bundle["vol_proxy"],
                    alpha_cfg=controls["alpha_cfg"],
                    construction_cfg=controls["construction_cfg"],
                    risk_cfg=controls["risk_cfg"],
                    implementation_cfg=controls["implementation_cfg"],
                    governance_cfg=controls["governance_cfg"],
                    fundamentals=st.session_state.fundamentals,
                    progress_callback=update_forward_progress,
                )
                progress_bar.progress(100)
                progress_text.caption("Forward test completed.")
                st.session_state.forward_test_signature = current_run_signature
                st.session_state.forward_test_controls_snapshot = controls
                st.session_state.forward_test_groups_snapshot = list(st.session_state.selected_groups)
                st.session_state.last_saved_backtest_id = None
                st.session_state.last_saved_backtest_signature = None

    st.caption(
        "Changing sidebar parameters will not re-run optimization automatically. "
        "Adjust several inputs first, then click `Run one-shot optimization` or `Run rolling forward test` when ready."
    )

    if st.session_state.forward_test_result:
        save_col1, save_col2 = st.columns([1, 3])
        already_saved = st.session_state.last_saved_backtest_signature == st.session_state.forward_test_signature
        with save_col1:
            save_clicked = st.button(
                "Save backtest record",
                use_container_width=True,
                disabled=already_saved,
            )
        with save_col2:
            if already_saved and st.session_state.last_saved_backtest_id:
                st.success(f"Current forward test already saved as `{st.session_state.last_saved_backtest_id}`.")
            else:
                st.caption("Backtest results are saved only when you click `Save backtest record`.")

        if save_clicked:
            controls_to_save = st.session_state.forward_test_controls_snapshot or controls
            groups_to_save = st.session_state.forward_test_groups_snapshot or list(st.session_state.selected_groups)
            st.session_state.last_saved_backtest_id = append_backtest_record(
                controls=controls_to_save,
                result=st.session_state.forward_test_result,
                selected_groups=groups_to_save,
            )
            st.session_state.last_saved_backtest_signature = st.session_state.forward_test_signature
            st.rerun()

    if st.session_state.one_shot_result:
        if st.session_state.one_shot_signature != current_run_signature:
            st.warning("One-shot optimization results below are from older settings. Adjust your parameters, then click `Run one-shot optimization` to refresh.")
        render_one_shot(st.session_state.one_shot_result)
    if st.session_state.forward_test_result:
        if st.session_state.forward_test_signature != current_run_signature:
            st.warning("Rolling forward test results below are from older settings. Adjust your parameters, then click `Run rolling forward test` to refresh.")
        render_forward_test(st.session_state.forward_test_result)


if __name__ == "__main__":
    main()
