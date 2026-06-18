from __future__ import annotations

import contextlib
import io
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yfinance as yf


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in [SRC, SCRIPTS]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

YFINANCE_CACHE = ROOT / "data" / "cache" / "dynamic_factor_copula" / ".yfinance"
YFINANCE_CACHE.mkdir(parents=True, exist_ok=True)
yf.set_tz_cache_location(str(YFINANCE_CACHE))

from dynamic_factor_copula import (  # noqa: E402
    build_factor_covariance,
    build_momentum_signal,
    compute_feature_table,
    compute_market_stress_signal,
    default_paths,
    drop_duplicate_share_classes,
    drop_duplicate_share_classes_available,
    get_set100_members_as_of,
    get_sp500_members_as_of,
    initialize_static_clusters,
    load_cached_market_data,
    load_set100_membership_intervals,
    load_sp500_membership_intervals,
    monthly_rebalance_dates,
    optimize_portfolio,
    optimize_risk_parity,
    run_dynamic_hmm,
    select_point_in_time_universe,
)


START_DATE = "2016-01-01"
LOOKBACK_DAYS = 504
N_CLUSTERS = 4
US_ASSETS = 30
TH_ASSETS = 30
STOCK_CAP = 0.08
PRIMARY_MODEL = "Dynamic HMM Copula"
RISK_FREE_RATE = 0.03
SELECTED_MIX = {"Equity": 0.65, "Gold": 0.25, "BTC": 0.10}
STRATEGY = "Final Best Sharpe Tactical TH/Gold/BTC 65/25/10 Gold crash protection"
RESULT_PREFIX = "us_th_tactical_perf_momentum_final_best"
FEATURE_FLAGS = {"resid_vol": False, "drawdown": False, "downside_beta": False}
FRESH_LOOKBACK_CALENDAR_DAYS = 850
FRESH_BATCH_SIZE = 80
OVERLAY_TICKERS = ["SPY", "^VIX", "GC=F", "BTC-USD", "USDTHB=X", "^SET.BK"]
GOLD_DD_WINDOW = 252
GOLD_WARN_DD = -0.08
GOLD_WARN_EXPOSURE = 0.50
GOLD_CRASH_DD = -0.20
GOLD_CRASH_EXPOSURE = 0.50
GOLD_RECOVERY_DD = -0.05
GOLD_PANIC_DD = -0.30
GOLD_PANIC_MA = 200
GOLD_PANIC_MOM = 63


def _fresh_start_date() -> str:
    return (pd.Timestamp.today().normalize() - pd.Timedelta(days=FRESH_LOOKBACK_CALENDAR_DAYS)).date().isoformat()


def _cached_columns(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {name for name in pq.ParquetFile(path).schema.names if not name.startswith("__")}


def _membership_frame(kind: str) -> pd.DataFrame:
    paths = default_paths(ROOT)
    if kind == "us":
        intervals = load_sp500_membership_intervals(paths)
        fallback = ROOT / "data" / "sp500" / "sp500_ticker_start_end.csv"
    else:
        intervals = load_set100_membership_intervals(paths)
        fallback = ROOT / "data" / "thai_stock" / "set100_ticker_start_end.csv"
    if not intervals.empty or not fallback.exists():
        return intervals
    intervals = pd.read_csv(fallback).rename(columns=str.lower)
    intervals["ticker"] = intervals["ticker"].astype(str).str.upper()
    if kind == "th":
        intervals["ticker"] = intervals["ticker"].map(lambda value: value if value.endswith(".BK") else f"{value}.BK")
    intervals["start_date"] = pd.to_datetime(intervals["start_date"], errors="coerce")
    intervals["end_date"] = pd.to_datetime(intervals["end_date"], errors="coerce")
    return intervals.dropna(subset=["ticker", "start_date"])


def _all_available_members(kind: str) -> list[str]:
    paths = default_paths(ROOT)
    intervals = _membership_frame(kind)
    source_cols = _cached_columns(paths.source_cache_root / "prices.parquet")
    local_cols = _cached_columns(paths.local_cache_root / "extra_prices.parquet")
    available = source_cols | local_cols
    members = intervals["ticker"].dropna().astype(str).str.upper().drop_duplicates().tolist()
    return drop_duplicate_share_classes_available([ticker for ticker in members if ticker in available], available)


def _active_members(kind: str, as_of: pd.Timestamp, available: list[str]) -> list[str]:
    paths = default_paths(ROOT)
    if kind == "us":
        active = get_sp500_members_as_of(as_of, paths)
        if not active:
            intervals = _membership_frame("us")
            mask = (intervals["start_date"] <= as_of) & (intervals["end_date"].isna() | (intervals["end_date"] >= as_of))
            active = intervals.loc[mask, "ticker"].dropna().astype(str).tolist()
    else:
        active = get_set100_members_as_of(as_of, paths)
        if not active:
            intervals = _membership_frame("th")
            mask = (intervals["start_date"] <= as_of) & (intervals["end_date"].isna() | (intervals["end_date"] >= as_of))
            active = intervals.loc[mask, "ticker"].dropna().astype(str).tolist()
    return drop_duplicate_share_classes_available([ticker for ticker in active if ticker in available], available)


def _load_overlay() -> pd.DataFrame:
    paths = default_paths(ROOT)
    overlay = pd.read_parquet(paths.local_cache_root / "overlay_compare_prices.parquet").sort_index().ffill()
    extra_prices = paths.local_cache_root / "extra_prices.parquet"
    if extra_prices.exists():
        extra = pd.read_parquet(extra_prices).sort_index()
        for column in ["^SET.BK"]:
            if column in extra.columns:
                overlay[column] = extra[column]
    required = ["SPY", "^VIX", "GC=F", "BTC-USD", "USDTHB=X", "^SET.BK"]
    missing = [column for column in required if column not in overlay.columns]
    if missing:
        raise RuntimeError(f"Missing required overlay columns: {missing}")
    return overlay.loc[START_DATE:, required].ffill()


def _latest_common_close(overlay: pd.DataFrame) -> pd.Timestamp:
    common = overlay.dropna(subset=["SPY", "^VIX", "GC=F", "BTC-USD", "USDTHB=X", "^SET.BK"])
    if common.empty:
        raise RuntimeError("No common close exists for SPY, VIX, Gold, BTC, USDTHB, and SET.")
    return pd.Timestamp(common.index.max())


def _source_close_date(index: pd.Index, effective_date: pd.Timestamp) -> str:
    dates = pd.DatetimeIndex(index[index < effective_date])
    source = dates.max() if len(dates) else effective_date
    return pd.Timestamp(source).date().isoformat()


def _extract_yfinance_close_volume(raw: pd.DataFrame, tickers: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    prices: dict[str, pd.Series] = {}
    volumes: dict[str, pd.Series] = {}
    for ticker in tickers:
        sub = None
        if isinstance(raw.columns, pd.MultiIndex):
            if ticker in raw.columns.get_level_values(0):
                sub = raw[ticker]
            elif "Close" in raw.columns.get_level_values(0) and ticker in raw["Close"].columns:
                close = raw["Close"][ticker].dropna().rename(ticker)
                volume = raw.get("Volume", pd.DataFrame(index=raw.index)).get(
                    ticker,
                    pd.Series(index=raw.index, dtype=float),
                )
                prices[ticker] = close
                volumes[ticker] = volume.reindex(close.index).fillna(0.0).rename(ticker)
                continue
        elif "Close" in raw.columns and len(tickers) == 1:
            sub = raw
        if sub is None or "Close" not in sub.columns:
            continue
        close = sub["Close"].dropna().rename(ticker)
        if close.empty:
            continue
        volume = sub.get("Volume", pd.Series(index=sub.index, dtype=float)).reindex(close.index).fillna(0.0).rename(ticker)
        prices[ticker] = close
        volumes[ticker] = volume
    return pd.DataFrame(prices).sort_index(), pd.DataFrame(volumes).sort_index()


def _download_yfinance_panel(tickers: list[str], start_date: str, end_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    price_frames: list[pd.DataFrame] = []
    volume_frames: list[pd.DataFrame] = []
    unique = list(dict.fromkeys(tickers))
    for i in range(0, len(unique), FRESH_BATCH_SIZE):
        batch = unique[i : i + FRESH_BATCH_SIZE]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            raw = yf.download(
                batch,
                start=start_date,
                end=end_date,
                auto_adjust=True,
                progress=False,
                group_by="ticker",
                threads=True,
            )
        prices, volumes = _extract_yfinance_close_volume(raw, batch)
        if not prices.empty:
            price_frames.append(prices)
            volume_frames.append(volumes)
    if not price_frames:
        raise RuntimeError("Fresh yfinance download returned no usable price data.")
    prices = pd.concat(price_frames, axis=1).sort_index()
    volumes = pd.concat(volume_frames, axis=1).sort_index()
    prices = prices.loc[:, ~prices.columns.duplicated()].ffill()
    volumes = volumes.loc[:, ~volumes.columns.duplicated()].fillna(0.0)
    return prices, volumes


def _tickers_overlapping_window(intervals: pd.DataFrame, start_date: str, end_date: str) -> list[str]:
    if intervals.empty:
        return []
    frame = intervals.copy()
    frame["start_date"] = pd.to_datetime(frame["start_date"], errors="coerce")
    frame["end_date"] = pd.to_datetime(frame["end_date"], errors="coerce")
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    mask = frame["start_date"].le(end) & (frame["end_date"].isna() | frame["end_date"].ge(start))
    return frame.loc[mask, "ticker"].dropna().astype(str).drop_duplicates().tolist()


def _fresh_us_th_panel() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str], list[str], pd.DataFrame, pd.Timestamp, str]:
    paths = default_paths(ROOT)
    fresh_start = _fresh_start_date()
    tomorrow = (pd.Timestamp.today().normalize() + pd.Timedelta(days=1)).date().isoformat()
    sp500_intervals = load_sp500_membership_intervals(paths)
    set100_intervals = load_set100_membership_intervals(paths)
    all_us = drop_duplicate_share_classes(_tickers_overlapping_window(sp500_intervals, fresh_start, tomorrow))
    all_th = _tickers_overlapping_window(set100_intervals, fresh_start, tomorrow)
    overlay_prices, overlay_volumes = _download_yfinance_panel(OVERLAY_TICKERS, fresh_start, tomorrow)
    prices_raw, volumes_raw = _download_yfinance_panel(all_us + all_th, fresh_start, tomorrow)
    prices_raw = pd.concat([prices_raw, overlay_prices], axis=1).loc[:, lambda frame: ~frame.columns.duplicated()].sort_index()
    volumes_raw = pd.concat([volumes_raw, overlay_volumes], axis=1).loc[:, lambda frame: ~frame.columns.duplicated()].sort_index()
    missing_overlay = [ticker for ticker in OVERLAY_TICKERS if ticker not in prices_raw.columns]
    if missing_overlay:
        raise RuntimeError(f"Fresh yfinance missing required overlay tickers: {missing_overlay}")
    overlay_common = prices_raw[OVERLAY_TICKERS].dropna()
    if overlay_common.empty:
        raise RuntimeError("Fresh yfinance has no common overlay close.")
    as_of = pd.Timestamp(overlay_common.index.max())
    fx = prices_raw["USDTHB=X"].ffill()
    us_cols = [ticker for ticker in all_us if ticker in prices_raw.columns]
    th_cols = [ticker for ticker in all_th if ticker in prices_raw.columns]
    us_price_df = prices_raw[us_cols].mul(fx, axis=0)
    th_price_df = prices_raw[th_cols]
    thb_prices = pd.concat([us_price_df, th_price_df], axis=1).loc[:as_of].ffill()
    volumes = volumes_raw.reindex(thb_prices.index).reindex(columns=thb_prices.columns).fillna(0.0)
    benchmark = prices_raw["SPY"].mul(fx).reindex(thb_prices.index).ffill().rename("benchmark")
    vol_proxy = prices_raw["^VIX"].reindex(thb_prices.index).ffill().rename("vol_proxy")
    common_index = thb_prices.index.intersection(benchmark.dropna().index).intersection(vol_proxy.dropna().index)
    thb_prices = thb_prices.reindex(common_index).ffill().dropna(how="all")
    volumes = volumes.reindex(thb_prices.index).fillna(0.0)
    benchmark = benchmark.reindex(thb_prices.index).ffill()
    vol_proxy = vol_proxy.reindex(thb_prices.index).ffill()
    overlay = prices_raw[OVERLAY_TICKERS].loc[:as_of].ffill()
    return thb_prices, volumes, benchmark, vol_proxy, us_cols, th_cols, overlay, as_of, fresh_start


def _monthly_returns(curves: pd.DataFrame) -> pd.DataFrame:
    month_end = curves.resample("ME").last().dropna(how="all")
    return month_end.pct_change(fill_method=None).dropna(how="all")


def _rolling_monthly_sharpe(monthly_returns: pd.DataFrame, months: int) -> pd.DataFrame:
    mean = monthly_returns.rolling(months, min_periods=months).mean()
    vol = monthly_returns.rolling(months, min_periods=months).std(ddof=0)
    return (mean / vol.replace(0.0, np.nan)) * np.sqrt(12.0)


def _signal_inputs(pair: pd.DataFrame, us_col: str, th_col: str, lookback: int) -> dict[str, pd.Series]:
    trailing_return = (1.0 + pair).rolling(lookback, min_periods=lookback).apply(np.prod, raw=True) - 1.0
    trailing_sharpe = _rolling_monthly_sharpe(pair, lookback)
    trailing_vol = pair.rolling(lookback, min_periods=lookback).std(ddof=0) * np.sqrt(12.0)
    curves = (1.0 + pair).cumprod()
    ma_window = max(3, min(12, lookback * 2))
    th_curve = curves[th_col]
    th_drawdown = th_curve.div(th_curve.cummax()).sub(1.0)
    return_spread = trailing_return[th_col] - trailing_return[us_col]
    sharpe_spread = trailing_sharpe[th_col] - trailing_sharpe[us_col]
    th_above_ma = th_curve > th_curve.rolling(ma_window, min_periods=ma_window).mean()
    drawdown_improving = th_drawdown > th_drawdown.shift(1)
    score = (
        (return_spread > 0.0).astype(float)
        + (sharpe_spread > 0.0).astype(float)
        + (trailing_return[th_col] > 0.0).astype(float)
        + th_above_ma.astype(float)
        + drawdown_improving.astype(float)
    )
    return {
        "return_spread": return_spread,
        "sharpe_spread": sharpe_spread,
        "th_return": trailing_return[th_col],
        "us_vol": trailing_vol[us_col],
        "th_vol": trailing_vol[th_col],
        "score": score,
    }


def _raw_signal(inputs: dict[str, pd.Series], mode: str, entry: float) -> pd.Series:
    if mode == "relative_return":
        return inputs["return_spread"] > entry
    if mode == "relative_sharpe":
        return inputs["sharpe_spread"] > entry
    if mode == "relative_return_and_sharpe":
        return (inputs["return_spread"] > entry) & (inputs["sharpe_spread"] > entry)
    if mode == "relative_return_or_sharpe":
        return (inputs["return_spread"] > entry) | (inputs["sharpe_spread"] > entry)
    if mode == "relative_return_positive_th":
        return (inputs["return_spread"] > entry) & (inputs["th_return"] > 0.0)
    if mode == "positive_th_return":
        return inputs["th_return"] > entry
    if mode == "score_3_of_5":
        return inputs["score"] >= 3.0
    raise ValueError(f"Unknown signal mode: {mode}")


def _exit_value(inputs: dict[str, pd.Series], mode: str) -> pd.Series:
    if mode in {
        "relative_return",
        "relative_return_and_sharpe",
        "relative_return_or_sharpe",
        "relative_return_positive_th",
    }:
        return inputs["return_spread"]
    if mode == "positive_th_return":
        return inputs["th_return"]
    if mode == "score_3_of_5":
        return inputs["score"] - 3.0
    return inputs["sharpe_spread"]


def _stateful_gate(
    inputs: dict[str, pd.Series],
    mode: str,
    entry: float,
    exit_threshold: float,
    min_hold: int,
    exit_confirm: int,
) -> pd.Series:
    raw_entry = _raw_signal(inputs, mode, entry).fillna(False)
    exit_metric = _exit_value(inputs, mode)
    active = False
    hold = 0
    bad_count = 0
    values: list[float] = []
    for date in raw_entry.index:
        if active:
            hold += 1
            if bool(exit_metric.loc[date] < exit_threshold):
                bad_count += 1
            else:
                bad_count = 0
            if hold >= min_hold and bad_count >= exit_confirm:
                active = False
                hold = 0
                bad_count = 0
        elif bool(raw_entry.loc[date]):
            active = True
            hold = 1
            bad_count = 0
        values.append(1.0 if active else 0.0)
    return pd.Series(values, index=raw_entry.index, name="TH Gate")


def _monthly_weight_signal(
    monthly: pd.DataFrame,
    mode: str,
    allocation_method: str,
    lookback: int,
    th_weight: float,
    entry: float,
    exit_threshold: float,
    min_hold: int,
    exit_confirm: int,
    us_col: str,
    th_col: str,
) -> pd.Series:
    pair = monthly[[us_col, th_col]].dropna()
    inputs = _signal_inputs(pair, us_col, th_col, lookback)
    gate = _stateful_gate(inputs, mode, entry, exit_threshold, min_hold, exit_confirm).shift(1).fillna(0.0)
    if allocation_method != "binary":
        raise ValueError(f"Unsupported allocation method for this strategy: {allocation_method}")
    return (th_weight * gate).rename("TH Weight")


def _daily_weight_from_monthly(monthly_weight: pd.Series, daily_index: pd.DatetimeIndex) -> pd.Series:
    month_key = daily_index.to_period("M").to_timestamp("M")
    mapped = pd.Series(month_key, index=daily_index).map(monthly_weight)
    return mapped.ffill().fillna(0.0).clip(0.0, 1.0)


def _nav_to_returns(nav: pd.Series) -> pd.Series:
    return nav.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _close_trend_exposure(price: pd.Series, ma_period: int, below_exposure: float, initial: float = 1.0) -> pd.Series:
    price = price.astype(float).sort_index().ffill()
    ma = price.rolling(ma_period, min_periods=max(20, int(ma_period * 0.20))).mean()
    signal = pd.Series(1.0, index=price.index, dtype=float)
    signal.loc[price < ma] = below_exposure
    signal.loc[ma.isna()] = initial
    return signal.shift(1).fillna(initial).rename("Daily Exposure")


def _gold_crash_protection_exposure(gold_price: pd.Series) -> pd.Series:
    price = gold_price.astype(float).sort_index().ffill()
    rolling_high = price.rolling(GOLD_DD_WINDOW, min_periods=max(20, GOLD_DD_WINDOW // 4)).max()
    drawdown = price.div(rolling_high).sub(1.0)
    panic_ma = price.rolling(GOLD_PANIC_MA, min_periods=max(20, GOLD_PANIC_MA // 4)).mean()
    panic_mom = price.pct_change(GOLD_PANIC_MOM)
    active = 1.0
    values: list[float] = []
    for date, dd in drawdown.items():
        panic = (
            pd.notna(dd)
            and dd <= GOLD_PANIC_DD
            and pd.notna(panic_ma.loc[date])
            and price.loc[date] < panic_ma.loc[date]
            and pd.notna(panic_mom.loc[date])
            and panic_mom.loc[date] < 0.0
        )
        if pd.isna(dd):
            active = 1.0
        elif panic:
            active = 0.0
        elif dd <= GOLD_CRASH_DD:
            active = GOLD_CRASH_EXPOSURE
        elif dd <= GOLD_WARN_DD:
            active = min(active, GOLD_WARN_EXPOSURE)
        elif dd >= GOLD_RECOVERY_DD:
            active = 1.0
        values.append(active)
    return pd.Series(values, index=drawdown.index, name="Gold Crash Protection Exposure").shift(1).fillna(1.0)


def _month_end(series: pd.Series) -> pd.Series:
    return series.groupby(series.index.to_period("M")).last()


def _th_tactical_weight(overlay: pd.DataFrame, as_of: pd.Timestamp) -> float:
    fx = overlay["USDTHB=X"].ffill()
    us_month = _month_end((overlay["SPY"] * fx).loc[:as_of].dropna())
    th_month = _month_end(overlay["^SET.BK"].loc[:as_of].dropna())
    monthly = pd.concat({"US": us_month, "TH": th_month}, axis=1).dropna()
    if len(monthly) < 2:
        return 0.0
    rel = monthly["TH"].pct_change(1) - monthly["US"].pct_change(1)
    signal = (rel > 0.0).astype(float).shift(1).ffill().fillna(0.0) * 0.30
    return float(signal.iloc[-1])


def _optimize_latest_sleeve(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    benchmark: pd.Series,
    vol_proxy: pd.Series,
    candidates: list[str],
    n_assets: int,
    as_of: pd.Timestamp,
) -> tuple[pd.Series, pd.Timestamp, pd.Timestamp, list[str]]:
    stock_dates = prices.dropna(how="all").index
    stock_dates = stock_dates[stock_dates <= as_of]
    if stock_dates.empty:
        raise RuntimeError("No stock data available on or before the overlay close.")
    stock_as_of = pd.Timestamp(stock_dates.max())
    loc = prices.index.get_loc(stock_as_of)
    train_index = prices.index[max(0, loc - LOOKBACK_DAYS + 1) : loc + 1]
    selected = select_point_in_time_universe(
        prices.reindex(train_index),
        volumes.reindex(train_index),
        candidates,
        n_assets=n_assets,
        min_history_ratio=0.75,
    )
    if not selected:
        raise RuntimeError("Point-in-time universe selection returned no assets.")
    returns = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    train_returns = returns.reindex(train_index)[selected].dropna(axis=1, thresh=max(int(0.75 * len(train_index)), 60))
    selected = drop_duplicate_share_classes_available(train_returns.columns.tolist(), train_returns.columns)
    train_returns = train_returns.reindex(columns=selected)
    features = compute_feature_table(
        train_returns,
        benchmark.pct_change(fill_method=None).reindex(train_index),
        vol_proxy.pct_change(fill_method=None).reindex(train_index),
        prices.reindex(train_index)[selected],
        include_momentum_features=True,
        feature_flags=FEATURE_FLAGS,
    )
    momentum = build_momentum_signal(features, mode="mom_63")
    cov = train_returns.cov().reindex(index=selected, columns=selected).fillna(0.0)
    weights = optimize_portfolio(
        cov,
        momentum,
        max_weight=STOCK_CAP,
        objective_mode="mean_variance",
        asset_caps={asset: STOCK_CAP for asset in selected},
    ).sort_values(ascending=False)
    return weights, stock_as_of, pd.Timestamp(train_index.min()), selected


def _run_joint_pit_reselect_model(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    benchmark: pd.Series,
    vol_proxy: pd.Series,
    us_all: list[str],
    th_all: list[str],
    us_assets: int,
    th_assets: int,
) -> dict[str, object]:
    returns = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    benchmark_ret = benchmark.pct_change(fill_method=None).rename("benchmark")
    vol_proxy_ret = vol_proxy.pct_change(fill_method=None).rename("vol_proxy")
    schedule = monthly_rebalance_dates(prices.index, lookback_days=LOOKBACK_DAYS, freq="ME")

    feature_history: dict[pd.Timestamp, pd.DataFrame] = {}
    market_stress_history: dict[pd.Timestamp, float] = {}

    for rebalance_date in schedule:
        loc = prices.index.get_loc(rebalance_date)
        train_index = prices.index[max(0, loc - LOOKBACK_DAYS + 1) : loc + 1]
        us_pool = [
            ticker
            for ticker in get_sp500_members_as_of(rebalance_date, default_paths(ROOT))
            if ticker in us_all and ticker in prices.columns
        ]
        us_pool = drop_duplicate_share_classes(us_pool)
        th_pool = [
            ticker
            for ticker in get_set100_members_as_of(rebalance_date, default_paths(ROOT))
            if ticker in th_all and ticker in prices.columns
        ]
        us_selected = select_point_in_time_universe(prices.loc[train_index], volumes.loc[train_index], us_pool, n_assets=us_assets)
        th_selected = select_point_in_time_universe(prices.loc[train_index], volumes.loc[train_index], th_pool, n_assets=th_assets)
        current_assets = list(dict.fromkeys(us_selected + th_selected))
        if not current_assets:
            continue
        train_returns = returns.reindex(train_index)[current_assets].dropna(axis=1, thresh=max(int(0.85 * len(train_index)), 60))
        if train_returns.shape[1] < max(N_CLUSTERS + 2, 6):
            continue
        current_assets = train_returns.columns.tolist()
        market_stress_history[rebalance_date] = compute_market_stress_signal(
            benchmark_ret.reindex(train_index),
            vol_proxy_ret.reindex(train_index),
        )
        feature_table = compute_feature_table(
            train_returns,
            benchmark_ret.reindex(train_index),
            vol_proxy_ret.reindex(train_index),
            prices.reindex(train_index)[current_assets],
            include_momentum_features=True,
            feature_flags=FEATURE_FLAGS,
        )
        if not feature_table.empty:
            feature_history[rebalance_date] = feature_table

    if not feature_history:
        raise RuntimeError("No feature history was available for the US/TH PIT reselect model.")

    first_date = min(feature_history)
    initial = initialize_static_clusters(feature_history[first_date], n_clusters=N_CLUSTERS)
    dynamic_state = run_dynamic_hmm(
        feature_history,
        initial_state=initial,
        gas_alpha=0.40,
        gas_beta=0.45,
        market_stress_history=market_stress_history,
        posterior_power=2.25,
    )
    static_post = pd.get_dummies(initial["labels"]).reindex(columns=range(N_CLUSTERS), fill_value=0.0)

    strategy_names = ["Equal Weight", "Risk Parity", "Static Copula", "Dynamic HMM Copula"]
    nav = {name: pd.Series(1.0, index=[schedule[0]]) for name in strategy_names}
    weights_history: dict[str, dict[pd.Timestamp, pd.Series]] = {name: {} for name in strategy_names}

    for idx, rebalance_date in enumerate(schedule[:-1]):
        next_date = schedule[idx + 1]
        if rebalance_date not in feature_history:
            continue
        loc = prices.index.get_loc(rebalance_date)
        train_index = prices.index[max(0, loc - LOOKBACK_DAYS + 1) : loc + 1]
        test_index = prices.index[(prices.index > rebalance_date) & (prices.index <= next_date)]
        if len(test_index) == 0:
            continue

        current_features = feature_history[rebalance_date]
        current_assets = current_features.index.tolist()
        train_returns = returns.loc[train_index, current_assets].dropna(how="all")
        bench_train = benchmark_ret.loc[train_index]
        momentum_signal = build_momentum_signal(current_features, mode="mom_63")

        eq_weights = pd.Series(1.0 / len(current_assets), index=current_assets)
        risk_parity_cov = train_returns.cov().reindex(index=current_assets, columns=current_assets).fillna(0.0)
        static_cov, _ = build_factor_covariance(
            train_returns,
            bench_train,
            static_post.reindex(current_assets).fillna(0.0),
            current_features,
            dynamic=False,
        )
        dyn_cov, _ = build_factor_covariance(
            train_returns,
            bench_train,
            dynamic_state["posterior_history"][rebalance_date].reindex(current_assets).fillna(0.0),
            current_features,
            dynamic=True,
            centroid_snapshot=dynamic_state["centroid_history"][rebalance_date],
        )
        active_caps = {asset: STOCK_CAP for asset in current_assets}
        weights = {
            "Equal Weight": eq_weights,
            "Risk Parity": optimize_risk_parity(risk_parity_cov, max_weight=STOCK_CAP),
            "Static Copula": optimize_portfolio(
                static_cov,
                momentum_signal,
                max_weight=STOCK_CAP,
                objective_mode="mean_variance",
                asset_caps=active_caps,
            ),
            "Dynamic HMM Copula": optimize_portfolio(
                dyn_cov,
                momentum_signal,
                max_weight=STOCK_CAP,
                objective_mode="mean_variance",
                asset_caps=active_caps,
            ),
        }
        period_returns = returns.reindex(test_index)[current_assets].fillna(0.0)
        for strategy, strategy_weights in weights.items():
            weights_history[strategy][rebalance_date] = strategy_weights
            weighted = period_returns.mul(strategy_weights, axis=1).sum(axis=1)
            starting_value = float(nav[strategy].iloc[-1])
            nav[strategy] = pd.concat([nav[strategy], starting_value * (1.0 + weighted).cumprod()])

    nav = {name: series[~series.index.duplicated(keep="last")].sort_index() for name, series in nav.items()}
    return {"nav": nav, "weights_history": weights_history}


def _latest_weights(history: dict[pd.Timestamp, pd.Series], sleeve: str, multiplier: float, date: pd.Timestamp) -> pd.DataFrame:
    latest_date = max(history)
    latest = history[latest_date].rename("Internal Weight").reset_index()
    latest.columns = ["Asset", "Internal Weight"]
    latest["Sleeve"] = sleeve
    latest["Sleeve Multiplier"] = multiplier
    latest["Effective Weight"] = latest["Internal Weight"].mul(multiplier)
    latest["Internal Weight Date"] = pd.Timestamp(latest_date).date().isoformat()
    latest["Date"] = date.date().isoformat()
    return latest


def _write_outputs(security: pd.DataFrame, sleeve: pd.DataFrame, meta: pd.DataFrame) -> None:
    for output_dir in [ROOT / "result", ROOT / "data" / "precomputed"]:
        output_dir.mkdir(parents=True, exist_ok=True)
        security.to_csv(output_dir / f"{RESULT_PREFIX}_latest_effective_security_weights_thb.csv", index=False)
        sleeve.to_csv(output_dir / f"{RESULT_PREFIX}_latest_effective_sleeve_weights_thb.csv", index=False)
        meta.to_csv(output_dir / f"{RESULT_PREFIX}_latest_meta.csv", index=False)
        payload = meta.iloc[0].to_dict() if not meta.empty else {}
        payload["calculated_at"] = pd.Timestamp.now(tz="Asia/Bangkok").isoformat()
        (output_dir / f"{RESULT_PREFIX}_latest_meta.json").write_text(
            json.dumps(payload, indent=2, default=str),
            encoding="utf-8",
        )


def main() -> None:
    paths = default_paths(ROOT)
    paths.result_dir.mkdir(parents=True, exist_ok=True)
    prices, volumes, benchmark, vol_proxy, us_all, th_all, overlay_raw, as_of, fresh_start = _fresh_us_th_panel()
    end_date = as_of.date().isoformat()

    us_results = _run_joint_pit_reselect_model(
        prices=prices,
        volumes=volumes,
        benchmark=benchmark,
        vol_proxy=vol_proxy,
        us_all=us_all,
        th_all=[],
        us_assets=US_ASSETS,
        th_assets=0,
    )
    set_benchmark = overlay_raw["^SET.BK"].reindex(prices.index).ffill().rename("benchmark")
    th_results = _run_joint_pit_reselect_model(
        prices=prices,
        volumes=volumes,
        benchmark=set_benchmark,
        vol_proxy=vol_proxy,
        us_all=[],
        th_all=th_all,
        us_assets=0,
        th_assets=TH_ASSETS,
    )

    us_nav = us_results["nav"][PRIMARY_MODEL].dropna()
    th_nav = th_results["nav"][PRIMARY_MODEL].dropna()
    common_index = us_nav.index.union(th_nav.index).sort_values()
    daily_returns = pd.DataFrame(
        {
            "US PIT optimized sleeve THB": _nav_to_returns(us_nav).reindex(common_index).fillna(0.0),
            "TH PIT optimized sleeve THB": _nav_to_returns(th_nav).reindex(common_index).fillna(0.0),
        }
    ).loc[:end_date].fillna(0.0)
    fx = overlay_raw["USDTHB=X"].reindex(daily_returns.index).ffill()
    overlay_prices = pd.DataFrame(
        {
            "S&P 500 ETF THB": overlay_raw["SPY"].reindex(daily_returns.index).ffill().mul(fx),
            "SET Index THB proxy": overlay_raw["^SET.BK"].reindex(daily_returns.index).ffill(),
            "Gold": overlay_raw["GC=F"].reindex(daily_returns.index).ffill().mul(fx),
            "BTC": overlay_raw["BTC-USD"].reindex(daily_returns.index).ffill().mul(fx),
        },
        index=daily_returns.index,
    ).ffill()
    signal_prices = pd.DataFrame(
        {
            "US Equity": overlay_raw["SPY"].reindex(daily_returns.index).ffill(),
            "TH Equity": overlay_raw["^SET.BK"].reindex(daily_returns.index).ffill(),
            "Gold": overlay_raw["GC=F"].reindex(daily_returns.index).ffill(),
            "BTC": overlay_raw["BTC-USD"].reindex(daily_returns.index).ffill(),
        },
        index=daily_returns.index,
    ).ffill()

    sleeve_curves = (1.0 + daily_returns).cumprod().mul(10_000.0)
    monthly = _monthly_returns(
        pd.concat(
            [
                sleeve_curves,
                overlay_prices[["S&P 500 ETF THB", "SET Index THB proxy"]]
                .div(overlay_prices[["S&P 500 ETF THB", "SET Index THB proxy"]].iloc[0])
                .mul(10_000.0),
            ],
            axis=1,
        )
    )
    th_monthly_weight = _monthly_weight_signal(
        monthly,
        mode="relative_return",
        allocation_method="binary",
        lookback=1,
        th_weight=0.30,
        entry=0.0,
        exit_threshold=0.0,
        min_hold=0,
        exit_confirm=1,
        us_col="S&P 500 ETF THB",
        th_col="SET Index THB proxy",
    )
    th_daily_weight = _daily_weight_from_monthly(th_monthly_weight, daily_returns.index)
    th_tactical_weight = float(th_daily_weight.loc[:as_of].iloc[-1])

    exposure = pd.DataFrame(
        {
            "US Equity": _close_trend_exposure(signal_prices["US Equity"], 300, 0.50),
            "TH Equity": _close_trend_exposure(signal_prices["TH Equity"], 200, 0.00),
            "Gold": _gold_crash_protection_exposure(signal_prices["Gold"]),
            "BTC": _close_trend_exposure(signal_prices["BTC"], 50, 0.00),
        }
    ).reindex(daily_returns.index).ffill().fillna(1.0).clip(0.0, 1.0)
    latest_exposure = exposure.loc[:as_of].iloc[-1]
    output_date = as_of

    raw_sleeve = pd.Series(
        {
            "US Equity": SELECTED_MIX["Equity"] * (1.0 - th_tactical_weight),
            "TH Equity": SELECTED_MIX["Equity"] * th_tactical_weight,
            "Gold": SELECTED_MIX["Gold"],
            "BTC": SELECTED_MIX["BTC"],
        },
        dtype=float,
    )
    effective_sleeve = raw_sleeve.mul(latest_exposure)
    effective_sleeve["Cash / Reduced Exposure"] = max(0.0, 1.0 - float(effective_sleeve.sum()))

    security = pd.concat(
        [
            _latest_weights(us_results["weights_history"][PRIMARY_MODEL], "US Equity", float(effective_sleeve["US Equity"]), output_date),
            _latest_weights(th_results["weights_history"][PRIMARY_MODEL], "TH Equity", float(effective_sleeve["TH Equity"]), output_date),
            pd.DataFrame(
                [
                    {
                        "Asset": "GC=F",
                        "Internal Weight": 1.0,
                        "Sleeve": "Gold",
                        "Sleeve Multiplier": float(effective_sleeve["Gold"]),
                        "Effective Weight": float(effective_sleeve["Gold"]),
                        "Internal Weight Date": output_date.date().isoformat(),
                        "Date": output_date.date().isoformat(),
                    },
                    {
                        "Asset": "BTC-USD",
                        "Internal Weight": 1.0,
                        "Sleeve": "BTC",
                        "Sleeve Multiplier": float(effective_sleeve["BTC"]),
                        "Effective Weight": float(effective_sleeve["BTC"]),
                        "Internal Weight Date": output_date.date().isoformat(),
                        "Date": output_date.date().isoformat(),
                    },
                    {
                        "Asset": "Cash / Reduced Exposure",
                        "Internal Weight": 1.0,
                        "Sleeve": "Cash / Reduced Exposure",
                        "Sleeve Multiplier": float(effective_sleeve["Cash / Reduced Exposure"]),
                        "Effective Weight": float(effective_sleeve["Cash / Reduced Exposure"]),
                        "Internal Weight Date": output_date.date().isoformat(),
                        "Date": output_date.date().isoformat(),
                    },
                ]
            ),
        ],
        ignore_index=True,
    )
    security["Strategy"] = STRATEGY
    security["Raw Sleeve Weight"] = security["Sleeve"].map(raw_sleeve).fillna(security["Sleeve Multiplier"])
    security["Daily Exposure"] = security["Sleeve"].map(latest_exposure).fillna(1.0)
    security["Effective Weight %"] = security["Effective Weight"].mul(100.0)
    us_internal_date = pd.Timestamp(max(us_results["weights_history"][PRIMARY_MODEL])).date().isoformat()
    th_internal_date = pd.Timestamp(max(th_results["weights_history"][PRIMARY_MODEL])).date().isoformat()
    security = security.loc[
        security["Effective Weight"].abs().gt(1e-12)
        | security["Sleeve"].isin({"Gold", "BTC", "Cash / Reduced Exposure"})
    ].sort_values("Effective Weight", ascending=False)

    sleeve = effective_sleeve.rename("Effective Weight").reset_index().rename(columns={"index": "Sleeve"})
    sleeve["Raw Sleeve Weight"] = sleeve["Sleeve"].map(raw_sleeve).fillna(0.0)
    sleeve["Daily Exposure"] = sleeve["Sleeve"].map(latest_exposure).fillna(1.0)
    sleeve["Date"] = output_date.date().isoformat()
    sleeve["Strategy"] = STRATEGY
    sleeve["Effective Weight %"] = sleeve["Effective Weight"].mul(100.0)

    meta = pd.DataFrame(
        [
            {
                "Date": output_date.date().isoformat(),
                "Strategy": STRATEGY,
                "Tactical Rule": "proxy_regime relative_return binary lb1 cap30 entry0 exit0 hold0 confirm1",
                "Overlay Mix": "Equity/Gold/BTC 65/25/10",
                "Daily Exposure": (
                    "US SPY MA300 below50%; TH SET MA200 below0%; "
                    "Gold DD252 warn-8%->50%, crash-20%->50%, panic-30% + below MA200 + mom63<0 -> 0%, recover-5%; "
                    "BTC MA50 below0%"
                ),
                "TH Tactical Weight Inside Equity Sleeve": th_tactical_weight,
                "US Sleeve Internal Weight Date": us_internal_date,
                "TH Sleeve Internal Weight Date": th_internal_date,
                "Risk Free Rate": RISK_FREE_RATE,
                "Fresh Start Date": fresh_start,
                "Fresh Lookback Calendar Days": FRESH_LOOKBACK_CALENDAR_DAYS,
                "BTC Price Source": "yfinance fresh BTC-USD",
                "BTC Price": float(signal_prices["BTC"].loc[:as_of].iloc[-1]),
                "BTC MA50": float(signal_prices["BTC"].rolling(50, min_periods=20).mean().loc[:as_of].iloc[-1]),
                "BTC Daily Exposure": float(latest_exposure["BTC"]),
                "Gold Price": float(signal_prices["Gold"].loc[:as_of].iloc[-1]),
                "Gold DD252": float(
                    signal_prices["Gold"].loc[:as_of].iloc[-1]
                    / signal_prices["Gold"].rolling(GOLD_DD_WINDOW, min_periods=max(20, GOLD_DD_WINDOW // 4)).max().loc[:as_of].iloc[-1]
                    - 1.0
                ),
                "Gold Daily Exposure": float(latest_exposure["Gold"]),
                "Timing Note": (
                    "Fresh yfinance download at run time; no price cache used for US stocks, TH stocks, Gold, BTC, FX, SPY, VIX, or SET. "
                    "Month-end tactical signal and close trend exposure are lagged before use."
                ),
            }
        ]
    )
    _write_outputs(security, sleeve, meta)
    print(meta.to_string(index=False))
    print(sleeve.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(
        security[
            ["Asset", "Sleeve", "Effective Weight", "Internal Weight", "Raw Sleeve Weight", "Daily Exposure", "Sleeve Multiplier"]
        ].head(40).to_string(index=False, float_format=lambda value: f"{value:.6f}")
    )


if __name__ == "__main__":
    main()
