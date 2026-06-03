from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in [SRC, SCRIPTS]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dynamic_factor_copula import (  # noqa: E402
    build_momentum_signal,
    compute_feature_table,
    default_paths,
    get_set100_members_as_of,
    get_sp500_members_as_of,
    load_cached_market_data,
    load_set100_membership_intervals,
    load_sp500_membership_intervals,
    optimize_portfolio,
    select_point_in_time_universe,
)
from share_class_utils import drop_duplicate_share_classes_available  # noqa: E402


START_DATE = "2016-01-01"
LOOKBACK_DAYS = 504
US_ASSETS = 30
TH_ASSETS = 30
STOCK_CAP = 0.08
SELECTED_MIX = {"Equity": 0.65, "Gold": 0.25, "BTC": 0.10}
STRATEGY = "Final Best Sharpe Tactical TH/Gold/BTC 65/25/10 Gold crash protection"
RESULT_PREFIX = "us_th_tactical_perf_momentum_final_best"
FEATURE_FLAGS = {"resid_vol": False, "drawdown": False, "downside_beta": False}
GOLD_DD_WINDOW = 252
GOLD_WARN_DD = -0.08
GOLD_WARN_EXPOSURE = 0.50
GOLD_CRASH_DD = -0.20
GOLD_CRASH_EXPOSURE = 0.50
GOLD_RECOVERY_DD = -0.05
GOLD_PANIC_DD = -0.30
GOLD_PANIC_MA = 200
GOLD_PANIC_MOM = 63


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


def _close_trend_exposure(price: pd.Series, ma_period: int, below_exposure: float) -> pd.Series:
    price = price.astype(float).sort_index().ffill()
    ma = price.rolling(ma_period, min_periods=max(20, int(ma_period * 0.20))).mean()
    signal = pd.Series(1.0, index=price.index, dtype=float)
    signal.loc[price < ma] = below_exposure
    signal.loc[ma.isna()] = 1.0
    return signal.shift(1).ffill().fillna(1.0).clip(0.0, 1.0)


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
    return pd.Series(values, index=drawdown.index, name="Gold Daily Exposure").shift(1).ffill().fillna(1.0)


def _source_close_date(index: pd.Index, effective_date: pd.Timestamp) -> str:
    dates = pd.DatetimeIndex(index[index < effective_date])
    source = dates.max() if len(dates) else effective_date
    return pd.Timestamp(source).date().isoformat()


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


def _security_rows(weights: pd.Series, sleeve: str, multiplier: float, as_of: pd.Timestamp, internal_date: pd.Timestamp) -> pd.DataFrame:
    rows = weights.rename("Internal Weight").reset_index().rename(columns={"index": "Asset"})
    rows["Sleeve"] = sleeve
    rows["Sleeve Multiplier"] = multiplier
    rows["Effective Weight"] = rows["Internal Weight"].mul(multiplier)
    rows["Internal Weight Date"] = pd.Timestamp(internal_date).date().isoformat()
    rows["Date"] = as_of.date().isoformat()
    return rows


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
    overlay = _load_overlay()
    as_of = _latest_common_close(overlay)

    us_all = _all_available_members("us")
    th_all = _all_available_members("th")
    us_active = _active_members("us", as_of, us_all)
    th_active = _active_members("th", as_of, th_all)
    all_tickers = list(dict.fromkeys(us_active + th_active))
    cached = load_cached_market_data(paths, tickers=all_tickers)
    fx = overlay["USDTHB=X"].reindex(cached["prices"].index).ffill()

    us_prices = cached["prices"].reindex(columns=us_active).mul(fx, axis=0).loc[START_DATE:as_of].ffill()
    th_prices = cached["prices"].reindex(columns=th_active).loc[START_DATE:as_of].ffill()
    us_volumes = cached["volumes"].reindex(us_prices.index).reindex(columns=us_active).fillna(0.0)
    th_volumes = cached["volumes"].reindex(th_prices.index).reindex(columns=th_active).fillna(0.0)
    benchmark_us = (overlay["SPY"] * overlay["USDTHB=X"]).reindex(us_prices.index).ffill().rename("benchmark")
    benchmark_th = overlay["^SET.BK"].reindex(th_prices.index).ffill().rename("benchmark")
    vol_proxy_us = overlay["^VIX"].reindex(us_prices.index).ffill().rename("vol_proxy")
    vol_proxy_th = pd.Series(0.0, index=th_prices.index, name="vol_proxy")

    us_weights, us_internal_date, us_train_start, us_selected = _optimize_latest_sleeve(
        us_prices,
        us_volumes,
        benchmark_us,
        vol_proxy_us,
        us_active,
        US_ASSETS,
        as_of,
    )
    th_weights, th_internal_date, th_train_start, th_selected = _optimize_latest_sleeve(
        th_prices,
        th_volumes,
        benchmark_th,
        vol_proxy_th,
        th_active,
        TH_ASSETS,
        as_of,
    )

    th_inside_equity = _th_tactical_weight(overlay, as_of)
    raw_sleeve = pd.Series(
        {
            "US Equity": SELECTED_MIX["Equity"] * (1.0 - th_inside_equity),
            "TH Equity": SELECTED_MIX["Equity"] * th_inside_equity,
            "Gold": SELECTED_MIX["Gold"],
            "BTC": SELECTED_MIX["BTC"],
        },
        dtype=float,
    )
    exposures = pd.Series(
        {
            "US Equity": float(_close_trend_exposure(overlay["SPY"], 300, 0.50).loc[:as_of].iloc[-1]),
            "TH Equity": float(_close_trend_exposure(overlay["^SET.BK"], 200, 0.00).loc[:as_of].iloc[-1]),
            "Gold": float(_gold_crash_protection_exposure(overlay["GC=F"]).loc[:as_of].iloc[-1]),
            "BTC": float(_close_trend_exposure(overlay["BTC-USD"], 50, 0.00).loc[:as_of].iloc[-1]),
        },
        dtype=float,
    )
    effective_sleeve = raw_sleeve.mul(exposures)
    effective_sleeve["Cash / Reduced Exposure"] = max(0.0, 1.0 - float(effective_sleeve.sum()))

    security = pd.concat(
        [
            _security_rows(us_weights, "US Equity", float(effective_sleeve["US Equity"]), as_of, us_internal_date),
            _security_rows(th_weights, "TH Equity", float(effective_sleeve["TH Equity"]), as_of, th_internal_date),
            pd.DataFrame(
                [
                    {
                        "Asset": "GC=F",
                        "Internal Weight": 1.0,
                        "Sleeve": "Gold",
                        "Sleeve Multiplier": float(effective_sleeve["Gold"]),
                        "Effective Weight": float(effective_sleeve["Gold"]),
                        "Internal Weight Date": as_of.date().isoformat(),
                        "Date": as_of.date().isoformat(),
                    },
                    {
                        "Asset": "BTC-USD",
                        "Internal Weight": 1.0,
                        "Sleeve": "BTC",
                        "Sleeve Multiplier": float(effective_sleeve["BTC"]),
                        "Effective Weight": float(effective_sleeve["BTC"]),
                        "Internal Weight Date": as_of.date().isoformat(),
                        "Date": as_of.date().isoformat(),
                    },
                    {
                        "Asset": "Cash / Reduced Exposure",
                        "Internal Weight": 1.0,
                        "Sleeve": "Cash / Reduced Exposure",
                        "Sleeve Multiplier": float(effective_sleeve["Cash / Reduced Exposure"]),
                        "Effective Weight": float(effective_sleeve["Cash / Reduced Exposure"]),
                        "Internal Weight Date": as_of.date().isoformat(),
                        "Date": as_of.date().isoformat(),
                    },
                ]
            ),
        ],
        ignore_index=True,
    )
    security["Strategy"] = STRATEGY
    security["Raw Sleeve Weight"] = security["Sleeve"].map(raw_sleeve).fillna(security["Sleeve Multiplier"])
    security["Daily Exposure"] = security["Sleeve"].map(exposures).fillna(1.0)
    security["Last Exposure Date"] = as_of.date().isoformat()
    security["Signal Source Close Date"] = _source_close_date(overlay.index, as_of)
    security["Effective Weight %"] = security["Effective Weight"].mul(100.0)
    security = security.loc[
        security["Effective Weight"].abs().gt(1e-12)
        | security["Sleeve"].isin({"Gold", "BTC", "Cash / Reduced Exposure"})
    ].sort_values("Effective Weight", ascending=False)

    sleeve = effective_sleeve.rename("Effective Weight").reset_index().rename(columns={"index": "Sleeve"})
    sleeve["Raw Sleeve Weight"] = sleeve["Sleeve"].map(raw_sleeve).fillna(0.0)
    sleeve["Daily Exposure"] = sleeve["Sleeve"].map(exposures).fillna(1.0)
    sleeve["Date"] = as_of.date().isoformat()
    sleeve["Strategy"] = STRATEGY
    sleeve["Effective Weight %"] = sleeve["Effective Weight"].mul(100.0)

    gold_high = overlay["GC=F"].rolling(GOLD_DD_WINDOW, min_periods=max(20, GOLD_DD_WINDOW // 4)).max()
    gold_dd = overlay["GC=F"].loc[as_of] / gold_high.loc[as_of] - 1.0
    btc_ma50 = overlay["BTC-USD"].rolling(50, min_periods=20).mean().loc[as_of]
    meta = pd.DataFrame(
        [
            {
                "Date": as_of.date().isoformat(),
                "Strategy": STRATEGY,
                "Tactical Rule": "proxy_regime relative_return binary lb1 cap30 entry0 exit0 hold0 confirm1",
                "Overlay Mix": "Equity/Gold/BTC 65/25/10",
                "Daily Exposure": (
                    "US SPY MA300 below50%; TH SET MA200 below0%; "
                    "Gold DD252 warn-8%->50%, crash-20%->50%, panic-30% + below MA200 + mom63<0 -> 0%, recover-5%; "
                    "BTC MA50 below0%"
                ),
                "US Sleeve Model": "PIT S&P 500 top30 liquidity / mean covariance / mean_variance + mom_63 / stock cap 8%",
                "TH Sleeve Model": "PIT SET100 top30 liquidity / mean covariance / mean_variance + mom_63 / stock cap 8%",
                "TH Tactical Weight Inside Equity Sleeve": th_inside_equity,
                "Selected US Assets": len(us_selected),
                "Selected TH Assets": len(th_selected),
                "US Sleeve Internal Weight Date": us_internal_date.date().isoformat(),
                "TH Sleeve Internal Weight Date": th_internal_date.date().isoformat(),
                "US Train Start": us_train_start.date().isoformat(),
                "TH Train Start": th_train_start.date().isoformat(),
                "Train End": as_of.date().isoformat(),
                "BTC Price": float(overlay["BTC-USD"].loc[as_of]),
                "BTC MA50": float(btc_ma50),
                "BTC Daily Exposure": float(exposures["BTC"]),
                "Gold Price": float(overlay["GC=F"].loc[as_of]),
                "Gold DD252": float(gold_dd),
                "Gold Daily Exposure": float(exposures["Gold"]),
                "Timing Note": "Standalone latest-weight refresh from this repo's refreshed cache; no static latest weights are read from dynamic_port_opt.",
            }
        ]
    )
    _write_outputs(security, sleeve, meta)
    print(meta.to_string(index=False))
    print(sleeve.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(security[["Asset", "Sleeve", "Effective Weight", "Internal Weight", "Raw Sleeve Weight", "Daily Exposure"]].head(40).to_string(index=False, float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()
