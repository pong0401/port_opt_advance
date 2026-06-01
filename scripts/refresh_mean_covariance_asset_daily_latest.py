from __future__ import annotations

from pathlib import Path
import json
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yfinance as yf


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dynamic_factor_copula import (  # noqa: E402
    build_momentum_signal,
    compute_feature_table,
    default_paths,
    get_sp500_members_as_of,
    load_cached_market_data,
    load_overlay_compare_prices,
    load_sp500_membership_intervals,
    optimize_portfolio,
    select_point_in_time_universe,
)
from share_class_utils import (  # noqa: E402
    assert_no_duplicate_share_classes,
    drop_duplicate_share_classes,
    drop_duplicate_share_classes_available,
)


START_DATE = "2016-01-01"
LOOKBACK_DAYS = 504
US_ASSETS = 30
STOCK_CAP = 0.08
GOLD_CAP = 0.30
BTC_CAP = 0.05
BIL_CAP = 0.00
OVERLAY_ASSETS = ["GC=F", "BTC-USD", "BIL"]
FEATURE_FLAGS = {"resid_vol": False, "drawdown": False, "downside_beta": False}
STRATEGY = "Mean Covariance Gold30 stock cap 8 mom_63 + asset-level daily exposure"


def _lag_close_signal_to_next_session(signal: pd.Series, initial: float = 1.0) -> pd.Series:
    return signal.astype(float).sort_index().shift(1).ffill().fillna(initial)


def _best_signal_config() -> pd.DataFrame:
    config_path = default_paths(ROOT).result_dir / "best_param_step3b_best_signal_config_used.csv"
    if config_path.exists():
        return pd.read_csv(config_path, index_col=0)
    return pd.DataFrame(
        {
            "Asset": {"SPY": "S&P 500", "GOLD": "Gold", "BTC": "BTC"},
            "MA Period": {"SPY": 300, "GOLD": 50, "BTC": 50},
            "Below Exposure": {"SPY": 0.50, "GOLD": 1.00, "BTC": 0.00},
        }
    )


def _close_trend_exposure(price: pd.Series, ma_period: int, below_exposure: float) -> pd.Series:
    price = price.astype(float).sort_index().ffill()
    min_periods = max(20, int(ma_period * 0.20))
    ma = price.rolling(ma_period, min_periods=min_periods).mean()
    signal = pd.Series(1.0, index=price.index, dtype=float)
    signal.loc[price < ma] = below_exposure
    signal.loc[ma.isna()] = 1.0
    return _lag_close_signal_to_next_session(signal, initial=1.0)


def _latest_common_close(overlay: pd.DataFrame) -> pd.Timestamp:
    required = ["SPY", "^VIX", "GC=F", "BIL"]
    common = overlay.dropna(subset=required)
    if common.empty:
        raise ValueError("No common latest close found for SPY, ^VIX, Gold, and BIL.")
    return pd.Timestamp(common.index.max())


def _load_overlay_prices(paths) -> pd.DataFrame:
    overlay_path = ROOT / "data" / "cache" / "dynamic_factor_copula" / "overlay_compare_prices.parquet"
    if overlay_path.exists():
        overlay = pd.read_parquet(overlay_path).sort_index()
        needed = ["SPY", "^VIX", *OVERLAY_ASSETS]
        if all(column in overlay.columns for column in needed):
            return overlay.loc[START_DATE:, needed].sort_index().ffill()
    return load_overlay_compare_prices(
        paths,
        start_date=START_DATE,
        tickers=["SPY", "^VIX", *OVERLAY_ASSETS],
    ).sort_index()


def _cached_price_columns(paths) -> set[str]:
    names = pq.ParquetFile(paths.source_cache_root / "prices.parquet").schema.names
    return {name for name in names if not name.startswith("__")}


def _read_source_market_cache(paths, tickers: list[str], start: str, end: pd.Timestamp) -> dict[str, pd.DataFrame]:
    columns = [ticker for ticker in dict.fromkeys(tickers) if ticker in _cached_price_columns(paths)]
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    frames: dict[str, list[pd.Series]] = {"prices": [], "volumes": []}
    extra_prices_path = paths.local_cache_root / "extra_prices.parquet"
    extra_volumes_path = paths.local_cache_root / "extra_volumes.parquet"
    extra_price_cols = set(pq.ParquetFile(extra_prices_path).schema.names) if extra_prices_path.exists() else set()
    extra_volume_cols = set(pq.ParquetFile(extra_volumes_path).schema.names) if extra_volumes_path.exists() else set()
    for column in columns:
        price = pd.read_parquet(paths.source_cache_root / "prices.parquet", columns=[column])[column]
        volume = pd.read_parquet(paths.source_cache_root / "volumes.parquet", columns=[column])[column]
        if column in extra_price_cols:
            extra_price = pd.read_parquet(extra_prices_path, columns=[column])[column]
            price = price.combine_first(extra_price)
        if column in extra_volume_cols:
            extra_volume = pd.read_parquet(extra_volumes_path, columns=[column])[column]
            volume = volume.combine_first(extra_volume)
        frames["prices"].append(price.loc[start_ts:end_ts].rename(column))
        frames["volumes"].append(volume.loc[start_ts:end_ts].rename(column))
    return {
        "prices": pd.concat(frames["prices"], axis=1).sort_index() if frames["prices"] else pd.DataFrame(),
        "volumes": pd.concat(frames["volumes"], axis=1).sort_index() if frames["volumes"] else pd.DataFrame(),
    }


def _merge_parquet(path: Path, update: pd.DataFrame) -> None:
    if update.empty:
        return
    existing = pd.read_parquet(path).sort_index() if path.exists() else pd.DataFrame()
    combined = update.combine_first(existing).sort_index()
    combined.to_parquet(path)


def _download_latest(tickers: list[str], start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    price_frames = []
    volume_frames = []
    for ticker in tickers:
        raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False, threads=False)
        if raw.empty:
            continue
        if isinstance(raw.columns, pd.MultiIndex):
            raw = raw.droplevel(1, axis=1)
        if "Close" not in raw.columns:
            continue
        close = raw["Close"].dropna().rename(ticker)
        if close.empty:
            continue
        volume = raw.get("Volume", pd.Series(index=close.index, dtype=float)).reindex(close.index).fillna(0.0).rename(ticker)
        price_frames.append(close)
        volume_frames.append(volume)
    prices = pd.concat(price_frames, axis=1).sort_index() if price_frames else pd.DataFrame()
    volumes = pd.concat(volume_frames, axis=1).sort_index() if volume_frames else pd.DataFrame()
    return prices, volumes


def _refresh_latest_stock_cache(paths, tickers: list[str], as_of: pd.Timestamp) -> None:
    if os.environ.get("PORT_OPT_REFRESH_YFINANCE") != "1":
        return
    paths.local_cache_root.mkdir(parents=True, exist_ok=True)
    yf_cache_dir = paths.local_cache_root / ".yfinance"
    yf_cache_dir.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(yf_cache_dir))
    start = (pd.Timestamp(as_of).date() - pd.Timedelta(days=70)).isoformat()
    end = (pd.Timestamp(as_of).date() + pd.Timedelta(days=1)).isoformat()
    prices, volumes = _download_latest(tickers, start, end)
    _merge_parquet(paths.local_cache_root / "extra_prices.parquet", prices)
    _merge_parquet(paths.local_cache_root / "extra_volumes.parquet", volumes)


def _source_close_date(index: pd.Index, effective_date: pd.Timestamp) -> str:
    source_candidates = pd.DatetimeIndex(index[index < effective_date])
    source_date = source_candidates.max() if len(source_candidates) else effective_date
    return pd.Timestamp(source_date).date().isoformat()


def _write_outputs(latest: pd.DataFrame, sleeve_latest: pd.DataFrame, meta: pd.DataFrame) -> None:
    outputs = [
        ROOT / "result",
        ROOT / "data" / "precomputed",
    ]
    for output_dir in outputs:
        output_dir.mkdir(parents=True, exist_ok=True)
        latest.to_csv(output_dir / "mean_covariance_gold30_asset_daily_latest_effective_weights.csv", index=False)
        latest.to_csv(output_dir / "mean_covariance_gold30_asset_daily_recheck_today_weights.csv", index=False)
        sleeve_latest.to_csv(output_dir / "mean_covariance_gold30_asset_daily_recheck_today_sleeve_weights.csv", index=False)
        meta.to_csv(output_dir / "mean_covariance_gold30_asset_daily_recheck_today_meta.csv", index=False)
        payload = meta.iloc[0].to_dict() if not meta.empty else {}
        payload["calculated_at"] = pd.Timestamp.now(tz="Asia/Bangkok").isoformat()
        (output_dir / "mean_covariance_gold30_asset_daily_recheck_today_meta.json").write_text(
            json.dumps(payload, indent=2, default=str),
            encoding="utf-8",
        )


def main() -> None:
    paths = default_paths(ROOT)
    paths.result_dir.mkdir(parents=True, exist_ok=True)

    source_cols = _cached_price_columns(paths)
    us_all = drop_duplicate_share_classes_available([
        ticker
        for ticker in load_sp500_membership_intervals(paths)["ticker"].dropna().astype(str).drop_duplicates()
        if ticker in source_cols
    ], source_cols)
    overlay_full = _load_overlay_prices(paths)
    as_of = _latest_common_close(overlay_full)
    sp500_pool = drop_duplicate_share_classes_available([
        ticker
        for ticker in get_sp500_members_as_of(as_of, paths)
        if ticker in us_all
    ], us_all)
    _refresh_latest_stock_cache(paths, sp500_pool, as_of)

    cached_panel = _read_source_market_cache(paths, sp500_pool, START_DATE, as_of)
    stock_prices = cached_panel["prices"].loc[START_DATE:as_of].reindex(columns=sp500_pool).sort_index().ffill()
    stock_volumes = cached_panel["volumes"].loc[START_DATE:as_of].reindex(columns=sp500_pool).fillna(0.0)
    if as_of not in stock_prices.index:
        stock_dates = stock_prices.index[stock_prices.index <= as_of]
        if stock_dates.empty:
            raise ValueError("No stock cache date is available on or before the latest overlay close.")
        as_of = pd.Timestamp(stock_dates.max())
        stock_prices = stock_prices.loc[:as_of]
        stock_volumes = stock_volumes.loc[:as_of]
    overlay = overlay_full.loc[START_DATE:as_of, ["SPY", "^VIX", *OVERLAY_ASSETS]].sort_index().ffill()

    prices = pd.concat(
        [stock_prices, overlay[OVERLAY_ASSETS].reindex(stock_prices.index).ffill()],
        axis=1,
    )
    volumes = stock_volumes.reindex(columns=prices.columns).fillna(0.0)
    volumes.loc[:, OVERLAY_ASSETS] = 1.0
    benchmark = overlay["SPY"].reindex(prices.index).ffill().rename("benchmark")
    vol_proxy = overlay["^VIX"].reindex(prices.index).ffill().rename("vol_proxy")
    common_index = prices.index.intersection(benchmark.dropna().index).intersection(vol_proxy.dropna().index)
    prices = prices.reindex(common_index).ffill()
    volumes = volumes.reindex(common_index).fillna(0.0)
    benchmark = benchmark.reindex(common_index)
    vol_proxy = vol_proxy.reindex(common_index)

    loc = prices.index.get_loc(as_of)
    train_index = prices.index[max(0, loc - LOOKBACK_DAYS + 1) : loc + 1]
    sp500_pool = [ticker for ticker in sp500_pool if ticker in prices.columns]
    us_selected = drop_duplicate_share_classes(select_point_in_time_universe(
        prices.reindex(train_index),
        volumes.reindex(train_index),
        sp500_pool,
        n_assets=US_ASSETS,
    ))
    current_assets = list(dict.fromkeys(us_selected + OVERLAY_ASSETS))
    returns = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    train_returns = returns.reindex(train_index)[current_assets].dropna(
        axis=1,
        thresh=max(int(0.85 * len(train_index)), 60),
    )
    current_assets = drop_duplicate_share_classes_available(train_returns.columns.tolist(), train_returns.columns)
    train_returns = train_returns.reindex(columns=current_assets)
    assert_no_duplicate_share_classes([asset for asset in current_assets if asset not in OVERLAY_ASSETS])
    benchmark_ret = benchmark.pct_change(fill_method=None).rename("benchmark")
    vol_proxy_ret = vol_proxy.pct_change(fill_method=None).rename("vol_proxy")

    features = compute_feature_table(
        train_returns,
        benchmark_ret.reindex(train_index),
        vol_proxy_ret.reindex(train_index),
        prices.reindex(train_index)[current_assets],
        include_momentum_features=True,
        feature_flags=FEATURE_FLAGS,
    )
    momentum_signal = build_momentum_signal(features, mode="mom_63")
    sample_cov = train_returns.cov().reindex(index=current_assets, columns=current_assets).fillna(0.0)
    asset_caps = {asset: STOCK_CAP for asset in current_assets}
    asset_caps.update({"GC=F": GOLD_CAP, "BTC-USD": BTC_CAP, "BIL": BIL_CAP})
    asset_caps = {asset: cap for asset, cap in asset_caps.items() if asset in current_assets}
    raw_weights = optimize_portfolio(
        sample_cov,
        momentum_signal,
        max_weight=max(STOCK_CAP, GOLD_CAP, BTC_CAP, BIL_CAP),
        objective_mode="mean_variance",
        asset_caps=asset_caps,
    ).sort_values(ascending=False)

    config = _best_signal_config()
    spy_cfg = config.loc["SPY"] if "SPY" in config.index else pd.Series({"MA Period": 300, "Below Exposure": 0.50})
    gold_cfg = config.loc["GOLD"] if "GOLD" in config.index else pd.Series({"MA Period": 50, "Below Exposure": 1.00})
    btc_cfg = config.loc["BTC"] if "BTC" in config.index else pd.Series({"MA Period": 50, "Below Exposure": 0.00})
    stock_exposure = _close_trend_exposure(benchmark, int(spy_cfg["MA Period"]), float(spy_cfg["Below Exposure"]))
    gold_exposure = _close_trend_exposure(prices["GC=F"], int(gold_cfg["MA Period"]), float(gold_cfg["Below Exposure"]))
    btc_exposure = _close_trend_exposure(prices["BTC-USD"], int(btc_cfg["MA Period"]), float(btc_cfg["Below Exposure"]))

    exposure_by_asset = pd.Series(1.0, index=raw_weights.index, dtype=float)
    stock_assets = [asset for asset in raw_weights.index if asset not in OVERLAY_ASSETS]
    exposure_by_asset.loc[stock_assets] = float(stock_exposure.reindex([as_of]).ffill().iloc[-1])
    if "GC=F" in exposure_by_asset.index:
        exposure_by_asset.loc["GC=F"] = float(gold_exposure.reindex([as_of]).ffill().iloc[-1])
    if "BTC-USD" in exposure_by_asset.index:
        exposure_by_asset.loc["BTC-USD"] = float(btc_exposure.reindex([as_of]).ffill().iloc[-1])
    if "BIL" in exposure_by_asset.index:
        exposure_by_asset.loc["BIL"] = 1.0

    effective = raw_weights.mul(exposure_by_asset).clip(lower=0.0)
    cash_weight = max(0.0, 1.0 - float(effective.sum()))
    if cash_weight > 1e-12:
        effective.loc["CASH"] = cash_weight

    latest = effective.rename("Effective Weight").reset_index().rename(columns={"index": "Asset"})
    latest["Raw Optimizer Weight"] = latest["Asset"].map(raw_weights).fillna(0.0)
    latest["Daily Exposure"] = latest["Asset"].map(exposure_by_asset).fillna(1.0)
    latest["Date"] = as_of.date().isoformat()
    latest["Strategy"] = STRATEGY
    latest["Sleeve"] = "US Equity"
    latest.loc[latest["Asset"].eq("GC=F"), "Sleeve"] = "Gold"
    latest.loc[latest["Asset"].eq("BTC-USD"), "Sleeve"] = "BTC"
    latest.loc[latest["Asset"].eq("BIL"), "Sleeve"] = "BIL"
    latest.loc[latest["Asset"].eq("CASH"), "Sleeve"] = "Cash"
    latest["Effective Weight %"] = latest["Effective Weight"].mul(100.0)
    latest["Raw Optimizer Weight %"] = latest["Raw Optimizer Weight"].mul(100.0)
    latest["Last Exposure Date"] = as_of.date().isoformat()
    latest["Signal Source Close Date"] = _source_close_date(stock_exposure.index, as_of)
    latest["Latest Cache Trading Date"] = as_of.date().isoformat()
    latest["Daily Exposure Variant"] = "Asset-level SPY/Gold/BTC trend"
    latest = latest.loc[latest["Effective Weight"].abs() > 1e-12].sort_values("Effective Weight", ascending=False)
    assert_no_duplicate_share_classes(
        latest.loc[latest["Sleeve"].eq("US Equity"), "Asset"].dropna().astype(str).tolist()
    )

    sleeve_latest = latest.groupby("Sleeve", as_index=False)["Effective Weight"].sum()
    sleeve_latest["Date"] = as_of.date().isoformat()
    sleeve_latest["Effective Weight %"] = sleeve_latest["Effective Weight"].mul(100.0)
    sleeve_latest = sleeve_latest.sort_values("Effective Weight", ascending=False)

    meta = pd.DataFrame(
        [
            {
                "Date": as_of.date().isoformat(),
                "Strategy": STRATEGY,
                "Train Start": pd.Timestamp(train_index.min()).date().isoformat(),
                "Train End": pd.Timestamp(train_index.max()).date().isoformat(),
                "Lookback Days": len(train_index),
                "Selected US Assets": len([asset for asset in current_assets if asset not in OVERLAY_ASSETS]),
                "Stock Exposure": float(stock_exposure.loc[as_of]),
                "Gold Exposure": float(gold_exposure.loc[as_of]),
                "BTC Exposure": float(btc_exposure.loc[as_of]) if "BTC-USD" in prices.columns else np.nan,
                "Weight Timing Note": "Fresh standalone recheck from latest common close in cache; signals are lagged by one session.",
            }
        ]
    )

    _write_outputs(latest, sleeve_latest, meta)
    print(meta.to_string(index=False))
    print(latest[["Asset", "Sleeve", "Effective Weight", "Raw Optimizer Weight", "Daily Exposure"]].to_string(index=False))
    print(sleeve_latest.to_string(index=False))


if __name__ == "__main__":
    main()
