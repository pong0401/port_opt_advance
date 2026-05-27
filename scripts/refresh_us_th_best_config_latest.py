from __future__ import annotations

from datetime import timedelta
from pathlib import Path
import sys

import pandas as pd
import yfinance as yf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
SCRIPTS = PROJECT_ROOT / "scripts"
for path in [SRC, SCRIPTS]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dynamic_factor_copula import default_paths  # noqa: E402
import run_us_th_best_config as best_config  # noqa: E402


YF_CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "dynamic_factor_copula" / ".yfinance"
LIVE_LATEST_WEIGHTS_FILE = "us_th_side_trigger_latest_asset_weights_live_thb.csv"
LIVE_LATEST_METADATA_FILE = "us_th_side_trigger_latest_asset_weights_live_metadata.json"
BEST_ASSET_LIVE_LATEST_WEIGHTS_FILE = "us_th_best_asset_sweep_latest_effective_weights_live_thb.csv"


def _download_latest(tickers: list[str], start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    price_frames: list[pd.Series] = []
    volume_frames: list[pd.Series] = []
    for ticker in tickers:
        raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False, threads=False)
        if raw.empty:
            print(f"Warning: no fresh rows for {ticker}")
            continue
        if isinstance(raw.columns, pd.MultiIndex):
            close_data = raw["Close"].iloc[:, 0] if "Close" in raw.columns.get_level_values(0) else pd.Series(dtype=float)
            volume_data = raw["Volume"].iloc[:, 0] if "Volume" in raw.columns.get_level_values(0) else pd.Series(dtype=float)
        else:
            close_data = raw["Close"] if "Close" in raw.columns else pd.Series(dtype=float)
            volume_data = raw["Volume"] if "Volume" in raw.columns else pd.Series(dtype=float)
        close = close_data.dropna().rename(ticker)
        if close.empty:
            print(f"Warning: empty close series for {ticker}")
            continue
        volume = volume_data.reindex(close.index).fillna(0.0).rename(ticker)
        price_frames.append(close)
        volume_frames.append(volume)
    prices = pd.concat(price_frames, axis=1).sort_index() if price_frames else pd.DataFrame()
    volumes = pd.concat(volume_frames, axis=1).sort_index() if volume_frames else pd.DataFrame()
    return prices, volumes


def _merge_parquet(path: Path, update: pd.DataFrame) -> None:
    if update.empty:
        return
    existing = pd.read_parquet(path).sort_index() if path.exists() else pd.DataFrame()
    combined = pd.concat([existing, update]).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    combined.to_parquet(path)


def _current_best_universe() -> list[str]:
    weight_file = PROJECT_ROOT / "result" / "us_th_best_asset_sweep_dynamic_weight_history_thb.csv"
    if weight_file.exists():
        frame = pd.read_csv(weight_file, index_col=0)
        return [column for column in frame.columns if column not in {"GOLD", "BTC", "CASH"}]
    exposure_file = PROJECT_ROOT / "result" / "us_th_side_trigger_daily_asset_exposure_realloc_stock_thb.csv"
    frame = pd.read_csv(exposure_file, index_col=0)
    return [column for column in frame.columns if column not in {"GOLD", "BTC", "CASH"}]


def _write_live_latest_weights(
    model_results: dict[str, object],
    panel_prices: pd.DataFrame,
    latest_date: str,
    downloaded_count: int,
    requested_count: int,
) -> pd.DataFrame:
    paths = default_paths(PROJECT_ROOT)
    overlay_prices = best_config.load_overlay_compare_prices(
        paths,
        start_date="2016-01-01",
        end_date=latest_date,
        tickers=["SPY", "GC=F", "BTC-USD", "^VIX", "USDTHB=X"],
    ).dropna(subset=["SPY", "GC=F", "BTC-USD", "^VIX", "USDTHB=X"])
    set_index = pd.read_parquet(paths.local_cache_root / "extra_prices.parquet")["^SET.BK"].loc[
        best_config.START_DATE : latest_date
    ].ffill()

    gold_thb = overlay_prices["GC=F"].mul(overlay_prices["USDTHB=X"])
    btc_thb = overlay_prices["BTC-USD"].mul(overlay_prices["USDTHB=X"])
    asset_prices = panel_prices.copy()
    asset_prices["GOLD"] = gold_thb.reindex(asset_prices.index).ffill()
    asset_prices["BTC"] = btc_thb.reindex(asset_prices.index).ffill()
    asset_returns = asset_prices.pct_change(fill_method=None).fillna(0.0)

    sample_returns = pd.Series(0.0, index=asset_returns.index, name="dummy")
    _, us_exposure_df = best_config.apply_daily_exposure_overlay(
        sample_returns,
        overlay_prices["SPY"].reindex(asset_returns.index).ffill(),
        overlay_prices["^VIX"].reindex(asset_returns.index).ffill(),
    )
    _, th_exposure_df = best_config.apply_daily_exposure_overlay(
        sample_returns,
        set_index.reindex(asset_returns.index).ffill(),
        None,
    )
    gold_exposure = best_config.compare_trend_exposure(overlay_prices["GC=F"], 0.50)
    btc_exposure = best_config.compare_trend_exposure(overlay_prices["BTC-USD"], 0.00)
    sleeve_weight_history = best_config._weights_history_to_frame(
        model_results["weights_history"]["Dynamic HMM Copula"]
    )
    live_exposure = best_config._side_trigger_asset_exposure(
        prices=asset_prices,
        sleeve_weight_history=sleeve_weight_history,
        us_exposure=us_exposure_df["Daily Exposure"],
        th_exposure=th_exposure_df["Daily Exposure"],
        gold_exposure=gold_exposure,
        btc_exposure=btc_exposure,
        reallocate_stock_sleeve=True,
    )

    latest_dt = live_exposure.index.max()
    latest = live_exposure.loc[latest_dt].rename("Portfolio Exposure").reset_index()
    latest.columns = ["Asset", "Portfolio Exposure"]
    latest = latest.loc[latest["Portfolio Exposure"] > 1e-10].copy()
    latest["Portfolio Exposure %"] = latest["Portfolio Exposure"] * 100.0
    latest["Sleeve"] = latest["Asset"].map(best_config._asset_sleeve)
    latest["Trigger Source"] = latest["Asset"].map(best_config._asset_trigger_source)
    latest.insert(0, "Date", pd.Timestamp(latest_dt).date().isoformat())
    latest = latest.sort_values("Portfolio Exposure %", ascending=False)
    latest.to_csv(paths.result_dir / LIVE_LATEST_WEIGHTS_FILE, index=False)

    metadata = {
        "calculated_at": pd.Timestamp.now(tz="Asia/Bangkok").isoformat(),
        "data_as_of": pd.Timestamp(latest_dt).date().isoformat(),
        "downloaded_tickers": downloaded_count,
        "requested_tickers": requested_count,
        "universe_tickers": len(sleeve_weight_history.columns),
        "note": "Latest live weights only. Historical backtest summary/curves are intentionally not updated.",
    }
    (paths.result_dir / LIVE_LATEST_METADATA_FILE).write_text(
        pd.Series(metadata).to_json(indent=2),
        encoding="utf-8",
    )
    return latest


def _write_best_asset_live_latest_weights(
    model_results: dict[str, object],
    benchmark: pd.Series,
    latest_date: str,
) -> pd.DataFrame:
    paths = default_paths(PROJECT_ROOT)
    overlay_prices = best_config.load_overlay_compare_prices(
        paths,
        start_date="2016-01-01",
        end_date=latest_date,
        tickers=["SPY", "GC=F", "BTC-USD", "^VIX", "USDTHB=X"],
    ).dropna(subset=["SPY", "GC=F", "BTC-USD", "^VIX", "USDTHB=X"])
    gold_thb = overlay_prices["GC=F"].mul(overlay_prices["USDTHB=X"])
    btc_thb = overlay_prices["BTC-USD"].mul(overlay_prices["USDTHB=X"])
    gold_returns = gold_thb.pct_change(fill_method=None).fillna(0.0)
    btc_returns = btc_thb.pct_change(fill_method=None).fillna(0.0)
    gold_exposure = best_config.compare_trend_exposure(overlay_prices["GC=F"], 0.50)
    btc_exposure = best_config.compare_trend_exposure(overlay_prices["BTC-USD"], 0.00)

    sample_index = model_results["nav"]["Dynamic HMM Copula"].index.intersection(benchmark.index).sort_values()
    equity_returns = (
        model_results["nav"]["Dynamic HMM Copula"]
        .reindex(sample_index)
        .pct_change(fill_method=None)
        .fillna(0.0)
    )
    _, equity_exposure = best_config.apply_daily_exposure_overlay(
        equity_returns,
        benchmark.reindex(sample_index).ffill(),
        overlay_prices["^VIX"].reindex(sample_index).ffill(),
    )
    sleeves = pd.concat(
        {
            "JOINT_EQUITY": equity_returns,
            "GOLD": gold_returns,
            "BTC": btc_returns,
        },
        axis=1,
    ).dropna()
    exposures = pd.concat(
        {
            "JOINT_EQUITY": equity_exposure["Daily Exposure"],
            "GOLD": gold_exposure,
            "BTC": btc_exposure,
        },
        axis=1,
    ).reindex(sleeves.index).ffill().bfill()
    strategic_weights = pd.Series({"JOINT_EQUITY": 0.60, "GOLD": 0.30, "BTC": 0.10}, dtype=float)
    sleeve_weight_history = best_config._weights_history_to_frame(
        model_results["weights_history"]["Dynamic HMM Copula"]
    )
    _, _, cash_drag_sleeves = best_config._dynamic_rebalanced_returns(
        sleeves,
        strategic_weights,
        exposures,
        rebalance_months=1,
        transaction_cost_bps=0.0,
        reallocate_cash=False,
    )
    cash_drag_assets = best_config._daily_asset_exposure_from_sleeves(sleeve_weight_history, cash_drag_sleeves)

    latest_dt = cash_drag_assets.index.max()
    latest = cash_drag_assets.loc[latest_dt].rename("Portfolio Exposure").reset_index()
    latest.columns = ["Asset", "Portfolio Exposure"]
    latest = latest.loc[latest["Portfolio Exposure"] > 1e-10].copy()
    latest["Portfolio Exposure %"] = latest["Portfolio Exposure"] * 100.0
    latest["Sleeve"] = latest["Asset"].map(best_config._asset_sleeve)
    latest["Trigger Source"] = latest["Asset"].map(
        lambda asset: "Dynamic HMM stock sleeve"
        if asset not in {"CASH", "GOLD", "BTC"} and not str(asset).endswith(".BK")
        else (
            "Dynamic HMM Thai stock sleeve"
            if str(asset).endswith(".BK")
            else best_config._asset_trigger_source(asset)
        )
    )
    latest.insert(0, "Date", pd.Timestamp(latest_dt).date().isoformat())
    latest = latest.sort_values("Portfolio Exposure %", ascending=False)
    latest.to_csv(paths.result_dir / BEST_ASSET_LIVE_LATEST_WEIGHTS_FILE, index=False)
    return latest


def main() -> None:
    paths = default_paths(PROJECT_ROOT)
    overlay_file = paths.local_cache_root / "overlay_compare_prices.parquet"
    overlay = pd.read_parquet(overlay_file).sort_index().ffill()
    latest_date = pd.Timestamp(overlay.index.max()).date()
    best_config.END_DATE = latest_date.isoformat()

    universe = _current_best_universe()
    trigger_tickers = ["SPY", "^VIX", "^SET.BK"]
    tickers = list(dict.fromkeys(universe + trigger_tickers))

    YF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(YF_CACHE_DIR))
    start = (latest_date - timedelta(days=45)).isoformat()
    end = (latest_date + timedelta(days=1)).isoformat()
    prices, volumes = _download_latest(tickers, start, end)
    _merge_parquet(paths.local_cache_root / "extra_prices.parquet", prices)
    _merge_parquet(paths.local_cache_root / "extra_volumes.parquet", volumes)

    panel_prices, panel_volumes, benchmark, vol_proxy, _ = best_config._load_thb_panel(universe)
    results = best_config._run_model_on_prices(
        panel_prices,
        panel_volumes,
        benchmark,
        vol_proxy,
        objective_mode=best_config.BEST_OBJECTIVE,
        max_weight=best_config.BEST_ASSET_SWEEP_CASE["max_weight"],
        include_latest_weight_asof=True,
    )
    latest_exposure = _write_live_latest_weights(
        results,
        panel_prices,
        latest_date.isoformat(),
        downloaded_count=len(prices.columns),
        requested_count=len(tickers),
    )
    latest_best_asset_exposure = _write_best_asset_live_latest_weights(
        results,
        benchmark,
        latest_date.isoformat(),
    )
    print(f"Updated Strategy B live latest weights through {latest_date.isoformat()}")
    print(f"Downloaded latest rows for {len(prices.columns)} / {len(tickers)} tickers")
    print("Backtest summary/curves were not updated.")
    print(latest_exposure.head(12).to_string(index=False))
    print("Updated Dynamic HMM cash-drag latest effective weights.")
    print(latest_best_asset_exposure.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
