from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "data" / "precomputed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

START_YEARS = 10
INITIAL_VALUE = 10_000.0
RISK_FREE_RATE = 0.03
REBALANCE_MONTHS = (1, 4, 7, 10)


def _repo_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


def _read_curve_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    frame.index.name = "Date"
    return frame.sort_index()


def _curve_from_returns(returns: pd.Series) -> pd.Series:
    clean = returns.dropna()
    curve = pd.Series(np.nan, index=returns.index, dtype=float)
    if clean.empty:
        return curve
    curve.loc[clean.index] = INITIAL_VALUE * (1.0 + clean.fillna(0.0)).cumprod()
    return curve


def _returns_from_curve(curve: pd.Series) -> pd.Series:
    return curve.astype(float).pct_change(fill_method=None).fillna(0.0)


def _metrics(curve: pd.Series) -> dict[str, float]:
    values = curve.dropna().astype(float)
    returns = values.pct_change(fill_method=None).dropna()
    if values.empty or returns.empty:
        return {
            "Total Return": np.nan,
            "CAGR": np.nan,
            "Annual Vol": np.nan,
            "Sharpe": np.nan,
            "Sortino": np.nan,
            "Max Drawdown": np.nan,
            "Hit Rate": np.nan,
        }
    years = len(returns) / 252.0
    total_return = values.iloc[-1] / values.iloc[0] - 1.0
    cagr = (values.iloc[-1] / values.iloc[0]) ** (1.0 / years) - 1.0 if years > 0 else np.nan
    annual_vol = returns.std() * np.sqrt(252.0)
    sharpe = (cagr - RISK_FREE_RATE) / annual_vol if annual_vol > 0 else np.nan
    downside = returns[returns < 0].std() * np.sqrt(252.0)
    sortino = (cagr - RISK_FREE_RATE) / downside if downside > 0 else np.nan
    drawdown = values / values.cummax() - 1.0
    return {
        "Total Return": float(total_return),
        "CAGR": float(cagr),
        "Annual Vol": float(annual_vol),
        "Sharpe": float(sharpe),
        "Sortino": float(sortino),
        "Max Drawdown": float(drawdown.min()),
        "Hit Rate": float((returns > 0).mean()),
    }


def _quarterly_rebalanced_returns(sleeves: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    weights_s = pd.Series(weights, dtype=float).reindex(sleeves.columns).fillna(0.0)
    if weights_s.sum() <= 0:
        raise ValueError("Weights must be positive.")
    weights_s = weights_s / weights_s.sum()
    month_ends = sleeves.groupby(sleeves.index.to_period("M")).tail(1).index
    rebalance_dates = {dt for dt in month_ends if dt.month in REBALANCE_MONTHS}
    values = weights_s * INITIAL_VALUE
    rows: list[tuple[pd.Timestamp, float]] = []
    for dt, row in sleeves.fillna(0.0).iterrows():
        before = float(values.sum())
        values = values * (1.0 + row)
        after = float(values.sum())
        rows.append((dt, after / before - 1.0 if before > 0 else 0.0))
        if dt in rebalance_dates and after > 0:
            values = weights_s * after
    return pd.Series(dict(rows), name="Portfolio").sort_index()


def _trend_exposure(price: pd.Series, below: float, ma_days: int = 200) -> pd.Series:
    ma = price.rolling(ma_days, min_periods=40).mean()
    exposure = pd.Series(1.0, index=price.index, dtype=float)
    exposure.loc[price < ma] = below
    exposure.loc[ma.isna()] = 1.0
    return exposure


def _sp_daily_exposure(spy_thb: pd.Series, vix: pd.Series) -> pd.Series:
    ma200 = spy_thb.rolling(200, min_periods=40).mean()
    drawdown = spy_thb / spy_thb.cummax() - 1.0
    exposure = pd.Series(1.0, index=spy_thb.index, dtype=float)
    exposure = pd.concat(
        [
            exposure,
            pd.Series(np.where(spy_thb < ma200, 0.65, 1.0), index=spy_thb.index),
            pd.Series(np.where(drawdown <= -0.08, 0.50, 1.0), index=spy_thb.index),
            pd.Series(np.where(drawdown <= -0.15, 0.25, 1.0), index=spy_thb.index),
            pd.Series(np.where(vix.reindex(spy_thb.index).ffill() >= 28.0, 0.50, 1.0), index=spy_thb.index),
            pd.Series(np.where(vix.reindex(spy_thb.index).ffill() >= 35.0, 0.25, 1.0), index=spy_thb.index),
        ],
        axis=1,
    ).min(axis=1)
    exposure.loc[ma200.isna()] = 1.0
    return exposure.clip(0.0, 1.0)


def _active_members(intervals: pd.DataFrame, as_of: pd.Timestamp, ticker_col: str = "ticker") -> list[str]:
    start = pd.to_datetime(intervals["start_date"], errors="coerce")
    end = pd.to_datetime(intervals["end_date"], errors="coerce")
    active = intervals.loc[(start <= as_of) & (end.isna() | (end >= as_of)), ticker_col]
    return active.dropna().astype(str).str.upper().drop_duplicates().tolist()


def _pit_equal_weight_returns(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    intervals: pd.DataFrame,
    n_assets: int,
    label: str,
    fx: pd.Series | None = None,
) -> pd.Series:
    prices = prices.sort_index().ffill()
    volumes = volumes.reindex(prices.index).reindex(columns=prices.columns).fillna(0.0)
    if fx is not None:
        prices = prices.mul(fx.reindex(prices.index).ffill(), axis=0)
    month_ends = prices.groupby(prices.index.to_period("M")).tail(1).index
    month_ends = month_ends[month_ends >= prices.index.min() + pd.DateOffset(years=1)]
    weights = pd.Series(0.0, index=prices.columns, dtype=float)
    rows: list[tuple[pd.Timestamp, float]] = []
    for idx, rebalance_date in enumerate(month_ends):
        next_date = month_ends[idx + 1] if idx + 1 < len(month_ends) else prices.index[-1]
        train_start = rebalance_date - pd.DateOffset(days=252)
        train_prices = prices.loc[(prices.index > train_start) & (prices.index <= rebalance_date)]
        train_volumes = volumes.loc[train_prices.index]
        active = [ticker for ticker in _active_members(intervals, rebalance_date) if ticker in prices.columns]
        if active:
            availability = train_prices[active].notna().mean()
            liquidity = (train_prices[active].ffill() * train_volumes[active]).median()
            ranked = (
                pd.DataFrame({"availability": availability, "liquidity": liquidity})
                .fillna(0.0)
                .query("availability >= 0.75")
                .sort_values(["liquidity", "availability"], ascending=False)
            )
            selected = ranked.head(n_assets).index.tolist()
            weights = pd.Series(0.0, index=prices.columns, dtype=float)
            if selected:
                weights.loc[selected] = 1.0 / len(selected)
        period = prices.loc[(prices.index > rebalance_date) & (prices.index <= next_date)]
        period_returns = period.pct_change(fill_method=None).fillna(0.0)
        for dt, row in period_returns.iterrows():
            rows.append((dt, float(row.reindex(weights.index).fillna(0.0) @ weights)))
    return pd.Series(dict(rows), name=label).sort_index()


def main() -> None:
    overlay_path = _repo_path("data", "cache", "dynamic_factor_copula", "overlay_compare_prices.parquet")
    if not overlay_path.exists():
        overlay_path = _repo_path("..", "dynamic_port_opt", "data", "cache", "dynamic_factor_copula", "overlay_compare_prices.parquet")
    overlay = pd.read_parquet(overlay_path).sort_index().ffill()
    overlay.index = pd.to_datetime(overlay.index)
    overlay = overlay.dropna(subset=["SPY", "GC=F", "BTC-USD", "USDTHB=X"])
    data_end = overlay.index.max()
    start = data_end - pd.DateOffset(years=START_YEARS)
    overlay = overlay.loc[overlay.index >= start].copy()

    fx = overlay["USDTHB=X"].ffill()
    spy_thb = overlay["SPY"] * fx
    gold_thb = overlay["GC=F"] * fx
    btc_thb = overlay["BTC-USD"] * fx
    vix = overlay["^VIX"].ffill()

    sleeve_returns = pd.DataFrame(
        {
            "SPY": spy_thb.pct_change(fill_method=None).fillna(0.0),
            "SPY_DAILY_EXPOSURE": spy_thb.pct_change(fill_method=None).fillna(0.0) * _sp_daily_exposure(spy_thb, vix),
            "GOLD": gold_thb.pct_change(fill_method=None).fillna(0.0),
            "GOLD_DAILY_EXPOSURE": gold_thb.pct_change(fill_method=None).fillna(0.0) * _trend_exposure(gold_thb, 0.50),
            "BTC": btc_thb.pct_change(fill_method=None).fillna(0.0),
            "BTC_DAILY_EXPOSURE": btc_thb.pct_change(fill_method=None).fillna(0.0) * _trend_exposure(btc_thb, 0.00),
        }
    )

    strategy_returns: dict[str, pd.Series] = {
        "S&P buy & hold": sleeve_returns["SPY"],
        "S&P daily exposure": sleeve_returns["SPY_DAILY_EXPOSURE"],
        "S&P Gold BTC 60/30/10": _quarterly_rebalanced_returns(
            sleeve_returns[["SPY", "GOLD", "BTC"]],
            {"SPY": 0.60, "GOLD": 0.30, "BTC": 0.10},
        ),
        "S&P Gold BTC 70/20/10": _quarterly_rebalanced_returns(
            sleeve_returns[["SPY", "GOLD", "BTC"]],
            {"SPY": 0.70, "GOLD": 0.20, "BTC": 0.10},
        ),
        "S&P Gold BTC daily exposure 60/30/10": _quarterly_rebalanced_returns(
            sleeve_returns[["SPY_DAILY_EXPOSURE", "GOLD_DAILY_EXPOSURE", "BTC_DAILY_EXPOSURE"]],
            {"SPY_DAILY_EXPOSURE": 0.60, "GOLD_DAILY_EXPOSURE": 0.30, "BTC_DAILY_EXPOSURE": 0.10},
        ),
        "S&P Gold BTC daily exposure 70/20/10": _quarterly_rebalanced_returns(
            sleeve_returns[["SPY_DAILY_EXPOSURE", "GOLD_DAILY_EXPOSURE", "BTC_DAILY_EXPOSURE"]],
            {"SPY_DAILY_EXPOSURE": 0.70, "GOLD_DAILY_EXPOSURE": 0.20, "BTC_DAILY_EXPOSURE": 0.10},
        ),
    }

    source_prices = _repo_path("data", "cache", "portopt_optimizer_proof", "20Y", "prices.parquet")
    source_volumes = _repo_path("data", "cache", "portopt_optimizer_proof", "20Y", "volumes.parquet")
    sp500_intervals = _repo_path("data", "sp500", "sp500_ticker_start_end.csv")
    if not sp500_intervals.exists():
        sp500_intervals = _repo_path("..", "sp500", "sp500_ticker_start_end.csv")
    if source_prices.exists() and source_volumes.exists() and sp500_intervals.exists():
        us_prices = pd.read_parquet(source_prices).loc[overlay.index.min() : overlay.index.max()]
        us_volumes = pd.read_parquet(source_volumes).loc[us_prices.index]
        us_intervals = pd.read_csv(sp500_intervals)
        for n_assets in [20, 30, 50]:
            strategy_returns[f"Top liquidity US {n_assets} PIT"] = _pit_equal_weight_returns(
                us_prices, us_volumes, us_intervals, n_assets, f"Top liquidity US {n_assets} PIT", fx=fx
            )

    th_prices_path = _repo_path("data", "cache", "dynamic_factor_copula", "extra_prices.parquet")
    th_volumes_path = _repo_path("data", "cache", "dynamic_factor_copula", "extra_volumes.parquet")
    th_intervals_path = _repo_path("data", "thai_stock", "set100_ticker_start_end.csv")
    if th_prices_path.exists() and th_volumes_path.exists() and th_intervals_path.exists():
        th_prices = pd.read_parquet(th_prices_path).loc[overlay.index.min() : overlay.index.max()]
        th_volumes = pd.read_parquet(th_volumes_path).loc[th_prices.index]
        th_intervals = pd.read_csv(th_intervals_path)
        for n_assets in [20, 30, 50]:
            strategy_returns[f"Top liquidity SET {n_assets} PIT"] = _pit_equal_weight_returns(
                th_prices, th_volumes, th_intervals, n_assets, f"Top liquidity SET {n_assets} PIT"
            )

    curve_sources = [
        _repo_path("..", "dynamic_port_opt", "result", "joint_confirm_603010_504d_1m_overlay_curves_thb.csv"),
        _repo_path("..", "dynamic_port_opt", "result", "us_th_joint_model_curves_thb.csv"),
        _repo_path("result", "us_th_side_trigger_reallocation_curves_thb.csv"),
    ]
    for path in curve_sources:
        if not path.exists():
            continue
        curves = _read_curve_csv(path).loc[overlay.index.min() : overlay.index.max()]
        for column in curves.columns:
            if column not in strategy_returns:
                strategy_returns[column] = _returns_from_curve(curves[column])

    returns = pd.DataFrame(strategy_returns).sort_index().loc[overlay.index.min() : overlay.index.max()]
    curves = returns.apply(_curve_from_returns)
    returns.to_parquet(OUT_DIR / "streamlit_10y_strategy_returns.parquet")
    curves.to_parquet(OUT_DIR / "streamlit_10y_strategy_curves.parquet")
    sleeve_returns.to_parquet(OUT_DIR / "streamlit_10y_sleeve_returns.parquet")

    rows = []
    for strategy in curves.columns:
        row = {
            "Strategy": strategy,
            "Start": curves[strategy].dropna().index.min().date().isoformat(),
            "End": curves[strategy].dropna().index.max().date().isoformat(),
            **_metrics(curves[strategy]),
        }
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values(["Sharpe", "CAGR"], ascending=[False, False])
    summary.to_csv(OUT_DIR / "streamlit_10y_strategy_summary.csv", index=False)

    metadata = {
        "generated_at": pd.Timestamp.now(tz="Asia/Bangkok").isoformat(),
        "data_start": overlay.index.min().date().isoformat(),
        "data_end": overlay.index.max().date().isoformat(),
        "currency": "THB",
        "notes": [
            "This is a frozen deploy-friendly performance dataset.",
            "It stores strategy/sleeve return series, not raw stock-level cache.",
            "Latest rebalance weights are intentionally not included.",
        ],
        "files": {
            "returns": "data/precomputed/streamlit_10y_strategy_returns.parquet",
            "curves": "data/precomputed/streamlit_10y_strategy_curves.parquet",
            "sleeves": "data/precomputed/streamlit_10y_sleeve_returns.parquet",
            "summary": "data/precomputed/streamlit_10y_strategy_summary.csv",
        },
    }
    (OUT_DIR / "streamlit_10y_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"\nWrote precomputed dataset to {OUT_DIR}")


if __name__ == "__main__":
    main()
