from __future__ import annotations

from datetime import date
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from portfolio_engine import calculate_performance_metrics


START_DATE = "2016-01-01"
END_DATE = date.today().isoformat()
RISK_FREE_RATE = 0.03
DBMF_WEIGHTS = [x / 100 for x in range(0, 31, 5)]
DBMF_BELOW_EXPOSURES = [0.00, 0.25, 0.50, 0.75]

TICKERS = ["SPY", "GC=F", "BTC-USD", "DBMF", "^VIX", "USDTHB=X"]
YFINANCE_CACHE_DIR = PROJECT_ROOT / ".yfinance_notebook"
YFINANCE_CACHE_DIR.mkdir(exist_ok=True)
yf.set_tz_cache_location(str(YFINANCE_CACHE_DIR))


def download_close_prices() -> pd.DataFrame:
    raw = yf.download(
        TICKERS,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].reindex(columns=TICKERS).ffill()
    else:
        prices = raw[["Close"]].rename(columns={"Close": TICKERS[0]})
    prices.index = pd.to_datetime(prices.index)
    return prices.dropna(subset=["SPY", "GC=F", "BTC-USD", "DBMF", "^VIX", "USDTHB=X"])


def cash_returns(fx_returns: pd.Series, cash_mode: str, index: pd.Index) -> pd.Series:
    if cash_mode == "USD":
        return fx_returns.reindex(index).fillna(0.0)
    if cash_mode in {"THB", "USD_STATIC"}:
        return pd.Series(0.0, index=index, dtype=float)
    raise ValueError(cash_mode)


def trend_exposure(price: pd.Series, below: float, ma_days: int = 200, min_periods: int = 40) -> pd.Series:
    ma = price.rolling(ma_days, min_periods=min_periods).mean()
    exposure = pd.Series(1.0, index=price.index, dtype=float)
    exposure.loc[price < ma] = below
    exposure.loc[ma.isna()] = 1.0
    return exposure


def sp_exposure(price: pd.Series, vix: pd.Series, cfg: dict[str, float]) -> pd.Series:
    ma = price.rolling(200, min_periods=40).mean()
    drawdown = price / price.cummax() - 1.0
    candidates = pd.concat(
        [
            pd.Series(1.0, index=price.index, dtype=float),
            pd.Series(np.where(price < ma, cfg["sp_trend_cap"], 1.0), index=price.index, dtype=float),
            pd.Series(np.where(drawdown <= -0.08, cfg["sp_warn_cap"], 1.0), index=price.index, dtype=float),
            pd.Series(np.where(drawdown <= -0.15, cfg["sp_crash_cap"], 1.0), index=price.index, dtype=float),
            pd.Series(np.where(vix >= 28.0, cfg["sp_warn_cap"], 1.0), index=price.index, dtype=float),
            pd.Series(np.where(vix >= 35.0, cfg["sp_crash_cap"], 1.0), index=price.index, dtype=float),
        ],
        axis=1,
    )
    return candidates.min(axis=1)


def apply_exposure(
    asset_returns: pd.Series,
    exposure: pd.Series,
    fx_returns: pd.Series,
    cash_mode: str,
) -> pd.Series:
    cash_ret = cash_returns(fx_returns, cash_mode, asset_returns.index)
    exposure = exposure.reindex(asset_returns.index).ffill().fillna(1.0)
    return asset_returns * exposure + cash_ret * (1.0 - exposure)


def rebalanced_portfolio(
    sleeve_returns: pd.DataFrame,
    weights: pd.Series,
    frequency: str = "quarterly",
) -> pd.Series:
    weights = weights.reindex(sleeve_returns.columns).fillna(0.0)
    weights = weights / weights.sum()
    values = weights * 10_000.0
    month_ends = sleeve_returns.groupby(sleeve_returns.index.to_period("M")).tail(1).index
    quarterly_dates = {dt for dt in month_ends if dt.month in (1, 4, 7, 10)}
    rows = []
    for dt, row in sleeve_returns.iterrows():
        total_before = float(values.sum())
        values = values * (1.0 + row.fillna(0.0))
        total_after = float(values.sum())
        rows.append((dt, total_after / total_before - 1.0 if total_before > 0 else 0.0))
        if frequency == "daily" or (frequency == "quarterly" and dt in quarterly_dates):
            values = weights * total_after
    return pd.Series(dict(rows), name="Portfolio").sort_index()


def main() -> None:
    prices = download_close_prices()
    fx = prices["USDTHB=X"].ffill()
    fx_returns = fx.pct_change(fill_method=None).fillna(0.0)

    local_prices = pd.DataFrame(index=prices.index)
    for ticker in ["SPY", "GC=F", "BTC-USD", "DBMF"]:
        local_prices[ticker] = prices[ticker] * fx
    local_prices["VIX"] = prices["^VIX"]
    local_prices = local_prices.dropna()

    cfg = {
        "sp_trend_cap": 0.50,
        "sp_warn_cap": 0.35,
        "sp_crash_cap": 0.15,
        "gold_below": 0.25,
        "btc_below": 0.00,
        "cash_mode": "THB",
    }

    returns = local_prices[["SPY", "GC=F", "BTC-USD", "DBMF"]].pct_change(fill_method=None).fillna(0.0)
    base_exposures = pd.concat(
        {
            "SP500_OVERLAY": sp_exposure(local_prices["SPY"], local_prices["VIX"], cfg),
            "GOLD": trend_exposure(local_prices["GC=F"], cfg["gold_below"]),
            "BTC": trend_exposure(local_prices["BTC-USD"], cfg["btc_below"]),
        },
        axis=1,
    ).reindex(local_prices.index).ffill()

    rows = []
    curves = {}
    exposure_snapshots = {}
    for dbmf_below in DBMF_BELOW_EXPOSURES:
        exposures = base_exposures.copy()
        exposures["DBMF"] = trend_exposure(local_prices["DBMF"], dbmf_below).reindex(local_prices.index).ffill()
        sleeves = pd.concat(
            {
                "SP500_OVERLAY": apply_exposure(returns["SPY"], exposures["SP500_OVERLAY"], fx_returns, cfg["cash_mode"]),
                "GOLD": apply_exposure(returns["GC=F"], exposures["GOLD"], fx_returns, cfg["cash_mode"]),
                "BTC": apply_exposure(returns["BTC-USD"], exposures["BTC"], fx_returns, cfg["cash_mode"]),
                "DBMF": apply_exposure(returns["DBMF"], exposures["DBMF"], fx_returns, cfg["cash_mode"]),
            },
            axis=1,
        ).dropna()
        exposure_snapshots[f"DBMF below {dbmf_below:.0%}"] = exposures["DBMF"]

        for dbmf_weight in DBMF_WEIGHTS:
            weights = pd.Series(
                {
                    "SP500_OVERLAY": 0.70 - dbmf_weight,
                    "GOLD": 0.20,
                    "BTC": 0.10,
                    "DBMF": dbmf_weight,
                },
                dtype=float,
            )
            if (weights < -1e-12).any():
                continue
            port_returns = rebalanced_portfolio(sleeves, weights, frequency="quarterly")
            curve = pd.DataFrame({"PortValue": 10_000.0 * (1.0 + port_returns).cumprod()}, index=port_returns.index)
            row = {
                "DBMF Weight": dbmf_weight,
                "SP500 Weight": weights["SP500_OVERLAY"],
                "DBMF Below Exposure": dbmf_below,
            }
            row.update(calculate_performance_metrics(curve, RISK_FREE_RATE).set_index("Metric")["Value"].to_dict())
            row["Avg DBMF Exposure"] = float(exposures["DBMF"].reindex(sleeves.index).mean())
            row["Start"] = sleeves.index.min().date().isoformat()
            row["End"] = sleeves.index.max().date().isoformat()
            rows.append(row)
            curves[f"DBMF {dbmf_weight:.0%} below {dbmf_below:.0%}"] = curve["PortValue"]

    summary = pd.DataFrame(rows).sort_values(["Sharpe", "CAGR"], ascending=[False, False])
    result_dir = PROJECT_ROOT / "result" / "gold_btc_sp500_overlay" / "dbmf_test"
    result_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(result_dir / "dbmf_weight_grid.csv", index=False)
    pd.DataFrame(curves).to_csv(result_dir / "dbmf_weight_curves.csv")
    pd.DataFrame(exposure_snapshots).to_csv(result_dir / "dbmf_daily_exposures.csv")
    print(summary.head(20).to_string(index=False))
    print(f"\nSaved: {result_dir.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
