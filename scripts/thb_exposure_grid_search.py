from __future__ import annotations

from itertools import product
import json
from pathlib import Path
import sys
from urllib.parse import quote
from urllib.request import urlopen

import numpy as np
import pandas as pd

PROJECT_ROOT = Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from portfolio_engine import calculate_performance_metrics


OUT = Path("result/gold_btc_sp500_overlay/thb_exposure_grid")
OUT.mkdir(parents=True, exist_ok=True)

START_DATE = "2014-09-20"
END_DATE = pd.Timestamp.today().date().isoformat()
TICKERS = ["SPY", "GC=F", "BTC-USD", "^VIX", "USDTHB=X"]
USD_ASSETS = ["SPY", "GC=F", "BTC-USD"]
TRADING_DAYS = 252
RISK_FREE_RATE = 0.03
TARGET_WEIGHTS = pd.Series({"SP500_OVERLAY": 0.70, "GOLD": 0.20, "BTC": 0.10}, dtype=float)
REBALANCE_MONTHS = (1, 4, 7, 10)


def curve_from_returns(returns: pd.Series, initial: float = 10_000.0) -> pd.DataFrame:
    return pd.DataFrame({"PortValue": initial * (1.0 + returns).cumprod()}, index=returns.index)


def metrics(curve: pd.DataFrame) -> dict:
    return calculate_performance_metrics(curve, RISK_FREE_RATE).set_index("Metric")["Value"].to_dict()


def cash_returns_from_mode(fx_returns: pd.Series, cash_mode: str) -> pd.Series:
    if cash_mode == "USD":
        return fx_returns
    if cash_mode == "THB":
        return pd.Series(0.0, index=fx_returns.index, dtype=float)
    raise ValueError(f"Unsupported cash mode: {cash_mode}")


def trend_exposure(local_price: pd.Series, below: float, ma_days: int = 200, min_periods: int = 40) -> pd.Series:
    ma = local_price.rolling(ma_days, min_periods=min_periods).mean()
    exposure = pd.Series(1.0, index=local_price.index, dtype=float)
    exposure.loc[local_price < ma] = below
    exposure.loc[ma.isna()] = 1.0
    return exposure


def apply_exposure(local_returns: pd.Series, exposure: pd.Series, cash_mode: str) -> pd.Series:
    cash_returns = cash_returns_from_mode(fx_returns.reindex(local_returns.index).fillna(0.0), cash_mode)
    lagged_exposure = exposure.shift(1).reindex(local_returns.index).ffill().fillna(1.0)
    return local_returns * lagged_exposure + cash_returns * (1.0 - lagged_exposure)


def build_sp500_exposure(trend_cap: float, warn_cap: float, crash_cap: float) -> pd.Series:
    candidates = pd.concat(
        [
            pd.Series(1.0, index=spy_local.index, dtype=float),
            pd.Series(np.where(spy_local < spy_ma200, trend_cap, 1.0), index=spy_local.index, dtype=float),
            pd.Series(np.where(spy_drawdown <= -0.08, warn_cap, 1.0), index=spy_local.index, dtype=float),
            pd.Series(np.where(spy_drawdown <= -0.15, crash_cap, 1.0), index=spy_local.index, dtype=float),
            pd.Series(np.where(vix_local >= 28.0, warn_cap, 1.0), index=spy_local.index, dtype=float),
            pd.Series(np.where(vix_local >= 35.0, crash_cap, 1.0), index=spy_local.index, dtype=float),
        ],
        axis=1,
    )
    return candidates.min(axis=1)


def simulate_quarterly_rebalance(sleeve_returns: pd.DataFrame, weights: pd.Series) -> tuple[pd.Series, pd.DataFrame]:
    weights = weights.reindex(sleeve_returns.columns).fillna(0.0)
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError("Weights must sum to 1.0")
    month_ends = sleeve_returns.groupby(sleeve_returns.index.to_period("M")).tail(1).index
    rebalance_dates = {dt for dt in month_ends if dt.month in REBALANCE_MONTHS}
    values = weights * 10_000.0
    portfolio_returns = []
    weight_rows = []
    for dt, row in sleeve_returns.iterrows():
        total_before = float(values.sum())
        values = values * (1.0 + row.fillna(0.0))
        total_after = float(values.sum())
        portfolio_returns.append((dt, total_after / total_before - 1.0 if total_before > 0 else 0.0))
        weight_rows.append({"Date": dt, **(values / total_after).to_dict()})
        if dt in rebalance_dates and total_after > 0:
            values = weights * total_after
    return pd.Series(dict(portfolio_returns), name="Portfolio").sort_index(), pd.DataFrame(weight_rows)


def download_yahoo_close(ticker: str, start: str, end: str) -> pd.Series:
    start_ts = int(pd.Timestamp(start).timestamp())
    end_ts = int((pd.Timestamp(end) + pd.Timedelta(days=1)).timestamp())
    encoded = quote(ticker, safe="")
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{encoded}"
        f"?period1={start_ts}&period2={end_ts}&interval=1d&includeAdjustedClose=true"
    )
    with urlopen(url, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    result = payload.get("chart", {}).get("result") or []
    if not result:
        raise RuntimeError(f"No price data downloaded for {ticker}.")
    result0 = result[0]
    timestamps = result0.get("timestamp") or []
    adjclose = (
        result0.get("indicators", {})
        .get("adjclose", [{}])[0]
        .get("adjclose")
    )
    if adjclose is None:
        adjclose = result0.get("indicators", {}).get("quote", [{}])[0].get("close")
    if adjclose is None:
        raise RuntimeError(f"Yahoo response missing close series for {ticker}.")
    series = pd.Series(adjclose, index=pd.to_datetime(timestamps, unit="s"), dtype=float).dropna()
    if series.empty:
        raise RuntimeError(f"Yahoo returned an empty close series for {ticker}.")
    return series.rename(ticker)


close_frames = [download_yahoo_close(ticker, START_DATE, END_DATE) for ticker in TICKERS]
prices = pd.concat(close_frames, axis=1).sort_index().ffill()
prices.index = pd.to_datetime(prices.index)
prices = prices.dropna(subset=["SPY", "GC=F", "BTC-USD", "USDTHB=X"])
fx = prices["USDTHB=X"].dropna()
fx_returns = fx.pct_change(fill_method=None).fillna(0.0)
local_prices = pd.DataFrame(index=prices.index)
for ticker in USD_ASSETS:
    local_prices[ticker] = prices[ticker] * fx.reindex(prices.index).ffill()
base = pd.concat(
    {
        "SPY_LOCAL": local_prices["SPY"],
        "GOLD_LOCAL": local_prices["GC=F"],
        "BTC_LOCAL": local_prices["BTC-USD"],
        "VIX": prices["^VIX"],
        "FX": fx,
    },
    axis=1,
).dropna()
spy_local = base["SPY_LOCAL"]
gold_local = base["GOLD_LOCAL"]
btc_local = base["BTC_LOCAL"]
vix_local = base["VIX"]
fx_returns = base["FX"].pct_change(fill_method=None).fillna(0.0)
spy_local_returns = spy_local.pct_change(fill_method=None).fillna(0.0)
gold_local_returns = gold_local.pct_change(fill_method=None).fillna(0.0)
btc_local_returns = btc_local.pct_change(fill_method=None).fillna(0.0)
spy_ma200 = spy_local.rolling(200, min_periods=40).mean()
spy_drawdown = spy_local / spy_local.cummax() - 1.0

sp_trend_caps = [0.50, 0.65, 0.80]
sp_reduce_pairs = [(0.35, 0.15), (0.50, 0.25), (0.65, 0.35)]
gold_below_values = [0.25, 0.50, 0.75]
btc_below_values = [0.00, 0.25, 0.50]
cash_modes = ["THB", "USD"]

grid_rows = []
best_by_sharpe = None
best_by_cagr_under_20dd = None
best_details: dict[str, pd.DataFrame | pd.Series | dict] = {}

for trend_cap, (warn_cap, crash_cap), gold_below, btc_below, cash_mode in product(
    sp_trend_caps,
    sp_reduce_pairs,
    gold_below_values,
    btc_below_values,
    cash_modes,
):
    sp_exposure = build_sp500_exposure(trend_cap, warn_cap, crash_cap)
    gold_exposure = trend_exposure(gold_local, below=gold_below)
    btc_exposure = trend_exposure(btc_local, below=btc_below)
    sp_returns = apply_exposure(spy_local_returns, sp_exposure, cash_mode)
    gold_returns = apply_exposure(gold_local_returns, gold_exposure, cash_mode)
    btc_returns = apply_exposure(btc_local_returns, btc_exposure, cash_mode)

    sleeve_returns = pd.concat(
        {"SP500_OVERLAY": sp_returns, "GOLD": gold_returns, "BTC": btc_returns},
        axis=1,
    ).dropna()
    if sleeve_returns.empty:
        continue
    portfolio_returns, effective_weights = simulate_quarterly_rebalance(sleeve_returns, TARGET_WEIGHTS)
    portfolio_curve = curve_from_returns(portfolio_returns)
    row = {
        "SP Trend Cap": trend_cap,
        "SP Warn Cap": warn_cap,
        "SP Crash Cap": crash_cap,
        "Gold Below Exposure": gold_below,
        "BTC Below Exposure": btc_below,
        "Cash Mode": cash_mode,
        "Avg SP Exposure": float(sp_exposure.reindex(sleeve_returns.index).mean()),
        "Avg Gold Exposure": float(gold_exposure.reindex(sleeve_returns.index).mean()),
        "Avg BTC Exposure": float(btc_exposure.reindex(sleeve_returns.index).mean()),
        **metrics(portfolio_curve),
    }
    grid_rows.append(row)

    if best_by_sharpe is None or row["Sharpe"] > best_by_sharpe["Sharpe"]:
        best_by_sharpe = row.copy()
        best_details["best_sharpe_curve"] = portfolio_curve
        best_details["best_sharpe_exposure"] = pd.concat(
            {
                "SP500_OVERLAY": sp_exposure.reindex(sleeve_returns.index),
                "GOLD": gold_exposure.reindex(sleeve_returns.index),
                "BTC": btc_exposure.reindex(sleeve_returns.index),
            },
            axis=1,
        )
        best_details["best_sharpe_weights"] = effective_weights
    if row["Max Drawdown"] >= -0.20 and (
        best_by_cagr_under_20dd is None or row["CAGR"] > best_by_cagr_under_20dd["CAGR"]
    ):
        best_by_cagr_under_20dd = row.copy()

grid = pd.DataFrame(grid_rows).sort_values(["Sharpe", "CAGR"], ascending=[False, False])
grid.to_csv(OUT / "thb_exposure_grid_results.csv", index=False)

summary_rows = []
if best_by_sharpe is not None:
    summary_rows.append({"Selection": "Best Sharpe", **best_by_sharpe})
if best_by_cagr_under_20dd is not None:
    summary_rows.append({"Selection": "Best CAGR with DD <= 20%", **best_by_cagr_under_20dd})
summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT / "thb_exposure_grid_summary.csv", index=False)

if "best_sharpe_exposure" in best_details:
    best_details["best_sharpe_exposure"].to_csv(OUT / "best_sharpe_exposure_history.csv")
if "best_sharpe_weights" in best_details:
    best_details["best_sharpe_weights"].to_csv(OUT / "best_sharpe_effective_weights.csv", index=False)
if "best_sharpe_curve" in best_details:
    best_details["best_sharpe_curve"].to_csv(OUT / "best_sharpe_equity_curve.csv")

summary_lines = [
    "# THB Exposure Grid Search",
    "",
    "Base portfolio: 70% S&P 500 overlay / 20% Gold overlay / 10% BTC overlay.",
    "All USD assets are converted to THB using USDTHB before returns and overlay signals are calculated.",
    "When exposure is reduced, the residual cash can stay in THB or in USD for each sleeve.",
    "",
    f"Sample: {grid.shape[0]} grid combinations from {sleeve_returns.index.min().date()} to {sleeve_returns.index.max().date()}.",
    "",
]
if best_by_sharpe is not None:
    summary_lines.extend(
        [
            "## Best Sharpe",
            "",
            f"- SP trend cap: {best_by_sharpe['SP Trend Cap']:.2f}",
            f"- SP warn/crash cap: {best_by_sharpe['SP Warn Cap']:.2f} / {best_by_sharpe['SP Crash Cap']:.2f}",
            f"- Gold below MA200 exposure: {best_by_sharpe['Gold Below Exposure']:.2f}",
            f"- BTC below MA200 exposure: {best_by_sharpe['BTC Below Exposure']:.2f}",
            f"- Reduced-exposure cash mode: {best_by_sharpe['Cash Mode']}",
            f"- CAGR: {best_by_sharpe['CAGR']:.2%}",
            f"- Sharpe: {best_by_sharpe['Sharpe']:.3f}",
            f"- Max Drawdown: {best_by_sharpe['Max Drawdown']:.2%}",
            "",
        ]
    )
if best_by_cagr_under_20dd is not None:
    summary_lines.extend(
        [
            "## Best CAGR with DD <= 20%",
            "",
            f"- SP trend cap: {best_by_cagr_under_20dd['SP Trend Cap']:.2f}",
            f"- SP warn/crash cap: {best_by_cagr_under_20dd['SP Warn Cap']:.2f} / {best_by_cagr_under_20dd['SP Crash Cap']:.2f}",
            f"- Gold below MA200 exposure: {best_by_cagr_under_20dd['Gold Below Exposure']:.2f}",
            f"- BTC below MA200 exposure: {best_by_cagr_under_20dd['BTC Below Exposure']:.2f}",
            f"- Reduced-exposure cash mode: {best_by_cagr_under_20dd['Cash Mode']}",
            f"- CAGR: {best_by_cagr_under_20dd['CAGR']:.2%}",
            f"- Sharpe: {best_by_cagr_under_20dd['Sharpe']:.3f}",
            f"- Max Drawdown: {best_by_cagr_under_20dd['Max Drawdown']:.2%}",
            "",
        ]
    )
(OUT / "THB_EXPOSURE_GRID_BEST.md").write_text("\n".join(summary_lines), encoding="utf-8")

print(summary.to_string(index=False))
print("wrote", OUT)
