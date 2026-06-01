from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize

PROJECT_ROOT = Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from portfolio_engine import calculate_performance_metrics
from scripts.top100_daily_buffer_test import OVERLAY_PROFILES, daily_overlay_cap

OUT = Path("result/gold_btc_sp500_overlay/long_history_test")
OUT.mkdir(parents=True, exist_ok=True)
YF_CACHE = Path("C:/tmp/yfinance_long_history")
YF_CACHE.mkdir(parents=True, exist_ok=True)
yf.set_tz_cache_location(str(YF_CACHE))
yf.cache.set_cache_location(str(YF_CACHE))

START_DATE = "2014-09-20"
END_DATE = pd.Timestamp.today().date().isoformat()
TICKERS = ["SPY", "GLD", "BTC-USD", "DBC", "TLT", "AGG", "^VIX"]
TRADING_DAYS = 252
RISK_FREE_RATE = 0.03


def curve_from_returns(returns: pd.Series, initial: float = 10_000.0) -> pd.DataFrame:
    return pd.DataFrame({"PortValue": initial * (1.0 + returns).cumprod()}, index=returns.index)


def metrics(curve: pd.DataFrame) -> dict:
    return calculate_performance_metrics(curve, RISK_FREE_RATE).set_index("Metric")["Value"].to_dict()


def trend_overlay_returns(
    price: pd.Series,
    below: float = 0.5,
    above: float = 1.0,
    ma_days: int = 200,
    min_periods: int = 40,
) -> tuple[pd.Series, pd.Series]:
    price = price.dropna().sort_index()
    returns = price.pct_change(fill_method=None).fillna(0.0)
    ma = price.rolling(ma_days, min_periods=min_periods).mean()
    exposure = pd.Series(above, index=price.index, dtype=float)
    exposure.loc[price < ma] = below
    exposure.loc[ma.isna()] = above
    return returns * exposure, exposure


raw = yf.download(TICKERS, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False, group_by="column")
if raw.empty:
    raise RuntimeError("No price data downloaded. Check network/yfinance availability.")
prices = raw["Close"].reindex(columns=TICKERS).ffill()
prices.index = pd.to_datetime(prices.index)
prices = prices.dropna(subset=["SPY", "GLD", "BTC-USD"])
vix = prices["^VIX"].dropna()

profile = OVERLAY_PROFILES["Daily trend + drawdown + VIX"]
spy = prices["SPY"].dropna()
spy_returns = spy.pct_change(fill_method=None).fillna(0.0)
sp_overlay = []
for dt, ret in spy_returns.items():
    exposure = daily_overlay_cap(dt, spy, vix, profile)
    sp_overlay.append((dt, float(ret) * exposure))
sp_overlay = pd.Series(dict(sp_overlay), name="SP500_OVERLAY").sort_index()

btc_overlay, _ = trend_overlay_returns(prices["BTC-USD"].rename("BTC"), below=0.0)
gold_overlay, _ = trend_overlay_returns(prices["GLD"].rename("GOLD"), below=0.5)
commodity_overlay, _ = trend_overlay_returns(prices["DBC"].rename("COMMODITY"), below=0.5)
tlt_overlay, _ = trend_overlay_returns(prices["TLT"].rename("TLT"), below=0.5)
agg_overlay, _ = trend_overlay_returns(prices["AGG"].rename("AGG"), below=0.5)

returns = pd.concat(
    {
        "SP500_OVERLAY": sp_overlay,
        "GOLD": gold_overlay,
        "BTC": btc_overlay,
        "COMMODITY": commodity_overlay,
        "TLT": tlt_overlay,
        "AGG": agg_overlay,
    },
    axis=1,
).dropna()

assets = list(returns.columns)


def bounds_for(asset: str) -> tuple[float, float]:
    return {
        "SP500_OVERLAY": (0.50, 0.90),
        "GOLD": (0.00, 0.30),
        "BTC": (0.00, 0.10),
        "COMMODITY": (0.00, 0.15),
        "TLT": (0.00, 0.25),
        "AGG": (0.00, 0.25),
    }.get(asset, (0.00, 0.25))


bounds = [bounds_for(asset) for asset in assets]


def train_stats(train: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    mu = train.mean() * TRADING_DAYS
    cov = train.cov() * TRADING_DAYS
    cov = cov * 0.75 + np.diag(np.diag(cov)) * 0.25
    return mu, cov


def optimize_max_sharpe(train: pd.DataFrame) -> pd.Series:
    mu, cov = train_stats(train)
    initial = np.array([(lo + hi) / 2 for lo, hi in bounds], dtype=float)
    initial = initial / initial.sum()
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    def objective(w: np.ndarray) -> float:
        vol = float(np.sqrt(w @ cov.values @ w))
        ret = float(w @ mu.values)
        return -((ret - RISK_FREE_RATE) / vol) if vol > 0 else 1e6

    result = minimize(objective, initial, method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": 2000})
    return pd.Series(result.x if result.success else initial, index=assets)


def optimize_min_vol(train: pd.DataFrame) -> pd.Series:
    _, cov = train_stats(train)
    initial = np.array([(lo + hi) / 2 for lo, hi in bounds], dtype=float)
    initial = initial / initial.sum()
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    def objective(w: np.ndarray) -> float:
        return float(np.sqrt(w @ cov.values @ w))

    result = minimize(objective, initial, method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": 2000})
    return pd.Series(result.x if result.success else initial, index=assets)


def optimize_risk_parity(train: pd.DataFrame) -> pd.Series:
    _, cov = train_stats(train)
    cov_values = cov.values
    initial = np.array([(lo + hi) / 2 for lo, hi in bounds], dtype=float)
    initial = initial / initial.sum()
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    def objective(w: np.ndarray) -> float:
        var = float(w @ cov_values @ w)
        if var <= 0:
            return 1e6
        marginal = cov_values @ w
        risk_contrib = w * marginal / var
        target = np.ones_like(w) / len(w)
        return float(((risk_contrib - target) ** 2).sum())

    result = minimize(objective, initial, method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": 2000})
    return pd.Series(result.x if result.success else initial, index=assets)


def normalize(template: dict[str, float]) -> pd.Series:
    weights = pd.Series(0.0, index=assets, dtype=float)
    for asset, weight in template.items():
        if asset in weights.index:
            weights.loc[asset] = weight
    return weights / weights.sum()


templates = {
    "Fixed 80/10/10": normalize({"SP500_OVERLAY": 0.80, "GOLD": 0.10, "BTC": 0.10}),
    "Fixed 70/20/10": normalize({"SP500_OVERLAY": 0.70, "GOLD": 0.20, "BTC": 0.10}),
    "Fixed 65/20/5/5/5": normalize({"SP500_OVERLAY": 0.65, "GOLD": 0.20, "BTC": 0.05, "COMMODITY": 0.05, "TLT": 0.05}),
    "Fixed 65/25/5/5": normalize({"SP500_OVERLAY": 0.65, "GOLD": 0.25, "BTC": 0.05, "TLT": 0.05}),
}


def adaptive_70_risk_parity(train: pd.DataFrame) -> pd.Series:
    trailing_sp = train["SP500_OVERLAY"].dropna().tail(TRADING_DAYS)
    if trailing_sp.empty:
        return templates["Fixed 70/20/10"]
    trailing_curve = (1.0 + trailing_sp).cumprod()
    trailing_drawdown = trailing_curve / trailing_curve.cummax() - 1.0
    trailing_vol = trailing_sp.std() * np.sqrt(TRADING_DAYS)
    stress = (trailing_drawdown.min() <= -0.08) or (trailing_vol >= 0.16)
    if stress:
        return optimize_risk_parity(train)
    return templates["Fixed 70/20/10"]


def classify_regime(train: pd.DataFrame) -> str:
    recent = train.tail(TRADING_DAYS)
    if len(recent) < 126:
        return "normal"

    sp = recent["SP500_OVERLAY"].dropna()
    gold = recent["GOLD"].dropna()
    btc = recent["BTC"].dropna()
    commodity = recent["COMMODITY"].dropna()

    sp_curve = (1.0 + sp).cumprod()
    sp_drawdown = sp_curve / sp_curve.cummax() - 1.0
    sp_return = (1.0 + sp).prod() - 1.0
    sp_vol = sp.std() * np.sqrt(TRADING_DAYS)
    gold_return = (1.0 + gold).prod() - 1.0
    btc_return = (1.0 + btc).prod() - 1.0
    commodity_return = (1.0 + commodity).prod() - 1.0

    inflation_stress = (
        sp_return < 0.04
        and (gold_return > sp_return or commodity_return > sp_return)
        and (gold_return > 0.02 or commodity_return > 0.02)
    )
    equity_stress = sp_drawdown.min() <= -0.10 or sp_vol >= 0.17
    risk_on = sp_return > 0.12 and btc_return > sp_return and sp_drawdown.min() > -0.08

    if inflation_stress:
        return "inflation_stress"
    if equity_stress:
        return "equity_stress"
    if risk_on:
        return "risk_on"
    return "normal"


def regime_template(regime: str) -> pd.Series:
    templates_by_regime = {
        "normal": {"SP500_OVERLAY": 0.70, "GOLD": 0.20, "BTC": 0.10},
        "risk_on": {"SP500_OVERLAY": 0.80, "GOLD": 0.10, "BTC": 0.10},
        "equity_stress": {"SP500_OVERLAY": 0.60, "GOLD": 0.25, "BTC": 0.05, "COMMODITY": 0.05, "TLT": 0.05},
        "inflation_stress": {"SP500_OVERLAY": 0.50, "GOLD": 0.30, "BTC": 0.05, "COMMODITY": 0.15},
    }
    return normalize(templates_by_regime.get(regime, templates_by_regime["normal"]))


def walk_forward(method: str, train_years: int = 3, hold_months: int = 3) -> tuple[pd.Series, pd.DataFrame, dict, pd.DataFrame]:
    month_ends = returns.groupby(returns.index.to_period("M")).tail(1).index.sort_values()
    rebalance_dates = list(month_ends[train_years * 12 :: hold_months])
    chunks = []
    rows = []
    previous = pd.Series(0.0, index=assets, dtype=float)
    active_regime = "normal"
    pending_regime = None
    pending_count = 0

    for idx, rebalance_date in enumerate(rebalance_dates):
        next_date = rebalance_dates[idx + 1] if idx + 1 < len(rebalance_dates) else returns.index[-1]
        train = returns.loc[
            (returns.index > rebalance_date - pd.DateOffset(years=train_years))
            & (returns.index <= rebalance_date),
            assets,
        ].dropna()
        if len(train) < 252 * 2:
            continue
        if method in templates:
            weights = templates[method]
        elif method == "Max Sharpe":
            weights = optimize_max_sharpe(train)
        elif method == "Min Vol":
            weights = optimize_min_vol(train)
        elif method == "Risk Parity":
            weights = optimize_risk_parity(train)
        elif method == "Adaptive 70/RP":
            weights = adaptive_70_risk_parity(train)
        elif method == "2Q Regime Rule":
            detected_regime = classify_regime(train)
            if detected_regime == active_regime:
                pending_regime = None
                pending_count = 0
            elif detected_regime == pending_regime:
                pending_count += 1
            else:
                pending_regime = detected_regime
                pending_count = 1
            if pending_regime is not None and pending_count >= 2:
                active_regime = pending_regime
                pending_regime = None
                pending_count = 0
            weights = regime_template(active_regime)
        else:
            raise ValueError(method)
        weights = weights / weights.sum()
        row = {"Date": rebalance_date, "Method": method, "Turnover": float((weights - previous).abs().sum()), **weights.to_dict()}
        if method == "2Q Regime Rule":
            row["Detected Regime"] = detected_regime
            row["Active Regime"] = active_regime
        rows.append(row)
        previous = weights

        test = returns.loc[(returns.index > rebalance_date) & (returns.index <= next_date), assets].dropna()
        if not test.empty:
            chunks.append(test @ weights)

    series = pd.concat(chunks).sort_index() if chunks else pd.Series(dtype=float)
    curve = curve_from_returns(series)
    return series, curve, metrics(curve), pd.DataFrame(rows)


methods = list(templates) + ["Max Sharpe", "Min Vol", "Risk Parity", "Adaptive 70/RP", "2Q Regime Rule"]
results = {method: walk_forward(method) for method in methods}
summary = pd.DataFrame(
    [
        {
            "Method": method,
            **result[2],
            "Avg Turnover": result[3]["Turnover"].mean() if not result[3].empty else np.nan,
            "Rebalances": len(result[3]),
        }
        for method, result in results.items()
    ]
).sort_values(["Sharpe", "CAGR"], ascending=[False, False])

summary.to_csv(OUT / "long_history_walk_forward_summary.csv", index=False)
pd.concat({method: result[1]["PortValue"] for method, result in results.items()}, axis=1).ffill().to_csv(
    OUT / "long_history_walk_forward_curves.csv"
)
for method, result in results.items():
    safe_method = method.lower().replace(" ", "_").replace("/", "_")
    result[3].to_csv(OUT / f"long_history_weights_{safe_method}.csv", index=False)

curves = pd.concat({method: result[1]["PortValue"] for method, result in results.items()}, axis=1).ffill()
strategy_returns = curves.pct_change(fill_method=None).fillna(0.0)

regimes = [
    ("WF start / late cycle", "2017-09-01", "2019-12-31"),
    ("Covid + liquidity boom", "2020-01-01", "2021-12-31"),
    ("Inflation / rate shock", "2022-01-01", "2022-12-31"),
    ("AI / high-rate recovery", "2023-01-01", END_DATE),
]

regime_rows = []
for regime, start, end in regimes:
    for method in methods:
        period_returns = strategy_returns.loc[start:end, method].dropna()
        if period_returns.empty:
            continue
        period_curve = curve_from_returns(period_returns)
        regime_rows.append({"Regime": regime, "Method": method, **metrics(period_curve)})

regime_summary = pd.DataFrame(regime_rows)
regime_summary.to_csv(OUT / "regime_performance_by_method.csv", index=False)

weight_rows = []
for method, result in results.items():
    weights = result[3].copy()
    if weights.empty:
        continue
    weights["Date"] = pd.to_datetime(weights["Date"])
    for regime, start, end in regimes:
        subset = weights[(weights["Date"] >= pd.Timestamp(start)) & (weights["Date"] <= pd.Timestamp(end))]
        if subset.empty:
            continue
        row = {"Regime": regime, "Method": method, "Rebalances": len(subset), "Avg Turnover": subset["Turnover"].mean()}
        for asset in assets:
            row[asset] = subset[asset].mean()
        weight_rows.append(row)

regime_weights = pd.DataFrame(weight_rows)
regime_weights.to_csv(OUT / "regime_average_weights.csv", index=False)

grid_rows = []
sp_values = np.arange(0.50, 0.91, 0.05)
gold_values = np.arange(0.00, 0.31, 0.05)
btc_values = np.arange(0.00, 0.11, 0.05)
commodity_values = np.arange(0.00, 0.16, 0.05)
tlt_values = np.arange(0.00, 0.26, 0.05)

for regime, start, end in regimes:
    period = returns.loc[start:end, assets].dropna()
    if period.empty:
        continue
    best_by_sharpe = None
    best_by_cagr = None
    for sp_weight in sp_values:
        for gold_weight in gold_values:
            for btc_weight in btc_values:
                for commodity_weight in commodity_values:
                    for tlt_weight in tlt_values:
                        weights = pd.Series(
                            {
                                "SP500_OVERLAY": sp_weight,
                                "GOLD": gold_weight,
                                "BTC": btc_weight,
                                "COMMODITY": commodity_weight,
                                "TLT": tlt_weight,
                                "AGG": 0.0,
                            },
                            dtype=float,
                        )
                        total = weights.sum()
                        if abs(total - 1.0) > 1e-9:
                            continue
                        period_curve = curve_from_returns(period @ weights[assets])
                        row = {"Regime": regime, **weights.to_dict(), **metrics(period_curve)}
                        if best_by_sharpe is None or row["Sharpe"] > best_by_sharpe["Sharpe"]:
                            best_by_sharpe = {**row, "Objective": "Best Sharpe"}
                        if best_by_cagr is None or row["CAGR"] > best_by_cagr["CAGR"]:
                            best_by_cagr = {**row, "Objective": "Best CAGR"}
    grid_rows.extend([best_by_sharpe, best_by_cagr])

hindsight_weights = pd.DataFrame(grid_rows)
hindsight_weights.to_csv(OUT / "hindsight_best_weights_by_regime.csv", index=False)

print("sample", returns.index.min(), returns.index.max(), len(returns), len(returns) / 252)
print(summary[["Method", "CAGR", "Annual Volatility", "Sharpe", "Sortino", "Max Drawdown", "Hit Rate", "Avg Turnover", "Rebalances"]].to_string(index=False))
print()
print("regime average weights")
print(regime_weights[["Regime", "Method", "SP500_OVERLAY", "GOLD", "BTC", "COMMODITY", "TLT", "AGG", "Avg Turnover"]].to_string(index=False))
print()
print("hindsight best weights by regime")
print(hindsight_weights[["Regime", "Objective", "SP500_OVERLAY", "GOLD", "BTC", "COMMODITY", "TLT", "CAGR", "Annual Volatility", "Sharpe", "Max Drawdown"]].to_string(index=False))
print("wrote", OUT)
