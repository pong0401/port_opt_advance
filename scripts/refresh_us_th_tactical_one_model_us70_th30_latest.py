from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize


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
    load_cached_market_data,
    select_point_in_time_universe,
)
from refresh_us_th_tactical_final_best_latest import (  # noqa: E402
    FEATURE_FLAGS,
    LOOKBACK_DAYS,
    START_DATE,
    STOCK_CAP,
    TH_ASSETS,
    US_ASSETS,
    _active_members,
    _all_available_members,
    _close_trend_exposure,
    _gold_crash_protection_exposure,
    _latest_common_close,
    _load_overlay,
    _source_close_date,
    _th_tactical_weight,
)
from share_class_utils import drop_duplicate_share_classes_available  # noqa: E402


STRATEGY = "One-model US cap 70% / TH cap 30% + daily exposure"
CASE = "US cap 70% / TH cap 30%"
RESULT_PREFIX = "us_th_tactical_one_model_us70_th30"
US_GROUP_CAP = 0.70
TH_GROUP_CAP = 0.30
GOLD_CAP = 0.30
BTC_CAP = 0.10
RISK_AVERSION = 8.0
CONCENTRATION_PENALTY = 0.02


def _select_stock_group(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    candidates: list[str],
    n_assets: int,
    as_of: pd.Timestamp,
) -> tuple[list[str], pd.Timestamp]:
    stock_dates = prices.dropna(how="all").index
    stock_dates = stock_dates[stock_dates <= as_of]
    if stock_dates.empty:
        return [], as_of
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
    return selected, stock_as_of


def _optimize_one_model(
    train_returns: pd.DataFrame,
    benchmark: pd.Series,
    vol_proxy: pd.Series,
    prices: pd.DataFrame,
    us_assets: set[str],
    th_assets: set[str],
) -> pd.Series:
    selected = train_returns.dropna(axis=1, thresh=max(int(0.75 * len(train_returns)), 60)).columns.tolist()
    selected = drop_duplicate_share_classes_available(selected, selected)
    train_returns = train_returns.reindex(columns=selected)
    if train_returns.empty or not selected:
        raise RuntimeError("No selected assets survived the combined one-model training window.")

    features = compute_feature_table(
        train_returns,
        benchmark.pct_change(fill_method=None).reindex(train_returns.index),
        vol_proxy.pct_change(fill_method=None).reindex(train_returns.index),
        prices.reindex(train_returns.index)[selected],
        include_momentum_features=True,
        feature_flags=FEATURE_FLAGS,
    )
    momentum = build_momentum_signal(features, mode="mom_63").reindex(selected)
    mu = momentum.fillna(momentum.median() if momentum.notna().any() else 0.0).to_numpy(dtype=float)
    if len(mu):
        mu = np.clip(mu, np.nanpercentile(mu, 10), np.nanpercentile(mu, 90))

    cov = train_returns.cov().reindex(index=selected, columns=selected).fillna(0.0)
    cov_matrix = cov.to_numpy(dtype=float)
    caps = pd.Series(STOCK_CAP, index=selected, dtype=float)
    caps.loc[[asset for asset in selected if asset == "GC=F"]] = GOLD_CAP
    caps.loc[[asset for asset in selected if asset == "BTC-USD"]] = BTC_CAP
    if float(caps.sum()) < 1.0 - 1e-12:
        raise RuntimeError("One-model caps are infeasible; caps sum below 100%.")

    x0 = caps / caps.sum()
    bounds = [(0.0, float(caps.loc[asset])) for asset in selected]
    constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1.0}]
    us_idx = [i for i, asset in enumerate(selected) if asset in us_assets]
    th_idx = [i for i, asset in enumerate(selected) if asset in th_assets]
    if us_idx:
        constraints.append({"type": "ineq", "fun": lambda x, idx=us_idx: US_GROUP_CAP - float(np.sum(x[idx]))})
    if th_idx:
        constraints.append({"type": "ineq", "fun": lambda x, idx=th_idx: TH_GROUP_CAP - float(np.sum(x[idx]))})

    def objective(x: np.ndarray) -> float:
        variance = float(x @ cov_matrix @ x)
        expected = float(mu @ x)
        concentration = float(np.sum(np.square(x)))
        return 0.5 * RISK_AVERSION * variance - expected + CONCENTRATION_PENALTY * concentration

    result = minimize(objective, x0=x0.to_numpy(dtype=float), bounds=bounds, constraints=constraints, method="SLSQP")
    if not result.success:
        weights = x0.copy()
    else:
        weights = pd.Series(result.x, index=selected).clip(lower=0.0)
    return (weights / weights.sum()).sort_values(ascending=False)


def _sleeve(asset: str, us_assets: set[str], th_assets: set[str]) -> str:
    if asset in us_assets:
        return "US Equity"
    if asset in th_assets:
        return "TH Equity"
    if asset == "GC=F":
        return "Gold"
    if asset == "BTC-USD":
        return "BTC"
    if asset == "Cash / Reduced Exposure":
        return "Cash / Reduced Exposure"
    return "Other"


def _write_outputs(security: pd.DataFrame, sleeve: pd.DataFrame, meta: pd.DataFrame) -> None:
    for output_dir in [ROOT / "result", ROOT / "data" / "precomputed"]:
        output_dir.mkdir(parents=True, exist_ok=True)
        security.to_csv(output_dir / f"{RESULT_PREFIX}_latest_effective_weights_thb.csv", index=False)
        sleeve.to_csv(output_dir / f"{RESULT_PREFIX}_latest_sleeve_weights_thb.csv", index=False)
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
    cached = load_cached_market_data(paths, tickers=list(dict.fromkeys(us_active + th_active)))
    fx = overlay["USDTHB=X"].reindex(cached["prices"].index).ffill()

    us_prices = cached["prices"].reindex(columns=us_active).mul(fx, axis=0).loc[START_DATE:as_of].ffill()
    th_prices = cached["prices"].reindex(columns=th_active).loc[START_DATE:as_of].ffill()
    us_volumes = cached["volumes"].reindex(us_prices.index).reindex(columns=us_active).fillna(0.0)
    th_volumes = cached["volumes"].reindex(th_prices.index).reindex(columns=th_active).fillna(0.0)

    us_selected, us_internal_date = _select_stock_group(us_prices, us_volumes, us_active, US_ASSETS, as_of)
    th_signal_weight = _th_tactical_weight(overlay, as_of)
    th_selected: list[str] = []
    th_internal_date = as_of
    if th_signal_weight > 1e-12:
        th_selected, th_internal_date = _select_stock_group(th_prices, th_volumes, th_active, TH_ASSETS, as_of)

    combined_index = us_prices.index.union(th_prices.index).union(overlay.index).sort_values()
    prices = pd.DataFrame(index=combined_index)
    for asset in us_selected:
        prices[asset] = us_prices[asset].reindex(combined_index)
    for asset in th_selected:
        prices[asset] = th_prices[asset].reindex(combined_index)
    prices["GC=F"] = (overlay["GC=F"] * overlay["USDTHB=X"]).reindex(combined_index)
    prices["BTC-USD"] = (overlay["BTC-USD"] * overlay["USDTHB=X"]).reindex(combined_index)
    prices = prices.loc[START_DATE:as_of].ffill()

    train_dates = prices.dropna(how="all").index
    loc = train_dates.get_loc(train_dates.max())
    train_index = train_dates[max(0, loc - LOOKBACK_DAYS + 1) : loc + 1]
    returns = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    train_returns = returns.reindex(train_index)
    benchmark = (overlay["SPY"] * overlay["USDTHB=X"]).reindex(train_index).ffill().rename("benchmark")
    vol_proxy = overlay["^VIX"].reindex(train_index).ffill().rename("vol_proxy")

    raw_weights = _optimize_one_model(
        train_returns,
        benchmark,
        vol_proxy,
        prices,
        set(us_selected),
        set(th_selected),
    )
    sleeve_map = raw_weights.index.to_series().map(lambda asset: _sleeve(str(asset), set(us_selected), set(th_selected)))
    exposures = pd.Series(
        {
            "US Equity": float(_close_trend_exposure(overlay["SPY"], 300, 0.50).loc[:as_of].iloc[-1]),
            "TH Equity": float(_close_trend_exposure(overlay["^SET.BK"], 200, 0.00).loc[:as_of].iloc[-1]),
            "Gold": float(_gold_crash_protection_exposure(overlay["GC=F"]).loc[:as_of].iloc[-1]),
            "BTC": float(_close_trend_exposure(overlay["BTC-USD"], 50, 0.00).loc[:as_of].iloc[-1]),
        },
        dtype=float,
    )
    asset_exposure = sleeve_map.map(exposures).fillna(1.0).astype(float)
    effective = raw_weights.mul(asset_exposure)
    cash_weight = max(0.0, 1.0 - float(effective.sum()))

    security = pd.DataFrame(
        {
            "Asset": raw_weights.index,
            "Sleeve": sleeve_map.to_numpy(),
            "Effective Weight": effective.to_numpy(dtype=float),
            "Raw Optimizer Weight": raw_weights.to_numpy(dtype=float),
            "Daily Exposure": asset_exposure.to_numpy(dtype=float),
        }
    )
    if cash_weight > 1e-12:
        security = pd.concat(
            [
                security,
                pd.DataFrame(
                    [
                        {
                            "Asset": "Cash / Reduced Exposure",
                            "Sleeve": "Cash / Reduced Exposure",
                            "Effective Weight": cash_weight,
                            "Raw Optimizer Weight": 0.0,
                            "Daily Exposure": 1.0,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    security["Effective Weight %"] = security["Effective Weight"].mul(100.0)
    security["Date"] = as_of.date().isoformat()
    security["Strategy"] = STRATEGY
    security["Case"] = CASE
    security["Last Exposure Date"] = as_of.date().isoformat()
    security["Signal Source Close Date"] = _source_close_date(overlay.index, as_of)
    security = security.loc[security["Effective Weight"].abs().gt(1e-12)].sort_values("Effective Weight", ascending=False)

    sleeve = (
        security.groupby("Sleeve", as_index=False)
        .agg(
            **{
                "Effective Weight": ("Effective Weight", "sum"),
                "Raw Optimizer Weight": ("Raw Optimizer Weight", "sum"),
                "Daily Exposure": ("Daily Exposure", "max"),
            }
        )
        .sort_values("Effective Weight", ascending=False)
    )
    sleeve["Effective Weight %"] = sleeve["Effective Weight"].mul(100.0)
    sleeve["Date"] = as_of.date().isoformat()
    sleeve["Strategy"] = STRATEGY
    sleeve["Case"] = CASE

    meta = pd.DataFrame(
        [
            {
                "Date": as_of.date().isoformat(),
                "Strategy": STRATEGY,
                "Case": CASE,
                "Model": "one combined mean-covariance optimizer",
                "Objective": "mean_variance + mom_63 + concentration penalty",
                "Universe": (
                    f"PIT S&P 500 top{US_ASSETS}, PIT SET100 top{TH_ASSETS} "
                    "when TH tactical signal is active, Gold, BTC"
                ),
                "Caps": (
                    f"stock {STOCK_CAP:.0%}; US group {US_GROUP_CAP:.0%}; "
                    f"TH group {TH_GROUP_CAP:.0%}; Gold {GOLD_CAP:.0%}; BTC {BTC_CAP:.0%}"
                ),
                "Daily Exposure": "US SPY MA300 below50%; TH SET MA200 below0%; Gold crash protection; BTC MA50 below0%; reduced exposure to cash",
                "TH Tactical Rule": "monthly SET-vs-SPY THB relative-return binary lb1 entry0 exit0 hold0 confirm1",
                "TH Tactical Active Weight": th_signal_weight,
                "Selected US Assets": len(us_selected),
                "Selected TH Assets": len(th_selected),
                "US Internal Weight Date": us_internal_date.date().isoformat(),
                "TH Internal Weight Date": th_internal_date.date().isoformat(),
                "Train Start": pd.Timestamp(train_index.min()).date().isoformat(),
                "Train End": pd.Timestamp(train_index.max()).date().isoformat(),
                "US Daily Exposure": float(exposures["US Equity"]),
                "TH Daily Exposure": float(exposures["TH Equity"]),
                "Gold Daily Exposure": float(exposures["Gold"]),
                "BTC Daily Exposure": float(exposures["BTC"]),
                "Latest Weight Source": "Standalone refresh from this repo's current data/cache; no static latest-weight file is read from dynamic_port_opt.",
            }
        ]
    )

    _write_outputs(security, sleeve, meta)
    print(meta.to_string(index=False))
    print(sleeve.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(security[["Asset", "Sleeve", "Effective Weight", "Raw Optimizer Weight", "Daily Exposure"]].head(50).to_string(index=False, float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()
