from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dynamic_factor_copula import (  # noqa: E402
    apply_daily_exposure_overlay,
    build_factor_covariance,
    build_momentum_signal,
    compare_rebalanced_portfolio,
    compare_trend_exposure,
    compute_feature_table,
    compute_market_stress_signal,
    compute_port_opt_style_metrics,
    curve_from_returns,
    default_paths,
    initialize_static_clusters,
    load_cached_market_data,
    load_overlay_compare_prices,
    load_sp500_membership_intervals,
    load_set100_membership_intervals,
    monthly_rebalance_dates,
    optimize_portfolio,
    optimize_risk_parity,
    run_dynamic_hmm,
)


START_DATE = "2016-01-01"
END_DATE = "2026-04-29"
LOOKBACK_DAYS = 504
N_CLUSTERS = 4
MAX_WEIGHT = 0.08
FEATURE_FLAGS = {"resid_vol": False, "drawdown": False, "downside_beta": False}
OBJECTIVE_MODES = [
    "mean_variance",
    "max_sharpe_mom",
    "min_vol_mom_tilt",
    "risk_parity_mom_tilt",
]
BEST_OBJECTIVE = "min_vol_mom_tilt"
BEST_ASSET_SWEEP_CASE = {"us_assets": 30, "th_assets": 30, "max_weight": 0.06, "label": "US30/TH30/max6"}
DIME_STYLE_COMMISSION_BPS = 15.0
SLIPPAGE_BPS = 2.0
ASSET_MAX_WEIGHT_SWEEP = [
    {"us_assets": 30, "th_assets": 30, "max_weight": 0.06},
    {"us_assets": 30, "th_assets": 30, "max_weight": 0.08},
    {"us_assets": 30, "th_assets": 30, "max_weight": 0.10},
    {"us_assets": 30, "th_assets": 40, "max_weight": 0.06},
    {"us_assets": 30, "th_assets": 40, "max_weight": 0.08},
    {"us_assets": 30, "th_assets": 40, "max_weight": 0.10},
    {"us_assets": 30, "th_assets": 50, "max_weight": 0.06},
    {"us_assets": 30, "th_assets": 50, "max_weight": 0.08},
    {"us_assets": 30, "th_assets": 50, "max_weight": 0.10},
]


def _read_tickers(path: Path) -> list[str]:
    return pd.read_csv(path)["ticker"].dropna().astype(str).tolist()


def _rank_liquid_tickers(tickers: list[str], top_n: int) -> list[str]:
    cached = load_cached_market_data(default_paths(ROOT), tickers=tickers)
    prices = cached["prices"].loc[START_DATE:END_DATE].reindex(columns=tickers).ffill()
    volumes = cached["volumes"].loc[START_DATE:END_DATE].reindex(columns=tickers).fillna(0.0)
    liquidity = (prices * volumes).median().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    availability = prices.notna().mean().fillna(0.0)
    ranked = (
        pd.DataFrame({"liquidity": liquidity, "availability": availability})
        .query("availability >= 0.75")
        .sort_values(["liquidity", "availability"], ascending=False)
    )
    return ranked.head(top_n).index.tolist()


def _build_ranked_us_th_universe(us_assets: int, th_assets: int) -> tuple[list[str], list[str]]:
    paths = default_paths(ROOT)
    cached = load_cached_market_data(paths)
    source_cols = set(cached["prices"].columns)
    extra_cols = set(pd.read_parquet(paths.local_cache_root / "extra_prices.parquet").columns)
    latest_date = pd.Timestamp(END_DATE)
    sp_intervals = load_sp500_membership_intervals(paths)
    sp_active = sp_intervals.loc[
        (sp_intervals["start_date"] <= latest_date)
        & (sp_intervals["end_date"].isna() | (sp_intervals["end_date"] >= latest_date))
    ]
    th_intervals = load_set100_membership_intervals(paths)
    th_active = th_intervals.loc[
        (th_intervals["start_date"] <= latest_date)
        & (th_intervals["end_date"] >= latest_date)
    ]
    us_candidates = [
        ticker
        for ticker in sp_active["ticker"].drop_duplicates()
        if ticker in source_cols
    ]
    th_candidates = [
        ticker
        for ticker in th_active["ticker"].drop_duplicates()
        if ticker in extra_cols
    ]
    return _rank_liquid_tickers(us_candidates, us_assets), _rank_liquid_tickers(th_candidates, th_assets)


def _load_thb_panel(asset_tickers: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    paths = default_paths(ROOT)
    usd_assets = [ticker for ticker in asset_tickers if not ticker.endswith(".BK")]
    th_assets = [ticker for ticker in asset_tickers if ticker.endswith(".BK")]
    needed = list(dict.fromkeys(asset_tickers + ["SPY", "^VIX"]))
    cached = load_cached_market_data(paths, tickers=needed)
    prices = cached["prices"].loc[START_DATE:END_DATE].reindex(columns=needed).sort_index().ffill()
    volumes = cached["volumes"].loc[START_DATE:END_DATE].reindex(columns=needed).fillna(0.0)

    overlay_prices = load_overlay_compare_prices(
        paths,
        start_date=START_DATE,
        end_date=END_DATE,
        tickers=["SPY", "GC=F", "BTC-USD", "^VIX", "USDTHB=X"],
    ).sort_index().ffill()
    fx = overlay_prices["USDTHB=X"].reindex(prices.index).ffill()

    thb_prices = pd.DataFrame(index=prices.index)
    if th_assets:
        thb_prices[th_assets] = prices[th_assets]
    for ticker in usd_assets:
        if ticker in prices.columns:
            thb_prices[ticker] = prices[ticker].mul(fx)
        elif ticker in overlay_prices.columns:
            thb_prices[ticker] = overlay_prices[ticker].reindex(prices.index).ffill().mul(fx)

    benchmark = overlay_prices["SPY"].reindex(prices.index).ffill().mul(fx).rename("benchmark")
    vol_proxy = overlay_prices["^VIX"].reindex(prices.index).ffill().rename("vol_proxy")
    common_index = thb_prices.index.intersection(benchmark.dropna().index).intersection(vol_proxy.dropna().index)
    thb_prices = thb_prices.reindex(common_index).ffill().dropna(how="all")
    volumes = volumes.reindex(thb_prices.index).reindex(columns=thb_prices.columns).fillna(0.0)
    benchmark = benchmark.reindex(thb_prices.index).ffill()
    vol_proxy = vol_proxy.reindex(thb_prices.index).ffill()
    fx = fx.reindex(thb_prices.index).ffill()
    return thb_prices, volumes, benchmark, vol_proxy, fx


def _run_model_on_prices(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    benchmark: pd.Series,
    vol_proxy: pd.Series,
    objective_mode: str = "mean_variance",
    max_weight: float = MAX_WEIGHT,
) -> dict[str, object]:
    returns = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    benchmark_ret = benchmark.pct_change().rename("benchmark")
    vol_proxy_ret = vol_proxy.pct_change().rename("vol_proxy")
    schedule = monthly_rebalance_dates(prices.index, lookback_days=LOOKBACK_DAYS, freq="ME")

    feature_history: dict[pd.Timestamp, pd.DataFrame] = {}
    market_stress_history: dict[pd.Timestamp, float] = {}
    universe_history: dict[pd.Timestamp, list[str]] = {}
    for rebalance_date in schedule:
        loc = prices.index.get_loc(rebalance_date)
        train_index = prices.index[max(0, loc - LOOKBACK_DAYS + 1) : loc + 1]
        train_returns = returns.reindex(train_index).dropna(axis=1, thresh=max(int(0.85 * len(train_index)), 60))
        if train_returns.shape[1] < max(N_CLUSTERS + 2, 6):
            continue
        universe_history[rebalance_date] = train_returns.columns.tolist()
        market_stress_history[rebalance_date] = compute_market_stress_signal(
            benchmark_ret.reindex(train_index),
            vol_proxy_ret.reindex(train_index),
        )
        feature_table = compute_feature_table(
            train_returns,
            benchmark_ret.reindex(train_index),
            vol_proxy_ret.reindex(train_index),
            prices.reindex(train_index)[train_returns.columns],
            include_momentum_features=True,
            feature_flags=FEATURE_FLAGS,
        )
        if not feature_table.empty:
            feature_history[rebalance_date] = feature_table

    if not feature_history:
        raise RuntimeError("No feature history was available for the joint model.")

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
        train_returns = returns.reindex(train_index)[current_assets].dropna(how="all")
        bench_train = benchmark_ret.reindex(train_index)
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

        weights = {
            "Equal Weight": eq_weights,
            "Risk Parity": optimize_risk_parity(risk_parity_cov, max_weight=max_weight),
            "Static Copula": optimize_portfolio(
                static_cov,
                momentum_signal,
                max_weight=max_weight,
                objective_mode=objective_mode,
            ),
            "Dynamic HMM Copula": optimize_portfolio(
                dyn_cov,
                momentum_signal,
                max_weight=max_weight,
                objective_mode=objective_mode,
            ),
        }
        period_returns = returns.reindex(test_index)[current_assets].fillna(0.0)
        for strategy, strategy_weights in weights.items():
            weights_history[strategy][rebalance_date] = strategy_weights
            weighted = period_returns.mul(strategy_weights, axis=1).sum(axis=1)
            starting_value = float(nav[strategy].iloc[-1])
            nav[strategy] = pd.concat([nav[strategy], starting_value * (1.0 + weighted).cumprod()])

    nav = {name: series[~series.index.duplicated(keep="last")].sort_index() for name, series in nav.items()}
    return {
        "nav": nav,
        "weights_history": weights_history,
        "universe_history": universe_history,
        "initial_clusters": initial,
        "dynamic_state": dynamic_state,
        "optimizer_objective": objective_mode,
        "max_weight": max_weight,
    }


def _overlay_equity_with_gold_btc(
    model_results: dict[str, object],
    benchmark: pd.Series,
    name_suffix: str = "",
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    paths = default_paths(ROOT)
    overlay_prices = load_overlay_compare_prices(
        paths,
        start_date="2016-01-01",
        end_date=END_DATE,
        tickers=["SPY", "GC=F", "BTC-USD", "^VIX", "USDTHB=X"],
    ).dropna(subset=["SPY", "GC=F", "BTC-USD", "^VIX", "USDTHB=X"])
    fx_returns = overlay_prices["USDTHB=X"].pct_change(fill_method=None).fillna(0.0)
    gold_thb = overlay_prices["GC=F"].mul(overlay_prices["USDTHB=X"])
    btc_thb = overlay_prices["BTC-USD"].mul(overlay_prices["USDTHB=X"])
    gold_returns = gold_thb.pct_change(fill_method=None).fillna(0.0)
    btc_returns = btc_thb.pct_change(fill_method=None).fillna(0.0)
    gold_overlay = gold_returns.mul(compare_trend_exposure(overlay_prices["GC=F"], 0.50).shift(1).reindex(gold_returns.index).ffill().fillna(1.0))
    btc_overlay = btc_returns.mul(compare_trend_exposure(overlay_prices["BTC-USD"], 0.00).shift(1).reindex(btc_returns.index).ffill().fillna(1.0))

    curves: dict[str, pd.Series] = {}
    rows = []
    for strategy in ["Static Copula", "Dynamic HMM Copula"]:
        sample_index = model_results["nav"][strategy].index.intersection(benchmark.index).sort_values()
        equity_returns = model_results["nav"][strategy].reindex(sample_index).pct_change(fill_method=None).fillna(0.0)
        equity_overlay, _ = apply_daily_exposure_overlay(
            equity_returns,
            benchmark.reindex(sample_index).ffill(),
            overlay_prices["^VIX"].reindex(sample_index).ffill(),
        )
        sleeves = pd.concat(
            {
                "JOINT_EQUITY": equity_overlay,
                "GOLD": gold_overlay,
                "BTC": btc_overlay,
            },
            axis=1,
        ).dropna()
        returns = compare_rebalanced_portfolio(
            sleeves,
            weights=pd.Series({"JOINT_EQUITY": 0.60, "GOLD": 0.30, "BTC": 0.10}, dtype=float),
            rebalance_months=1,
        )
        name = f"Joint US+TH {strategy}/Gold/BTC 60/30/10{name_suffix}"
        curve = curve_from_returns(returns)
        curves[name] = curve
        row = compute_port_opt_style_metrics(curve, risk_free_rate=0.03).to_dict()
        row["Start"] = curve.dropna().index.min().date().isoformat()
        row["End"] = curve.dropna().index.max().date().isoformat()
        row["Strategy"] = name
        rows.append(row)
    return pd.DataFrame(rows).set_index("Strategy"), curves


def _dynamic_rebalanced_returns(
    sleeve_returns: pd.DataFrame,
    strategic_weights: pd.Series,
    exposures: pd.DataFrame,
    rebalance_months: int = 1,
    transaction_cost_bps: float = 0.0,
    reallocate_cash: bool = False,
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    sleeve_returns = sleeve_returns.sort_index().fillna(0.0)
    strategic_weights = strategic_weights.reindex(sleeve_returns.columns).fillna(0.0)
    strategic_weights = strategic_weights / strategic_weights.sum()
    exposures = (
        exposures.reindex(sleeve_returns.index)
        .reindex(columns=sleeve_returns.columns)
        .shift(1)
        .ffill()
        .fillna(1.0)
        .clip(0.0, 1.0)
    )
    month_ends = sleeve_returns.groupby(sleeve_returns.index.to_period("M")).tail(1).index
    strategic_rebalance_dates = set(month_ends if rebalance_months <= 1 else month_ends[::rebalance_months])
    current_strategic = strategic_weights.copy()
    previous_effective = pd.Series(0.0, index=sleeve_returns.columns, dtype=float)
    returns = []
    turnovers = []
    effective_weights = []
    cost_rate = transaction_cost_bps / 10_000.0
    for dt, row in sleeve_returns.iterrows():
        if dt in strategic_rebalance_dates:
            current_strategic = strategic_weights.copy()
        desired = current_strategic * exposures.loc[dt]
        if reallocate_cash:
            idle_weight = float(current_strategic.sum() - desired.sum())
            eligible = exposures.loc[dt] >= 0.999
            eligible_weights = current_strategic.where(eligible, 0.0)
            if idle_weight > 0 and eligible_weights.sum() > 0:
                desired = desired + idle_weight * eligible_weights / eligible_weights.sum()
        desired = desired.clip(lower=0.0)
        if desired.sum() > 1.0:
            desired = desired / desired.sum()

        traded = float((desired - previous_effective).abs().sum())

        gross_return = float((desired * row).sum())
        net_return = gross_return - traded * cost_rate
        returns.append((dt, net_return))
        turnovers.append((dt, traded))
        effective_weights.append(desired.rename(dt))
        previous_effective = desired.copy()

    return (
        pd.Series(dict(returns), name="Portfolio").sort_index(),
        pd.Series(dict(turnovers), name="Traded Notional").sort_index(),
        pd.DataFrame(effective_weights).fillna(0.0).sort_index(),
    )


def _best_config_extension_tests(
    model_results: dict[str, object],
    benchmark: pd.Series,
    label_prefix: str = "Best joint dynamic",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = default_paths(ROOT)
    overlay_prices = load_overlay_compare_prices(
        paths,
        start_date="2016-01-01",
        end_date=END_DATE,
        tickers=["SPY", "GC=F", "BTC-USD", "^VIX", "USDTHB=X"],
    ).dropna(subset=["SPY", "GC=F", "BTC-USD", "^VIX", "USDTHB=X"])
    gold_thb = overlay_prices["GC=F"].mul(overlay_prices["USDTHB=X"])
    btc_thb = overlay_prices["BTC-USD"].mul(overlay_prices["USDTHB=X"])
    gold_returns = gold_thb.pct_change(fill_method=None).fillna(0.0)
    btc_returns = btc_thb.pct_change(fill_method=None).fillna(0.0)
    gold_exposure = compare_trend_exposure(overlay_prices["GC=F"], 0.50)
    btc_exposure = compare_trend_exposure(overlay_prices["BTC-USD"], 0.00)

    sample_index = model_results["nav"]["Dynamic HMM Copula"].index.intersection(benchmark.index).sort_values()
    equity_returns = model_results["nav"]["Dynamic HMM Copula"].reindex(sample_index).pct_change(fill_method=None).fillna(0.0)
    _, equity_exposure = apply_daily_exposure_overlay(
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
    weights = pd.Series({"JOINT_EQUITY": 0.60, "GOLD": 0.30, "BTC": 0.10}, dtype=float)
    total_cost_bps = DIME_STYLE_COMMISSION_BPS + SLIPPAGE_BPS
    variants = {
        f"{label_prefix} cash drag, no cost": {"cost_bps": 0.0, "reallocate": False},
        f"{label_prefix} cash drag, fee+slippage": {"cost_bps": total_cost_bps, "reallocate": False},
        f"{label_prefix} realloc idle exposure, no cost": {"cost_bps": 0.0, "reallocate": True},
        f"{label_prefix} realloc idle exposure, fee+slippage": {"cost_bps": total_cost_bps, "reallocate": True},
    }
    rows = []
    curves = {}
    for name, params in variants.items():
        returns, traded, _ = _dynamic_rebalanced_returns(
            sleeves,
            weights,
            exposures,
            rebalance_months=1,
            transaction_cost_bps=params["cost_bps"],
            reallocate_cash=params["reallocate"],
        )
        curve = curve_from_returns(returns)
        curves[name] = curve
        row = compute_port_opt_style_metrics(curve, risk_free_rate=0.03).to_dict()
        row["Start"] = curve.dropna().index.min().date().isoformat()
        row["End"] = curve.dropna().index.max().date().isoformat()
        row["Strategy"] = name
        row["Objective"] = BEST_OBJECTIVE
        row["Fee Bps"] = params["cost_bps"]
        row["Reallocate Idle Exposure"] = params["reallocate"]
        row["Avg Monthly Traded Notional"] = float(traded[traded > 0].mean())
        rows.append(row)
    return pd.DataFrame(rows).set_index("Strategy").sort_values("Sharpe", ascending=False), pd.DataFrame(curves).dropna(how="all")


def _summarize_all_asset_model(
    model_results: dict[str, object],
    name_suffix: str = "",
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    rows = []
    curves = {}
    for strategy in ["Static Copula", "Dynamic HMM Copula"]:
        name = f"All assets in one {strategy} model{name_suffix}"
        curve = model_results["nav"][strategy].loc["2016-01-01":].mul(10_000.0)
        curves[name] = curve
        row = compute_port_opt_style_metrics(curve, risk_free_rate=0.03).to_dict()
        row["Start"] = curve.dropna().index.min().date().isoformat()
        row["End"] = curve.dropna().index.max().date().isoformat()
        row["Strategy"] = name
        rows.append(row)
    return pd.DataFrame(rows).set_index("Strategy"), curves


def _asset_count_max_weight_sweep() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    curves = {}
    for case in ASSET_MAX_WEIGHT_SWEEP:
        us_tickers, th_tickers = _build_ranked_us_th_universe(case["us_assets"], case["th_assets"])
        tickers = list(dict.fromkeys(us_tickers + th_tickers))
        label = f"US{case['us_assets']}/TH{case['th_assets']}/max{int(case['max_weight'] * 100)}"
        print(f"Running asset/max-weight sweep: {label}")
        prices, volumes, benchmark, vol_proxy, _ = _load_thb_panel(tickers)
        results = _run_model_on_prices(
            prices,
            volumes,
            benchmark,
            vol_proxy,
            objective_mode=BEST_OBJECTIVE,
            max_weight=case["max_weight"],
        )
        summary, case_curves = _overlay_equity_with_gold_btc(
            results,
            benchmark,
            name_suffix=f" [{label}]",
        )
        dynamic_name = f"Joint US+TH Dynamic HMM Copula/Gold/BTC 60/30/10 [{label}]"
        static_name = f"Joint US+TH Static Copula/Gold/BTC 60/30/10 [{label}]"
        for strategy_name in [dynamic_name, static_name]:
            row = summary.loc[strategy_name].to_dict()
            row["Strategy"] = strategy_name
            row["Case"] = label
            row["US Assets"] = case["us_assets"]
            row["TH Assets"] = case["th_assets"]
            row["Max Weight"] = case["max_weight"]
            row["Objective"] = BEST_OBJECTIVE
            rows.append(row)
        curves.update(case_curves)
    return pd.DataFrame(rows).set_index("Strategy").sort_values("Sharpe", ascending=False), pd.DataFrame(curves).dropna(how="all")


def _best_asset_sweep_extension_tests() -> tuple[pd.DataFrame, pd.DataFrame]:
    us_tickers, th_tickers = _build_ranked_us_th_universe(
        BEST_ASSET_SWEEP_CASE["us_assets"],
        BEST_ASSET_SWEEP_CASE["th_assets"],
    )
    tickers = list(dict.fromkeys(us_tickers + th_tickers))
    prices, volumes, benchmark, vol_proxy, _ = _load_thb_panel(tickers)
    results = _run_model_on_prices(
        prices,
        volumes,
        benchmark,
        vol_proxy,
        objective_mode=BEST_OBJECTIVE,
        max_weight=BEST_ASSET_SWEEP_CASE["max_weight"],
    )
    _write_best_asset_weight_history(results)
    _write_best_asset_daily_exposure(results, benchmark)
    _side_trigger_reallocation_test(results, prices)
    summary, curves = _best_config_extension_tests(
        results,
        benchmark,
        label_prefix=f"Best asset sweep {BEST_ASSET_SWEEP_CASE['label']} dynamic",
    )
    summary["Case"] = BEST_ASSET_SWEEP_CASE["label"]
    summary["US Assets"] = BEST_ASSET_SWEEP_CASE["us_assets"]
    summary["TH Assets"] = BEST_ASSET_SWEEP_CASE["th_assets"]
    summary["Max Weight"] = BEST_ASSET_SWEEP_CASE["max_weight"]
    return summary, curves


def _weights_history_to_frame(history: dict[pd.Timestamp, pd.Series]) -> pd.DataFrame:
    rows = []
    for rebalance_date, weights in sorted(history.items()):
        rows.append(weights.rename(pd.Timestamp(rebalance_date)))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).fillna(0.0).sort_index(axis=1)


def _write_best_asset_weight_history(model_results: dict[str, object]) -> None:
    paths = default_paths(ROOT)
    dynamic_history = model_results["weights_history"]["Dynamic HMM Copula"]
    sleeve_weights = _weights_history_to_frame(dynamic_history)
    sleeve_weights.to_csv(paths.result_dir / "us_th_best_asset_sweep_dynamic_weight_history_thb.csv")

    full_weights = 0.60 * sleeve_weights
    full_weights["GOLD"] = 0.30
    full_weights["BTC"] = 0.10
    full_weights = full_weights.reindex(sorted(sleeve_weights.columns) + ["GOLD", "BTC"], axis=1)
    full_weights.to_csv(paths.result_dir / "us_th_best_asset_sweep_full_asset_weight_history_thb.csv")

    latest_date = full_weights.index.max()
    latest = pd.DataFrame(
        {
            "Asset": full_weights.columns,
            "Portfolio Weight": full_weights.loc[latest_date].values,
        }
    )
    latest["Equity Sleeve Weight"] = latest["Asset"].map(
        lambda asset: float(sleeve_weights.loc[latest_date, asset]) if asset in sleeve_weights.columns else np.nan
    )
    latest["Sleeve"] = "US Equity"
    latest.loc[latest["Asset"].str.endswith(".BK"), "Sleeve"] = "TH Equity"
    latest.loc[latest["Asset"].isin(["GOLD", "BTC"]), "Sleeve"] = "Overlay"
    latest["Date"] = pd.Timestamp(latest_date).date().isoformat()
    latest = latest.sort_values("Portfolio Weight", ascending=False)
    latest.to_csv(paths.result_dir / "us_th_best_asset_sweep_latest_asset_weights_thb.csv", index=False)


def _daily_asset_exposure_from_sleeves(
    sleeve_weights: pd.DataFrame,
    effective_sleeve_weights: pd.DataFrame,
) -> pd.DataFrame:
    valid_index = effective_sleeve_weights.index[effective_sleeve_weights.index >= sleeve_weights.index.min()]
    effective_sleeve_weights = effective_sleeve_weights.reindex(valid_index)
    daily_stock_weights = sleeve_weights.reindex(valid_index).ffill().fillna(0.0)
    asset_exposure = daily_stock_weights.mul(effective_sleeve_weights["JOINT_EQUITY"], axis=0)
    asset_exposure["GOLD"] = effective_sleeve_weights["GOLD"]
    asset_exposure["BTC"] = effective_sleeve_weights["BTC"]
    asset_exposure["CASH"] = (1.0 - asset_exposure.sum(axis=1)).clip(lower=0.0)
    return asset_exposure.sort_index(axis=1)


def _write_best_asset_daily_exposure(model_results: dict[str, object], benchmark: pd.Series) -> None:
    paths = default_paths(ROOT)
    overlay_prices = load_overlay_compare_prices(
        paths,
        start_date="2016-01-01",
        end_date=END_DATE,
        tickers=["SPY", "GC=F", "BTC-USD", "^VIX", "USDTHB=X"],
    ).dropna(subset=["SPY", "GC=F", "BTC-USD", "^VIX", "USDTHB=X"])
    gold_thb = overlay_prices["GC=F"].mul(overlay_prices["USDTHB=X"])
    btc_thb = overlay_prices["BTC-USD"].mul(overlay_prices["USDTHB=X"])
    gold_returns = gold_thb.pct_change(fill_method=None).fillna(0.0)
    btc_returns = btc_thb.pct_change(fill_method=None).fillna(0.0)
    gold_exposure = compare_trend_exposure(overlay_prices["GC=F"], 0.50)
    btc_exposure = compare_trend_exposure(overlay_prices["BTC-USD"], 0.00)

    sample_index = model_results["nav"]["Dynamic HMM Copula"].index.intersection(benchmark.index).sort_values()
    equity_returns = model_results["nav"]["Dynamic HMM Copula"].reindex(sample_index).pct_change(fill_method=None).fillna(0.0)
    _, equity_exposure = apply_daily_exposure_overlay(
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
    sleeve_weight_history = _weights_history_to_frame(model_results["weights_history"]["Dynamic HMM Copula"])

    _, _, cash_drag_sleeves = _dynamic_rebalanced_returns(
        sleeves,
        strategic_weights,
        exposures,
        rebalance_months=1,
        transaction_cost_bps=0.0,
        reallocate_cash=False,
    )
    _, _, realloc_sleeves = _dynamic_rebalanced_returns(
        sleeves,
        strategic_weights,
        exposures,
        rebalance_months=1,
        transaction_cost_bps=0.0,
        reallocate_cash=True,
    )
    cash_drag_assets = _daily_asset_exposure_from_sleeves(sleeve_weight_history, cash_drag_sleeves)
    realloc_assets = _daily_asset_exposure_from_sleeves(sleeve_weight_history, realloc_sleeves)

    cash_drag_assets.to_csv(paths.result_dir / "us_th_best_asset_sweep_daily_asset_exposure_cash_drag_thb.csv")
    realloc_assets.to_csv(paths.result_dir / "us_th_best_asset_sweep_daily_asset_exposure_realloc_idle_thb.csv")
    cash_drag_sleeves.to_csv(paths.result_dir / "us_th_best_asset_sweep_daily_sleeve_exposure_cash_drag_thb.csv")
    realloc_sleeves.to_csv(paths.result_dir / "us_th_best_asset_sweep_daily_sleeve_exposure_realloc_idle_thb.csv")


def _side_trigger_asset_exposure(
    prices: pd.DataFrame,
    sleeve_weight_history: pd.DataFrame,
    us_exposure: pd.Series,
    th_exposure: pd.Series,
    gold_exposure: pd.Series,
    btc_exposure: pd.Series,
    reallocate_stock_sleeve: bool,
) -> pd.DataFrame:
    index = prices.index[prices.index >= sleeve_weight_history.index.min()]
    weights = sleeve_weight_history.reindex(index).ffill().fillna(0.0)
    base_stock = 0.60 * weights
    us_cols = [column for column in base_stock.columns if not str(column).endswith(".BK")]
    th_cols = [column for column in base_stock.columns if str(column).endswith(".BK")]

    exposure = pd.DataFrame(0.0, index=index, columns=base_stock.columns, dtype=float)
    exposure[us_cols] = base_stock[us_cols].mul(us_exposure.reindex(index).ffill().bfill(), axis=0)
    exposure[th_cols] = base_stock[th_cols].mul(th_exposure.reindex(index).ffill().bfill(), axis=0)

    if reallocate_stock_sleeve:
        for dt in index:
            idle = 0.60 - float(exposure.loc[dt, base_stock.columns].sum())
            if idle <= 1e-12:
                continue
            eligible_cols = []
            if float(us_exposure.reindex(index).ffill().bfill().loc[dt]) >= 0.999:
                eligible_cols.extend(us_cols)
            if float(th_exposure.reindex(index).ffill().bfill().loc[dt]) >= 0.999:
                eligible_cols.extend(th_cols)
            eligible_base = base_stock.loc[dt, eligible_cols]
            eligible_base = eligible_base[eligible_base > 0.0]
            if eligible_base.sum() > 0:
                exposure.loc[dt, eligible_base.index] += idle * eligible_base / eligible_base.sum()

    exposure["GOLD"] = 0.30 * gold_exposure.reindex(index).ffill().bfill()
    exposure["BTC"] = 0.10 * btc_exposure.reindex(index).ffill().bfill()
    exposure["CASH"] = (1.0 - exposure.sum(axis=1)).clip(lower=0.0)
    return exposure.sort_index(axis=1)


def _returns_from_asset_exposure(
    asset_returns: pd.DataFrame,
    exposure: pd.DataFrame,
    transaction_cost_bps: float,
) -> tuple[pd.Series, pd.Series]:
    lagged_exposure = exposure.shift(1).ffill().fillna(1.0)
    aligned_returns = asset_returns.reindex(lagged_exposure.index).reindex(columns=lagged_exposure.columns).fillna(0.0)
    aligned_returns["CASH"] = 0.0
    cost_rate = transaction_cost_bps / 10_000.0
    previous = pd.Series(0.0, index=lagged_exposure.columns, dtype=float)
    rows = []
    turnovers = []
    for dt, weights in lagged_exposure.iterrows():
        traded = float((weights - previous).abs().sum())
        rows.append((dt, float((weights * aligned_returns.loc[dt]).sum()) - traded * cost_rate))
        turnovers.append((dt, traded))
        previous = weights.copy()
    return (
        pd.Series(dict(rows), name="Portfolio").sort_index(),
        pd.Series(dict(turnovers), name="Traded Notional").sort_index(),
    )


def _side_trigger_reallocation_test(
    model_results: dict[str, object],
    prices: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = default_paths(ROOT)
    overlay_prices = load_overlay_compare_prices(
        paths,
        start_date="2016-01-01",
        end_date=END_DATE,
        tickers=["SPY", "GC=F", "BTC-USD", "^VIX", "USDTHB=X"],
    ).dropna(subset=["SPY", "GC=F", "BTC-USD", "^VIX", "USDTHB=X"])
    set_index = pd.read_parquet(paths.local_cache_root / "extra_prices.parquet")["^SET.BK"].loc[START_DATE:END_DATE].ffill()
    fx = overlay_prices["USDTHB=X"].reindex(prices.index).ffill()
    gold_thb = overlay_prices["GC=F"].mul(overlay_prices["USDTHB=X"])
    btc_thb = overlay_prices["BTC-USD"].mul(overlay_prices["USDTHB=X"])

    asset_prices = prices.copy()
    asset_prices["GOLD"] = gold_thb.reindex(asset_prices.index).ffill()
    asset_prices["BTC"] = btc_thb.reindex(asset_prices.index).ffill()
    asset_returns = asset_prices.pct_change(fill_method=None).fillna(0.0)

    sample_returns = pd.Series(0.0, index=asset_returns.index, name="dummy")
    _, us_exposure_df = apply_daily_exposure_overlay(
        sample_returns,
        overlay_prices["SPY"].reindex(asset_returns.index).ffill(),
        overlay_prices["^VIX"].reindex(asset_returns.index).ffill(),
    )
    _, th_exposure_df = apply_daily_exposure_overlay(
        sample_returns,
        set_index.reindex(asset_returns.index).ffill(),
        None,
    )
    gold_exposure = compare_trend_exposure(overlay_prices["GC=F"], 0.50)
    btc_exposure = compare_trend_exposure(overlay_prices["BTC-USD"], 0.00)
    sleeve_weight_history = _weights_history_to_frame(model_results["weights_history"]["Dynamic HMM Copula"])

    total_cost_bps = DIME_STYLE_COMMISSION_BPS + SLIPPAGE_BPS
    variants = {
        "Side trigger cash drag, no cost": {"reallocate": False, "cost_bps": 0.0},
        "Side trigger cash drag, fee+slippage": {"reallocate": False, "cost_bps": total_cost_bps},
        "Side trigger realloc to active stock side, no cost": {"reallocate": True, "cost_bps": 0.0},
        "Side trigger realloc to active stock side, fee+slippage": {"reallocate": True, "cost_bps": total_cost_bps},
    }
    rows = []
    curves = {}
    exposures_to_save = {}
    for name, params in variants.items():
        exposure = _side_trigger_asset_exposure(
            prices=asset_prices,
            sleeve_weight_history=sleeve_weight_history,
            us_exposure=us_exposure_df["Daily Exposure"],
            th_exposure=th_exposure_df["Daily Exposure"],
            gold_exposure=gold_exposure,
            btc_exposure=btc_exposure,
            reallocate_stock_sleeve=params["reallocate"],
        )
        returns, traded = _returns_from_asset_exposure(asset_returns, exposure, params["cost_bps"])
        curve = curve_from_returns(returns)
        curves[name] = curve
        row = compute_port_opt_style_metrics(curve, risk_free_rate=0.03).to_dict()
        row["Start"] = curve.dropna().index.min().date().isoformat()
        row["End"] = curve.dropna().index.max().date().isoformat()
        row["Strategy"] = name
        row["Objective"] = BEST_OBJECTIVE
        row["Fee Bps"] = params["cost_bps"]
        row["Reallocate Stock Sleeve"] = params["reallocate"]
        row["Avg Monthly Traded Notional"] = float(traded[traded > 0].mean())
        rows.append(row)
        exposures_to_save[name] = exposure

    summary = pd.DataFrame(rows).set_index("Strategy").sort_values("Sharpe", ascending=False)
    curve_df = pd.DataFrame(curves).dropna(how="all")
    summary.to_csv(paths.result_dir / "us_th_side_trigger_reallocation_summary_thb.csv")
    curve_df.to_csv(paths.result_dir / "us_th_side_trigger_reallocation_curves_thb.csv")
    _write_side_trigger_best_files(summary, exposures_to_save)
    exposures_to_save["Side trigger cash drag, no cost"].to_csv(
        paths.result_dir / "us_th_side_trigger_daily_asset_exposure_cash_drag_thb.csv"
    )
    exposures_to_save["Side trigger realloc to active stock side, no cost"].to_csv(
        paths.result_dir / "us_th_side_trigger_daily_asset_exposure_realloc_stock_thb.csv"
    )
    pd.concat(
        {
            "US_SPY_Trigger": us_exposure_df["Daily Exposure"],
            "TH_SET_Trigger": th_exposure_df["Daily Exposure"],
            "GOLD_Trigger": gold_exposure,
            "BTC_Trigger": btc_exposure,
        },
        axis=1,
    ).reindex(asset_returns.index).ffill().bfill().to_csv(paths.result_dir / "us_th_side_trigger_daily_trigger_exposure_thb.csv")
    return summary, curve_df


def _asset_sleeve(asset: str) -> str:
    if asset == "CASH":
        return "Cash"
    if asset in {"GOLD", "BTC"}:
        return "Overlay"
    if asset.endswith(".BK"):
        return "TH Equity"
    return "US Equity"


def _asset_trigger_source(asset: str) -> str:
    if asset == "CASH":
        return "Uninvested exposure"
    if asset == "GOLD":
        return "Gold trend"
    if asset == "BTC":
        return "BTC trend"
    if asset.endswith(".BK"):
        return "^SET.BK"
    return "SPY + ^VIX"


def _write_side_trigger_best_files(
    summary: pd.DataFrame,
    exposures_to_save: dict[str, pd.DataFrame],
) -> None:
    paths = default_paths(ROOT)
    realistic = summary.loc[summary["Fee Bps"] > 0].sort_values("Sharpe", ascending=False)
    if realistic.empty:
        return

    best_name = realistic.index[0]
    best_row = realistic.iloc[0].copy()
    best_row["Strategy"] = best_name
    best_summary = best_row.to_frame().T
    best_summary.to_csv(paths.result_dir / "us_th_best_config_side_trigger_fee_slippage.csv", index=False)

    best_exposure = exposures_to_save[best_name]
    latest_date = best_exposure.index.max()
    latest = best_exposure.loc[latest_date].rename("Portfolio Exposure").reset_index()
    latest.columns = ["Asset", "Portfolio Exposure"]
    latest = latest.loc[latest["Portfolio Exposure"] > 1e-10].copy()
    latest["Portfolio Exposure %"] = latest["Portfolio Exposure"] * 100.0
    latest["Sleeve"] = latest["Asset"].map(_asset_sleeve)
    latest["Trigger Source"] = latest["Asset"].map(_asset_trigger_source)
    latest.insert(0, "Date", pd.Timestamp(latest_date).date().isoformat())
    latest = latest.sort_values("Portfolio Exposure %", ascending=False)
    latest.to_csv(paths.result_dir / "us_th_side_trigger_latest_asset_weights_thb.csv", index=False)

    config = {
        "selected_from": "fee_slippage_results_only",
        "strategy": best_name,
        "objective": BEST_OBJECTIVE,
        "fee_bps": float(best_row["Fee Bps"]),
        "commission_bps": DIME_STYLE_COMMISSION_BPS,
        "slippage_bps": SLIPPAGE_BPS,
        "reallocate_stock_sleeve": bool(best_row["Reallocate Stock Sleeve"]),
        "us_trigger": "SPY + ^VIX",
        "thai_trigger": "^SET.BK",
        "gold_trigger": "Gold trend exposure",
        "btc_trigger": "BTC trend exposure",
        "us_assets": BEST_ASSET_SWEEP_CASE["us_assets"],
        "th_assets": BEST_ASSET_SWEEP_CASE["th_assets"],
        "max_weight": BEST_ASSET_SWEEP_CASE["max_weight"],
        "equity_sleeve_weight": 0.60,
        "gold_weight": 0.30,
        "btc_weight": 0.10,
        "lookback_days": LOOKBACK_DAYS,
        "n_clusters": N_CLUSTERS,
        "momentum_signal_mode": "mom_63",
        "feature_flags": FEATURE_FLAGS,
        "metrics": {
            key: (float(best_row[key]) if isinstance(best_row[key], (int, float, np.integer, np.floating)) else best_row[key])
            for key in [
                "Total Return",
                "CAGR",
                "Annual Vol",
                "Sharpe",
                "Sortino",
                "Max Drawdown",
                "Hit Rate",
                "Start",
                "End",
                "Avg Monthly Traded Notional",
            ]
        },
        "files": {
            "summary": "result/us_th_side_trigger_reallocation_summary_thb.csv",
            "best_config_csv": "result/us_th_best_config_side_trigger_fee_slippage.csv",
            "latest_weights": "result/us_th_side_trigger_latest_asset_weights_thb.csv",
            "daily_exposure_realloc": "result/us_th_side_trigger_daily_asset_exposure_realloc_stock_thb.csv",
            "daily_triggers": "result/us_th_side_trigger_daily_trigger_exposure_thb.csv",
        },
    }
    config_file = paths.result_dir / "us_th_best_config_side_trigger_fee_slippage.json"
    config_file.write_text(json.dumps(config, indent=2), encoding="utf-8")

    doc_file = paths.root / "doc" / "BEST_CONFIG_US_TH_SIDE_TRIGGER.md"
    doc_file.parent.mkdir(parents=True, exist_ok=True)
    doc_file.write_text(
        "\n".join(
            [
                "# Best US/TH Side Trigger Config",
                "",
                "Selected from fee/slippage-adjusted results only.",
                "",
                f"- Strategy: `{best_name}`",
                f"- Objective: `{BEST_OBJECTIVE}`",
                f"- Fee + slippage: `{float(best_row['Fee Bps']):.1f}` bps",
                f"- US trigger: `SPY + ^VIX`",
                f"- Thailand trigger: `^SET.BK`",
                f"- Reallocate stock sleeve: `{bool(best_row['Reallocate Stock Sleeve'])}`",
                f"- US assets: `{BEST_ASSET_SWEEP_CASE['us_assets']}`",
                f"- Thailand assets: `{BEST_ASSET_SWEEP_CASE['th_assets']}`",
                f"- Max stock weight: `{BEST_ASSET_SWEEP_CASE['max_weight']:.2%}` inside equity sleeve",
                f"- Strategic weights: `Equity 60% / Gold 30% / BTC 10%`",
                "",
                "## Metrics",
                "",
                f"- CAGR: `{float(best_row['CAGR']):.4%}`",
                f"- Sharpe: `{float(best_row['Sharpe']):.4f}`",
                f"- Sortino: `{float(best_row['Sortino']):.4f}`",
                f"- Max Drawdown: `{float(best_row['Max Drawdown']):.4%}`",
                f"- Hit Rate: `{float(best_row['Hit Rate']):.4f}`",
                f"- Start: `{best_row['Start']}`",
                f"- End: `{best_row['End']}`",
                "",
                "## Files",
                "",
                "- `result/us_th_best_config_side_trigger_fee_slippage.json`",
                "- `result/us_th_best_config_side_trigger_fee_slippage.csv`",
                "- `result/us_th_side_trigger_latest_asset_weights_thb.csv`",
                "- `result/us_th_side_trigger_reallocation_summary_thb.csv`",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    paths = default_paths(ROOT)
    args = set(sys.argv[1:])

    if "--help" in args or "-h" in args:
        print(
            "\n".join(
                [
                    "Run the US/TH best-config test copied from dynamic_port_opt.",
                    "",
                    "Default:",
                    "  python scripts/run_us_th_best_config.py",
                    "  Runs the current best asset sweep config only: US30/TH30/max6,",
                    "  min_vol_mom_tilt, fee+slippage, side-trigger stock-sleeve reallocation.",
                    "",
                    "Options:",
                    "  --best-config                 Same as default.",
                    "  --asset-sweep-only            Re-run the asset-count / max-weight sweep.",
                    "  --best-asset-extension-only   Backward-compatible alias for --best-config.",
                    "  --full-sweep                  Re-run the full US/TH joint model workflow.",
                ]
            )
        )
        return

    if "--asset-sweep-only" in args:
        sweep_summary, sweep_curves = _asset_count_max_weight_sweep()
        sweep_summary.to_csv(paths.result_dir / "us_th_asset_count_max_weight_sweep_thb.csv")
        sweep_curves.to_csv(paths.result_dir / "us_th_asset_count_max_weight_sweep_curves_thb.csv")
        print(sweep_summary.to_string())
        return

    if "--full-sweep" not in args:
        summary, curves = _best_asset_sweep_extension_tests()
        summary.to_csv(paths.result_dir / "us_th_best_asset_sweep_fee_realloc_summary_thb.csv")
        curves.to_csv(paths.result_dir / "us_th_best_asset_sweep_fee_realloc_curves_thb.csv")
        print(summary.to_string())
        return

    us_tickers = _read_tickers(paths.result_dir / "latest_us_hmm_members.csv")
    th_tickers = _read_tickers(paths.result_dir / "latest_th_hmm_members.csv")

    joint_equity_tickers = list(dict.fromkeys(us_tickers + th_tickers))
    all_asset_tickers = list(dict.fromkeys(joint_equity_tickers + ["GC=F", "BTC-USD"]))

    print("Running joint US+TH equity model in THB...")
    equity_prices, equity_volumes, benchmark, vol_proxy, _ = _load_thb_panel(joint_equity_tickers)
    joint_equity_results = _run_model_on_prices(equity_prices, equity_volumes, benchmark, vol_proxy)
    joint_overlay_summary, joint_overlay_curves = _overlay_equity_with_gold_btc(joint_equity_results, benchmark)

    print("Running all-asset one-model portfolio in THB...")
    all_prices, all_volumes, all_benchmark, all_vol_proxy, _ = _load_thb_panel(all_asset_tickers)
    all_asset_results = _run_model_on_prices(all_prices, all_volumes, all_benchmark, all_vol_proxy)
    all_asset_summary, all_asset_curves = _summarize_all_asset_model(all_asset_results)

    summary = pd.concat([joint_overlay_summary, all_asset_summary]).sort_values("Sharpe", ascending=False)
    curves = pd.DataFrame({**joint_overlay_curves, **all_asset_curves}).dropna(how="all")
    summary.to_csv(paths.result_dir / "us_th_joint_model_summary_thb.csv")
    curves.to_csv(paths.result_dir / "us_th_joint_model_curves_thb.csv")

    sweep_summaries = []
    sweep_curves = {}
    for objective_mode in OBJECTIVE_MODES:
        print(f"Running objective sweep: {objective_mode}")
        equity_results = _run_model_on_prices(
            equity_prices,
            equity_volumes,
            benchmark,
            vol_proxy,
            objective_mode=objective_mode,
        )
        equity_summary, equity_curves = _overlay_equity_with_gold_btc(
            equity_results,
            benchmark,
            name_suffix=f" [{objective_mode}]",
        )
        equity_summary = equity_summary.assign(Model_Group="Joint US+TH equity + Gold/BTC", Objective=objective_mode)
        all_results = _run_model_on_prices(
            all_prices,
            all_volumes,
            all_benchmark,
            all_vol_proxy,
            objective_mode=objective_mode,
        )
        all_summary, all_curves = _summarize_all_asset_model(
            all_results,
            name_suffix=f" [{objective_mode}]",
        )
        all_summary = all_summary.assign(Model_Group="All assets one model", Objective=objective_mode)
        sweep_summaries.extend([equity_summary, all_summary])
        sweep_curves.update(equity_curves)
        sweep_curves.update(all_curves)

    objective_sweep_summary = pd.concat(sweep_summaries).sort_values("Sharpe", ascending=False)
    objective_sweep_curves = pd.DataFrame(sweep_curves).dropna(how="all")
    objective_sweep_summary.to_csv(paths.result_dir / "us_th_joint_model_objective_sweep_thb.csv")
    objective_sweep_curves.to_csv(paths.result_dir / "us_th_joint_model_objective_sweep_curves_thb.csv")

    print(f"Running best config extension tests: {BEST_OBJECTIVE}")
    best_equity_results = _run_model_on_prices(
        equity_prices,
        equity_volumes,
        benchmark,
        vol_proxy,
        objective_mode=BEST_OBJECTIVE,
    )
    extension_summary, extension_curves = _best_config_extension_tests(best_equity_results, benchmark)
    extension_summary.to_csv(paths.result_dir / "us_th_best_config_extension_summary_thb.csv")
    extension_curves.to_csv(paths.result_dir / "us_th_best_config_extension_curves_thb.csv")

    sweep_summary, sweep_curves = _asset_count_max_weight_sweep()
    sweep_summary.to_csv(paths.result_dir / "us_th_asset_count_max_weight_sweep_thb.csv")
    sweep_curves.to_csv(paths.result_dir / "us_th_asset_count_max_weight_sweep_curves_thb.csv")

    best_asset_extension_summary, best_asset_extension_curves = _best_asset_sweep_extension_tests()
    best_asset_extension_summary.to_csv(paths.result_dir / "us_th_best_asset_sweep_fee_realloc_summary_thb.csv")
    best_asset_extension_curves.to_csv(paths.result_dir / "us_th_best_asset_sweep_fee_realloc_curves_thb.csv")

    pd.Series(joint_equity_tickers, name="ticker").to_csv(paths.result_dir / "us_th_joint_model_equity_members.csv", index=False)
    pd.Series(all_asset_tickers, name="ticker").to_csv(paths.result_dir / "us_th_joint_model_all_asset_members.csv", index=False)

    print(summary.to_string())
    print("\nObjective sweep")
    print(objective_sweep_summary.to_string())
    print("\nBest config extension")
    print(extension_summary.to_string())
    print("\nAsset count / max weight sweep")
    print(sweep_summary.to_string())
    print("\nBest asset sweep fee/realloc extension")
    print(best_asset_extension_summary.to_string())


if __name__ == "__main__":
    main()
