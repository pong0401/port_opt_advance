from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from portfolio_engine import (
    AlphaConfig,
    EPSILON,
    RiskConfig,
    apply_point_in_time_universe_filters,
    calculate_performance_metrics,
    detect_regime,
    filter_historical_universe,
)


CACHE_ROOT = Path("data/cache/portopt_optimizer_proof")
RESULT_ROOT = Path("result/top100_daily_buffer")
RESULT_ROOT.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class TestWindow:
    eval_start: pd.Timestamp
    eval_end: pd.Timestamp


TEST_WINDOWS = {
    "10Y": TestWindow(
        eval_start=pd.Timestamp(date.today()) - pd.DateOffset(years=10),
        eval_end=pd.Timestamp(date.today()),
    ),
    "20Y": TestWindow(
        eval_start=pd.Timestamp(date.today()) - pd.DateOffset(years=20),
        eval_end=pd.Timestamp(date.today()),
    ),
}

SELECTION_SCENARIOS = {
    "Top 100": {"count": 100, "buffer_count": None},
    "Top 100 buffer 125": {"count": 100, "buffer_count": 125},
    "Top 100 buffer 150": {"count": 100, "buffer_count": 150},
    "Top 100 buffer 200": {"count": 100, "buffer_count": 200},
    "Top 50": {"count": 50, "buffer_count": None},
    "Top 50 buffer 75": {"count": 50, "buffer_count": 75},
    "Top 50 buffer 100": {"count": 50, "buffer_count": 100},
    "Top 50 buffer 125": {"count": 50, "buffer_count": 125},
    "Top 25": {"count": 25, "buffer_count": None},
    "Top 25 buffer 50": {"count": 25, "buffer_count": 50},
    "Top 25 buffer 75": {"count": 25, "buffer_count": 75},
    "Top 25 buffer 100": {"count": 25, "buffer_count": 100},
}

OVERLAY_PROFILES = {
    "No daily overlay": {
        "use_daily_trend": False,
        "use_daily_drawdown": False,
        "use_daily_vix": False,
    },
    "Daily trend": {
        "use_daily_trend": True,
        "trend_cap": 0.65,
        "use_daily_drawdown": False,
        "use_daily_vix": False,
    },
    "Daily trend + drawdown": {
        "use_daily_trend": True,
        "trend_cap": 0.65,
        "use_daily_drawdown": True,
        "drawdown_warn": -0.08,
        "drawdown_warn_cap": 0.50,
        "drawdown_crash": -0.15,
        "drawdown_crash_cap": 0.25,
        "use_daily_vix": False,
    },
    "Daily trend + drawdown + VIX": {
        "use_daily_trend": True,
        "trend_cap": 0.65,
        "use_daily_drawdown": True,
        "drawdown_warn": -0.08,
        "drawdown_warn_cap": 0.50,
        "drawdown_crash": -0.15,
        "drawdown_crash_cap": 0.25,
        "use_daily_vix": True,
        "vix_warn": 28.0,
        "vix_warn_cap": 0.50,
        "vix_crash": 35.0,
        "vix_crash_cap": 0.25,
    },
}


def read_cached_frame(stem: Path) -> pd.DataFrame:
    parquet_path = stem.with_suffix(".parquet")
    pickle_path = stem.with_suffix(".pkl")
    csv_path = stem.with_suffix(".csv")
    if parquet_path.exists():
        frame = pd.read_parquet(parquet_path)
    elif pickle_path.exists():
        frame = pd.read_pickle(pickle_path)
    elif csv_path.exists():
        frame = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    else:
        raise FileNotFoundError(stem)
    frame.index = pd.to_datetime(frame.index)
    return frame.sort_index()


def load_market_bundle(window_name: str) -> dict[str, pd.DataFrame | pd.Series]:
    folder = CACHE_ROOT / window_name
    return {
        "prices": read_cached_frame(folder / "prices"),
        "volumes": read_cached_frame(folder / "volumes"),
        "benchmark": read_cached_frame(folder / "benchmark")["value"],
        "vol_proxy": read_cached_frame(folder / "vol_proxy")["value"],
    }


def month_end_rebalance_dates(
    prices: pd.DataFrame,
    lookback_months: int,
    rebalance_months: int,
    rebalance_offset: int = 0,
) -> list[pd.Timestamp]:
    month_ends = prices.groupby(prices.index.to_period("M")).tail(1).index.sort_values()
    if len(month_ends) <= lookback_months:
        return []
    start_idx = lookback_months + int(rebalance_offset)
    return list(month_ends[start_idx::rebalance_months])


def daily_overlay_cap(
    date_i: pd.Timestamp,
    benchmark: pd.Series,
    vol_proxy: pd.Series,
    profile: dict,
) -> float:
    cap = 1.0
    bench_hist = benchmark.loc[:date_i].dropna()
    if bench_hist.empty:
        return cap

    last_bench = float(bench_hist.iloc[-1])
    ma200 = float(bench_hist.rolling(200, min_periods=40).mean().iloc[-1]) if len(bench_hist) >= 40 else np.nan
    if profile.get("use_daily_trend", False) and not pd.isna(ma200) and last_bench < ma200:
        cap = min(cap, float(profile.get("trend_cap", 0.65)))

    if profile.get("use_daily_drawdown", False):
        drawdown = last_bench / float(bench_hist.cummax().iloc[-1]) - 1.0
        if drawdown <= float(profile.get("drawdown_crash", -0.15)):
            cap = min(cap, float(profile.get("drawdown_crash_cap", 0.25)))
        elif drawdown <= float(profile.get("drawdown_warn", -0.08)):
            cap = min(cap, float(profile.get("drawdown_warn_cap", 0.50)))

    if profile.get("use_daily_vix", False) and vol_proxy is not None and not vol_proxy.empty:
        vol_hist = vol_proxy.loc[:date_i].dropna()
        if not vol_hist.empty:
            vix_now = float(vol_hist.iloc[-1])
            if vix_now >= float(profile.get("vix_crash", 35.0)):
                cap = min(cap, float(profile.get("vix_crash_cap", 0.25)))
            elif vix_now >= float(profile.get("vix_warn", 28.0)):
                cap = min(cap, float(profile.get("vix_warn_cap", 0.50)))
    return float(np.clip(cap, 0.0, 1.0))


def monthly_regime_exposure(
    benchmark_window: pd.Series,
    train_prices: pd.DataFrame,
    vol_window: pd.Series,
    risk_cfg: RiskConfig,
) -> tuple[float, str]:
    regime_info = detect_regime(
        benchmark_window.dropna(),
        train_prices,
        vol_window.dropna() if vol_window is not None and not vol_window.empty else None,
        risk_cfg,
    )
    if regime_info["regime"] == "Bull":
        exposure = risk_cfg.bull_exposure
    elif regime_info["regime"] == "Neutral":
        exposure = risk_cfg.neutral_exposure
    else:
        exposure = risk_cfg.bear_exposure
    if risk_cfg.use_trend_filter and regime_info["trend_score"] <= EPSILON:
        exposure = min(exposure, risk_cfg.neutral_exposure)
    return float(np.clip(exposure, 0.0, 1.0)), str(regime_info["regime"])


def select_top100_with_buffer(
    eligible_prices: pd.DataFrame,
    liquidity_snapshot: pd.DataFrame,
    scenario_cfg: dict,
    previous_selected: list[str],
) -> list[str]:
    count = int(scenario_cfg["count"])
    clean = eligible_prices.ffill().dropna(axis=1, how="all")
    if clean.empty or liquidity_snapshot.empty:
        return []
    ordered = [
        ticker
        for ticker in liquidity_snapshot.sort_values("liquidity_rank")["ticker"].tolist()
        if ticker in clean.columns
    ]
    if not ordered:
        return []

    buffer_count = scenario_cfg.get("buffer_count")
    if buffer_count and previous_selected:
        buffer_members = set(ordered[: min(int(buffer_count), len(ordered))])
        kept = [ticker for ticker in previous_selected if ticker in buffer_members]
        additions = [ticker for ticker in ordered if ticker not in kept]
        return (kept + additions)[: min(count, len(ordered))]
    return ordered[: min(count, len(ordered))]


def run_equal_weight_top100_test(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    benchmark: pd.Series,
    vol_proxy: pd.Series,
    alpha_cfg: AlphaConfig,
    risk_cfg: RiskConfig,
    scenario_name: str,
    scenario_cfg: dict,
    overlay_name: str,
    overlay_profile: dict,
    lookback_months: int = 12,
    rebalance_months: int = 1,
    rebalance_offset: int = 0,
    safety_check_months: int | None = None,
    transaction_cost_bps: float = 12.0,
) -> dict:
    full_rebalance_dates = month_end_rebalance_dates(prices, lookback_months, rebalance_months, rebalance_offset)
    if safety_check_months and full_rebalance_dates:
        month_ends = prices.groupby(prices.index.to_period("M")).tail(1).index.sort_values()
        start_pos = int(np.searchsorted(month_ends.values, full_rebalance_dates[0].to_datetime64()))
        rebalance_dates = list(month_ends[start_pos::safety_check_months])
    else:
        rebalance_dates = full_rebalance_dates
    full_rebalance_set = set(pd.to_datetime(full_rebalance_dates))
    all_assets = list(prices.columns)
    previous_weights = pd.Series(0.0, index=all_assets)
    previous_selected: list[str] = []
    previous_exposure = 0.0
    current_weights = pd.Series(0.0, index=all_assets)
    current_base_exposure = 1.0
    equity = 10_000.0
    curve_rows = []
    exposure_rows = []
    rebalance_rows = []

    for idx, rebalance_date in enumerate(rebalance_dates):
        is_full_rebalance = pd.Timestamp(rebalance_date) in full_rebalance_set or not previous_selected
        next_date = rebalance_dates[idx + 1] if idx + 1 < len(rebalance_dates) else prices.index[-1]
        train_start = rebalance_date - pd.DateOffset(months=lookback_months)
        train_prices = prices.loc[(prices.index > train_start) & (prices.index <= rebalance_date)]
        train_volumes = volumes.reindex(train_prices.index)
        if train_prices.empty:
            continue

        snap_prices, snap_volumes, liquidity_snapshot = apply_point_in_time_universe_filters(
            prices_window=train_prices,
            volumes_window=train_volumes,
            full_prices=prices,
            as_of_date=rebalance_date,
            alpha_cfg=alpha_cfg,
        )
        eligible_prices = filter_historical_universe(snap_prices, prices, rebalance_date, alpha_cfg)
        selected = select_top100_with_buffer(eligible_prices, liquidity_snapshot, scenario_cfg, previous_selected)
        if len(selected) < 2:
            continue

        benchmark_window = benchmark.reindex(train_prices.index).dropna()
        vol_window = vol_proxy.reindex(train_prices.index).dropna() if not vol_proxy.empty else pd.Series(dtype=float)
        replacement_count = len(set(selected).symmetric_difference(previous_selected)) if previous_selected else len(selected)
        should_trade = is_full_rebalance or replacement_count > 0
        if should_trade:
            equal_weights = pd.Series(1.0 / len(selected), index=selected)
            current_base_exposure, regime = monthly_regime_exposure(benchmark_window, train_prices[selected], vol_window, risk_cfg)

            target_weights = pd.Series(0.0, index=all_assets)
            target_weights.loc[selected] = equal_weights * current_base_exposure
            stock_turnover = float((target_weights - previous_weights).abs().sum())
            equity *= max(1.0 - stock_turnover * transaction_cost_bps / 10_000.0, 0.0)
            current_weights = equal_weights
        else:
            selected = previous_selected
            regime = "Safety Check"
            target_weights = previous_weights
            stock_turnover = 0.0
            current_weights = current_weights.reindex(selected).fillna(0.0)

        period_prices = prices.loc[(prices.index > rebalance_date) & (prices.index <= next_date), selected]
        period_returns = period_prices.pct_change(fill_method=None).fillna(0.0)
        if period_returns.empty:
            previous_weights = target_weights
            previous_selected = selected
            continue

        daily_exposure_turnover_sum = 0.0
        for date_i, row in period_returns.iterrows():
            daily_cap = daily_overlay_cap(date_i, benchmark, vol_proxy, overlay_profile)
            daily_exposure = float(np.clip(min(current_base_exposure, daily_cap), 0.0, 1.0))
            exposure_turnover = abs(daily_exposure - previous_exposure)
            daily_exposure_turnover_sum += exposure_turnover
            equity *= max(1.0 - exposure_turnover * transaction_cost_bps / 10_000.0, 0.0)
            equity *= 1.0 + float(row.fillna(0.0) @ (current_weights * daily_exposure))
            curve_rows.append({"Date": date_i, "PortValue": equity})
            exposure_rows.append(
                {
                    "Date": date_i,
                    "Scenario": scenario_name,
                    "Overlay": overlay_name,
                    "Base Exposure": current_base_exposure,
                    "Daily Cap": daily_cap,
                    "Daily Exposure": daily_exposure,
                }
            )
            previous_exposure = daily_exposure

        rebalance_rows.append(
            {
                "Date": rebalance_date,
                "Scenario": scenario_name,
                "Overlay": overlay_name,
                "Event Type": "Full Rebalance" if is_full_rebalance else "Safety Check",
                "Regime": regime,
                "Target Holdings": len(selected),
                "Base Exposure": current_base_exposure,
                "Stock Turnover": stock_turnover,
                "Replacement Count": replacement_count,
                "Daily Exposure Turnover Sum": daily_exposure_turnover_sum,
                "Selected Assets": ", ".join(selected),
            }
        )
        previous_weights = target_weights
        previous_selected = selected

    curve = pd.DataFrame(curve_rows).drop_duplicates(subset="Date").set_index("Date").sort_index() if curve_rows else pd.DataFrame(columns=["PortValue"])
    exposure_history = pd.DataFrame(exposure_rows).drop_duplicates(subset="Date").set_index("Date").sort_index() if exposure_rows else pd.DataFrame()
    rebalance_report = pd.DataFrame(rebalance_rows) if rebalance_rows else pd.DataFrame()
    return {"curve": curve, "exposure_history": exposure_history, "rebalance_report": rebalance_report}


def metrics_to_dict(metrics: pd.DataFrame) -> dict:
    if metrics.empty:
        return {}
    return metrics.set_index("Metric")["Value"].to_dict()


def summarize_result(
    window_name: str,
    window: TestWindow,
    scenario_name: str,
    overlay_name: str,
    result: dict,
    benchmark: pd.Series,
    rebalance_months: int = 1,
    rebalance_offset: int = 0,
    safety_check_months: int | None = None,
    risk_free_rate: float = 0.03,
) -> dict:
    curve = result["curve"].loc[window.eval_start : window.eval_end]
    metrics = metrics_to_dict(calculate_performance_metrics(curve, risk_free_rate))
    benchmark_slice = benchmark.reindex(curve.index).ffill().dropna()
    if not benchmark_slice.empty:
        benchmark_curve = pd.DataFrame(
            {"PortValue": benchmark_slice / benchmark_slice.iloc[0] * 10_000.0},
            index=benchmark_slice.index,
        )
    else:
        benchmark_curve = pd.DataFrame(columns=["PortValue"])
    benchmark_metrics = metrics_to_dict(calculate_performance_metrics(benchmark_curve, risk_free_rate))

    rebalances = result["rebalance_report"].copy()
    if not rebalances.empty:
        rebalances["Date"] = pd.to_datetime(rebalances["Date"])
        rebalances = rebalances.loc[(rebalances["Date"] >= window.eval_start) & (rebalances["Date"] <= window.eval_end)]
    exposure = result["exposure_history"].loc[window.eval_start : window.eval_end]

    return {
        "Window": window_name,
        "Scenario": scenario_name,
        "Overlay": overlay_name,
        "Rebalance Months": rebalance_months,
        "Rebalance Offset": rebalance_offset,
        "Safety Check Months": safety_check_months if safety_check_months else np.nan,
        "CAGR": metrics.get("CAGR", np.nan),
        "Benchmark CAGR": benchmark_metrics.get("CAGR", np.nan),
        "Excess CAGR": metrics.get("CAGR", np.nan) - benchmark_metrics.get("CAGR", np.nan),
        "Sharpe": metrics.get("Sharpe", np.nan),
        "Benchmark Sharpe": benchmark_metrics.get("Sharpe", np.nan),
        "Sharpe Excess": metrics.get("Sharpe", np.nan) - benchmark_metrics.get("Sharpe", np.nan),
        "Sortino": metrics.get("Sortino", np.nan),
        "Max Drawdown": metrics.get("Max Drawdown", np.nan),
        "Benchmark Max Drawdown": benchmark_metrics.get("Max Drawdown", np.nan),
        "Avg Daily Exposure": exposure["Daily Exposure"].mean() if not exposure.empty else np.nan,
        "Min Daily Exposure": exposure["Daily Exposure"].min() if not exposure.empty else np.nan,
        "Avg Stock Turnover": rebalances["Stock Turnover"].mean() if not rebalances.empty else np.nan,
        "Avg Daily Exposure Turnover": rebalances["Daily Exposure Turnover Sum"].mean() if not rebalances.empty else np.nan,
        "Rebalances": len(rebalances),
        "Full Rebalances": int((rebalances["Event Type"] == "Full Rebalance").sum()) if "Event Type" in rebalances else len(rebalances),
        "Safety Checks": int((rebalances["Event Type"] == "Safety Check").sum()) if "Event Type" in rebalances else 0,
        "Replacements": rebalances["Replacement Count"].sum() if "Replacement Count" in rebalances else np.nan,
    }


def run_selected(
    windows: list[str] | None = None,
    scenarios: list[str] | None = None,
    overlays: list[str] | None = None,
    rebalance_months: int = 1,
    rebalance_offset: int = 0,
    safety_check_months: int | None = None,
    result_root: Path = RESULT_ROOT,
) -> pd.DataFrame:
    result_root.mkdir(parents=True, exist_ok=True)
    alpha_cfg = AlphaConfig(
        top_n=30,
        target_holdings=100,
        liquidity_cut=500,
        min_history_ratio=0.85,
        min_avg_dollar_volume_millions=0.0,
        use_fundamental_factors=False,
        use_historical_eligibility=True,
        min_listing_days=24 * 21,
    )
    risk_cfg = RiskConfig(
        use_trend_filter=True,
        use_regime_filter=True,
        max_drawdown_stop=0.0,
        bull_exposure=1.00,
        neutral_exposure=0.65,
        bear_exposure=0.25,
        regime_trend_weight=0.35,
        regime_breadth_weight=0.35,
        regime_vol_weight=0.05,
        regime_drawdown_weight=0.25,
        bull_score_threshold=0.70,
        neutral_score_threshold=0.40,
    )

    selected_windows = windows or list(TEST_WINDOWS)
    selected_scenarios = scenarios or list(SELECTION_SCENARIOS)
    selected_overlays = overlays or list(OVERLAY_PROFILES)

    summary_rows = []
    for window_name in selected_windows:
        window = TEST_WINDOWS[window_name]
        bundle = load_market_bundle(window_name)
        prices = bundle["prices"]
        volumes = bundle["volumes"].reindex(prices.index).reindex(columns=prices.columns).fillna(0.0)
        benchmark = bundle["benchmark"]
        vol_proxy = bundle["vol_proxy"]

        for scenario_name in selected_scenarios:
            scenario_cfg = SELECTION_SCENARIOS[scenario_name]
            for overlay_name in selected_overlays:
                overlay_profile = OVERLAY_PROFILES[overlay_name]
                print(f"Running {window_name}: {scenario_name} / {overlay_name}")
                result = run_equal_weight_top100_test(
                    prices=prices,
                    volumes=volumes,
                    benchmark=benchmark,
                    vol_proxy=vol_proxy,
                    alpha_cfg=alpha_cfg,
                    risk_cfg=risk_cfg,
                    scenario_name=scenario_name,
                    scenario_cfg=scenario_cfg,
                    overlay_name=overlay_name,
                    overlay_profile=overlay_profile,
                    rebalance_months=rebalance_months,
                    rebalance_offset=rebalance_offset,
                    safety_check_months=safety_check_months,
                )
                result["rebalance_report"].to_csv(
                    result_root / f"rebalance_{window_name}_{scenario_name}_{overlay_name}.csv".replace(" ", "_").replace("+", "plus"),
                    index=False,
                )
                if not result["exposure_history"].empty:
                    result["exposure_history"].to_csv(
                        result_root / f"exposure_{window_name}_{scenario_name}_{overlay_name}.csv".replace(" ", "_").replace("+", "plus")
                    )
                summary_rows.append(
                    summarize_result(
                        window_name,
                        window,
                        scenario_name,
                        overlay_name,
                        result,
                        benchmark,
                        rebalance_months=rebalance_months,
                        rebalance_offset=rebalance_offset,
                        safety_check_months=safety_check_months,
                    )
                )

    summary = pd.DataFrame(summary_rows).sort_values(
        ["Window", "Sharpe", "Avg Stock Turnover"],
        ascending=[True, False, True],
    )
    summary.to_csv(result_root / "summary.csv", index=False)

    summary["Baseline Scenario"] = np.select(
        [
            summary["Scenario"].str.startswith("Top 25"),
            summary["Scenario"].str.startswith("Top 50"),
        ],
        ["Top 25", "Top 50"],
        default="Top 100",
    )
    baseline = summary.loc[
        (summary["Scenario"] == summary["Baseline Scenario"]) & (summary["Overlay"] == "No daily overlay"),
        ["Window", "Baseline Scenario", "Avg Stock Turnover", "CAGR", "Sharpe", "Max Drawdown"],
    ].rename(
        columns={
            "Avg Stock Turnover": "Baseline Avg Stock Turnover",
            "CAGR": "Baseline CAGR",
            "Sharpe": "Baseline Sharpe",
            "Max Drawdown": "Baseline Max Drawdown",
        }
    )
    comparison = summary.merge(baseline, on=["Window", "Baseline Scenario"], how="left")
    comparison["Stock Turnover Reduction"] = comparison["Baseline Avg Stock Turnover"] - comparison["Avg Stock Turnover"]
    comparison["Stock Turnover Reduction %"] = comparison["Stock Turnover Reduction"] / comparison["Baseline Avg Stock Turnover"]
    comparison["CAGR Change vs Baseline"] = comparison["CAGR"] - comparison["Baseline CAGR"]
    comparison["Sharpe Change vs Baseline"] = comparison["Sharpe"] - comparison["Baseline Sharpe"]
    comparison["MaxDD Change vs Baseline"] = comparison["Max Drawdown"] - comparison["Baseline Max Drawdown"]
    comparison.to_csv(result_root / "comparison_vs_plain_top100.csv", index=False)
    return comparison


def run_all() -> pd.DataFrame:
    return run_selected()


if __name__ == "__main__":
    output = run_all()
    cols = [
        "Window",
        "Scenario",
        "Overlay",
        "CAGR",
        "Sharpe",
        "Max Drawdown",
        "Avg Daily Exposure",
        "Avg Stock Turnover",
        "Stock Turnover Reduction %",
        "CAGR Change vs Baseline",
        "Sharpe Change vs Baseline",
    ]
    print(output[cols].to_string(index=False))
