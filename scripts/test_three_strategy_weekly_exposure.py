from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DYNAMIC_ROOT = ROOT.parent / "dynamic_port_opt"
for path in [DYNAMIC_ROOT / "src", DYNAMIC_ROOT / "scripts"]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dynamic_factor_copula import (  # type: ignore  # noqa: E402
    compare_rebalanced_portfolio,
    compute_port_opt_style_metrics,
    curve_from_returns,
    default_paths,
    load_overlay_compare_prices,
)
from run_us_th_tactical_gold_crash_protection_sweep import (  # type: ignore  # noqa: E402
    SELECTED_MIX,
    _gold_crash_exposure,
)
from run_us_th_tactical_gold_exposure_sweep import _load_overlay_inputs_from_cache  # type: ignore  # noqa: E402
from run_us_th_tactical_one_model import _load_full_us_th_overlay_panel_from_cache  # type: ignore  # noqa: E402
from run_us_th_tactical_one_model_asym_group_caps import (  # type: ignore  # noqa: E402
    CASH_ASSET,
    _run_case,
)
from run_us_th_tactical_perf_momentum import (  # type: ignore  # noqa: E402
    END_DATE,
    RESULT_PREFIX,
    RISK_FREE_RATE,
    START_DATE,
    _best_tactical_daily_weight,
    _close_trend_exposure,
)
import run_us_th_tactical_gold_exposure_sweep as gold_exposure_sweep  # type: ignore  # noqa: E402
import run_us_th_tactical_one_model as one_model_runner  # type: ignore  # noqa: E402
import run_us_th_tactical_one_model_asym_group_caps as asym_runner  # type: ignore  # noqa: E402
import run_us_th_tactical_perf_momentum as tactical_runner  # type: ignore  # noqa: E402


OUT_SUMMARY = ROOT / "result" / "three_strategy_weekly_exposure_test_summary.csv"
OUT_CURVES = ROOT / "result" / "three_strategy_weekly_exposure_test_curves.csv"
OUT_ALIGNED = ROOT / "result" / "three_strategy_weekly_exposure_test_timing_aligned_summary.csv"
TARGET_ONE_MODEL_CASE = "US cap 70% / TH cap 30%"
TARGET_GOLD_STRATEGY = "Gold25 crash DD252 warn-8%/exp50% crash-20%/exp50% recover-5% panic-30%/MA200/mom63->0"
TIMING_ALIGNED_START = "2018-01-02"
INITIAL_VALUE = 10_000.0

for module in [gold_exposure_sweep, one_model_runner, asym_runner, tactical_runner]:
    module.ROOT = ROOT


def _load_tactical_th_signal_from_source(index: pd.DatetimeIndex) -> tuple[str, pd.Series]:
    paths = default_paths(DYNAMIC_ROOT)
    monthly = pd.read_csv(paths.result_dir / f"{RESULT_PREFIX}_monthly_returns_thb.csv", index_col=0, parse_dates=True)
    tactical_summary = pd.read_csv(paths.result_dir / f"{RESULT_PREFIX}_tactical_exit_summary_thb.csv")
    tactical_weights = pd.read_csv(
        paths.result_dir / f"{RESULT_PREFIX}_tactical_exit_weight_history_thb.csv",
        index_col=0,
        parse_dates=True,
    )
    return _best_tactical_daily_weight(tactical_summary, tactical_weights, monthly, index)


one_model_runner._load_tactical_th_signal = _load_tactical_th_signal_from_source
asym_runner._load_tactical_th_signal = _load_tactical_th_signal_from_source


def _weekly_exposure(exposure: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    weekly = exposure.resample("W-FRI").last()
    return weekly.reindex(exposure.index).ffill().fillna(1.0).clip(0.0, 1.0)


def _metrics(curve: pd.Series, strategy: str, mode: str, family: str) -> dict[str, object]:
    clean = curve.dropna().astype(float)
    row = compute_port_opt_style_metrics(clean, risk_free_rate=RISK_FREE_RATE).to_dict()
    row.update(
        {
            "Family": family,
            "Strategy": strategy,
            "Mode": mode,
            "Start": clean.index.min().date().isoformat(),
            "End": clean.index.max().date().isoformat(),
        }
    )
    return row


def _curve_from_weighted_returns(
    returns: pd.DataFrame,
    weights: pd.DataFrame,
    name: str,
) -> pd.Series:
    asset_cols = [column for column in weights.columns if column in returns.columns]
    weighted = returns.reindex(weights.index)[asset_cols].fillna(0.0).mul(weights[asset_cols], axis=0).sum(axis=1)
    return curve_from_returns(weighted, initial=INITIAL_VALUE).rename(name)


def _one_model_weekly() -> dict[str, pd.Series]:
    raw_curve, daily_curve, raw_weights, _daily_weights, _selected = _run_case(TARGET_ONE_MODEL_CASE, 0.70, 0.30)
    prices, _volumes, _benchmark, _vol_proxy, _us_all, _th_all = _load_full_us_th_overlay_panel_from_cache()
    returns = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)

    _, signal_prices = _load_overlay_inputs_from_cache(raw_curve.index)
    exposure = pd.DataFrame(
        {
            "US": _close_trend_exposure(signal_prices["US Equity"], 300, 0.50),
            "TH": _close_trend_exposure(signal_prices["TH Equity"], 200, 0.00),
            "GC=F": _gold_crash_exposure(
                signal_prices["Gold"],
                dd_window=252,
                warn_dd=-0.08,
                crash_dd=-0.20,
                warn_exposure=0.50,
                crash_exposure=0.50,
                recovery_dd=-0.05,
                panic_dd=-0.30,
                panic_ma_period=200,
                panic_mom_period=63,
            ),
            "BTC-USD": _close_trend_exposure(signal_prices["BTC"], 50, 0.00),
        },
        index=raw_curve.index,
    ).ffill().fillna(1.0).clip(0.0, 1.0)
    weekly = _weekly_exposure(exposure)

    weekly_weights = raw_weights.reindex(raw_curve.index).ffill().fillna(0.0).copy()
    for column in weekly_weights.columns:
        if column == CASH_ASSET:
            continue
        if column == "GC=F":
            weekly_weights[column] *= weekly["GC=F"]
        elif column == "BTC-USD":
            weekly_weights[column] *= weekly["BTC-USD"]
        elif str(column).endswith(".BK"):
            weekly_weights[column] *= weekly["TH"]
        else:
            weekly_weights[column] *= weekly["US"]
    noncash = weekly_weights.drop(columns=[CASH_ASSET], errors="ignore").sum(axis=1)
    weekly_weights[CASH_ASSET] = weekly_weights.get(CASH_ASSET, 0.0) + (
        1.0 - weekly_weights.get(CASH_ASSET, 0.0) - noncash
    ).clip(lower=0.0)
    weekly_curve = _curve_from_weighted_returns(
        returns,
        weekly_weights,
        "One-model US cap 70% / TH cap 30% + weekly exposure W-FRI",
    )
    return {
        "raw": raw_curve.rename("One-model US cap 70% / TH cap 30%"),
        "daily": daily_curve.rename("One-model US cap 70% / TH cap 30% + daily exposure"),
        "weekly": weekly_curve,
    }


def _tactical_weekly() -> dict[str, pd.Series]:
    paths = default_paths(DYNAMIC_ROOT)
    monthly = pd.read_csv(paths.result_dir / f"{RESULT_PREFIX}_monthly_returns_thb.csv", index_col=0, parse_dates=True)
    comparison = pd.read_csv(
        paths.result_dir / f"{RESULT_PREFIX}_comparison_curves_thb.csv",
        index_col=0,
        parse_dates=True,
    ).sort_index()
    tactical_summary = pd.read_csv(paths.result_dir / f"{RESULT_PREFIX}_tactical_exit_summary_thb.csv")
    tactical_weights = pd.read_csv(
        paths.result_dir / f"{RESULT_PREFIX}_tactical_exit_weight_history_thb.csv",
        index_col=0,
        parse_dates=True,
    )

    daily_returns = comparison[["US PIT optimized sleeve THB", "TH PIT optimized sleeve THB"]].pct_change(
        fill_method=None
    ).fillna(0.0)
    index = daily_returns.index
    _best_strategy, th_tactical_weight = _best_tactical_daily_weight(tactical_summary, tactical_weights, monthly, index)
    overlay_prices, signal_prices = _load_overlay_inputs_from_cache(index)
    asset_returns = pd.DataFrame(
        {
            "US Equity": daily_returns["US PIT optimized sleeve THB"],
            "TH Equity": daily_returns["TH PIT optimized sleeve THB"],
            "Gold": overlay_prices["Gold"].pct_change(fill_method=None).fillna(0.0),
            "BTC": overlay_prices["BTC"].pct_change(fill_method=None).fillna(0.0),
        },
        index=index,
    ).fillna(0.0)
    raw_weights = pd.DataFrame(
        {
            "US Equity": SELECTED_MIX["Equity"] * (1.0 - th_tactical_weight),
            "TH Equity": SELECTED_MIX["Equity"] * th_tactical_weight,
            "Gold": SELECTED_MIX["Gold"],
            "BTC": SELECTED_MIX["BTC"],
        },
        index=index,
    ).ffill().fillna(0.0)
    exposure = pd.DataFrame(
        {
            "US Equity": _close_trend_exposure(signal_prices["US Equity"], 300, 0.50),
            "TH Equity": _close_trend_exposure(signal_prices["TH Equity"], 200, 0.00),
            "Gold": _gold_crash_exposure(
                signal_prices["Gold"],
                dd_window=252,
                warn_dd=-0.08,
                crash_dd=-0.20,
                warn_exposure=0.50,
                crash_exposure=0.50,
                recovery_dd=-0.05,
                panic_dd=-0.30,
                panic_ma_period=200,
                panic_mom_period=63,
            ),
            "BTC": _close_trend_exposure(signal_prices["BTC"], 50, 0.00),
        },
        index=index,
    ).ffill().fillna(1.0).clip(0.0, 1.0)
    daily_weights = raw_weights.mul(exposure, axis=0)
    weekly_weights = raw_weights.mul(_weekly_exposure(exposure), axis=0)
    return {
        "raw": _curve_from_weighted_returns(asset_returns, raw_weights, "Tactical TH/Gold/BTC 65/25/10 raw"),
        "daily": _curve_from_weighted_returns(asset_returns, daily_weights, "Tactical TH/Gold/BTC 65/25/10 daily exposure"),
        "weekly": _curve_from_weighted_returns(
            asset_returns,
            weekly_weights,
            "Tactical TH/Gold/BTC 65/25/10 weekly exposure W-FRI",
        ),
    }


def _best_stock_weekly() -> dict[str, pd.Series]:
    source_paths = default_paths(DYNAMIC_ROOT)
    cache_paths = default_paths(ROOT)
    stock_curves = pd.read_csv(
        source_paths.result_dir / "pit_reselect_step1_stock_only_momentum_objective_maxweight_curves_thb.csv",
        index_col=0,
        parse_dates=True,
    ).sort_index()
    equity_col = "US stock only Dynamic HMM Copula [mean_variance] [with momentum] max10 PIT reselect"
    equity_curve = stock_curves[equity_col].dropna()
    index = equity_curve.index
    overlay = pd.read_parquet(cache_paths.local_cache_root / "overlay_compare_prices.parquet")
    overlay = overlay.loc[START_DATE:END_DATE, ["SPY", "GC=F", "BTC-USD", "USDTHB=X"]].sort_index().ffill()
    fx = overlay["USDTHB=X"].reindex(index).ffill()
    overlay_prices = pd.DataFrame(
        {
            "Gold": overlay["GC=F"].reindex(index).ffill().mul(fx),
            "BTC": overlay["BTC-USD"].reindex(index).ffill().mul(fx),
        },
        index=index,
    )
    asset_returns = pd.DataFrame(
        {
            "Equity": equity_curve.pct_change(fill_method=None).fillna(0.0),
            "Gold": overlay_prices["Gold"].pct_change(fill_method=None).fillna(0.0),
            "BTC": overlay_prices["BTC"].pct_change(fill_method=None).fillna(0.0),
        },
        index=index,
    ).fillna(0.0)
    raw_weights = pd.DataFrame({"Equity": 0.55, "Gold": 0.40, "BTC": 0.05}, index=index)
    exposure = pd.DataFrame(
        {
            "Equity": _close_trend_exposure(overlay["SPY"].reindex(index).ffill(), 300, 0.50),
            "Gold": _gold_crash_exposure(
                overlay["GC=F"].reindex(index).ffill(),
                dd_window=252,
                warn_dd=-0.08,
                crash_dd=-0.20,
                warn_exposure=0.50,
                crash_exposure=0.50,
                recovery_dd=-0.05,
                panic_dd=-0.30,
                panic_ma_period=200,
                panic_mom_period=63,
            ),
            "BTC": _close_trend_exposure(overlay["BTC-USD"].reindex(index).ffill(), 50, 0.00),
        },
        index=index,
    ).ffill().fillna(1.0).clip(0.0, 1.0)
    monthly_returns = compare_rebalanced_portfolio(asset_returns, weights=pd.Series({"Equity": 0.55, "Gold": 0.40, "BTC": 0.05}), rebalance_months=1)
    daily_curve = _curve_from_weighted_returns(asset_returns, raw_weights.mul(exposure, axis=0), "Best stock sleeve + 55/40/5 daily exposure")
    weekly_curve = _curve_from_weighted_returns(
        asset_returns,
        raw_weights.mul(_weekly_exposure(exposure), axis=0),
        "Best stock sleeve + 55/40/5 weekly exposure W-FRI",
    )
    return {
        "raw": curve_from_returns(monthly_returns, initial=INITIAL_VALUE).rename("Best stock sleeve + EQUITY/GOLD/BTC 55/40/5"),
        "daily": daily_curve,
        "weekly": weekly_curve,
    }


def main() -> None:
    families = {
        "One-model US cap 70 / TH cap 30": _one_model_weekly(),
        "US/TH tactical 65/25/10 Gold crash": _tactical_weekly(),
        "Best stock sleeve 55/40/5": _best_stock_weekly(),
    }
    rows = []
    aligned_rows = []
    curves = {}
    for family, variants in families.items():
        for mode, curve in variants.items():
            rows.append(_metrics(curve, curve.name, mode, family))
            aligned = curve.loc[curve.index >= pd.Timestamp(TIMING_ALIGNED_START)]
            aligned_rows.append(_metrics(aligned, curve.name, mode, family))
            curves[curve.name] = curve
    summary = pd.DataFrame(rows).sort_values(["Family", "Sharpe"], ascending=[True, False])
    aligned_summary = pd.DataFrame(aligned_rows).sort_values(["Family", "Sharpe"], ascending=[True, False])
    curve_df = pd.DataFrame(curves).dropna(how="all")
    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_SUMMARY, index=False)
    aligned_summary.to_csv(OUT_ALIGNED, index=False)
    curve_df.to_csv(OUT_CURVES)
    print("Full-history summary")
    print(summary[["Family", "Mode", "CAGR", "Annual Vol", "Sharpe", "Max Drawdown", "Start", "End"]].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print()
    print(f"Timing-aligned summary from {TIMING_ALIGNED_START}")
    print(aligned_summary[["Family", "Mode", "CAGR", "Annual Vol", "Sharpe", "Max Drawdown", "Start", "End"]].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nWrote {OUT_SUMMARY.relative_to(ROOT)}")
    print(f"Wrote {OUT_ALIGNED.relative_to(ROOT)}")
    print(f"Wrote {OUT_CURVES.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
