from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from qqq_combo_gtaa import PRICE_CACHE as QQQ_COMBO_PRICE_CACHE
from qqq_combo_gtaa import STRATEGY as QQQ_COMBO_STRATEGY
from qqq_combo_gtaa import daily_returns as qqq_combo_daily_returns
from share_class_utils import drop_duplicate_share_classes_available

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "data" / "precomputed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

START_YEARS = 10
INITIAL_VALUE = 10_000.0
RISK_FREE_RATE = 0.03
REBALANCE_MONTHS = (1, 4, 7, 10)
LOOKAHEAD_SERIES_PREFIXES = (
    "Joint US+TH Dynamic HMM Copula/Gold/BTC 60/30/10",
    "Joint US+TH Static Copula/Gold/BTC 60/30/10",
    "Side trigger whipsaw",
)
LOOKAHEAD_SERIES_NAMES = {
    "Side trigger realloc to active stock side, fee+slippage (with Gold/BTC 60/30/10)",
}
EXCLUDED_STRATEGY_NAME_PARTS = ("no cost",)


def _is_excluded_strategy_name(name: object) -> bool:
    text = str(name).casefold()
    return any(part in text for part in EXCLUDED_STRATEGY_NAME_PARTS)

HANDOFF_CURVE_SOURCES = [
    {
        "label": "S&P 500 buy and hold",
        "path": ("..", "dynamic_port_opt", "result", "best_param_step1_sp500_buy_hold_curve.csv"),
        "column": "S&P 500 Buy Hold",
        "family": "BEST_PARAM_S&P_PORT_OPT_ADVANCE",
    },
    {
        "label": "Monthly allocation SPY/Gold/BTC/BIL 35/40/10/15",
        "path": ("..", "dynamic_port_opt", "result", "best_param_step2_multi_asset_best_curves.csv"),
        "column": "SPY/Gold/BTC/BIL/IEF/VXUS/TIP 35/40/10/15/0/0/0",
        "family": "BEST_PARAM_S&P_PORT_OPT_ADVANCE",
    },
    {
        "label": "Country ETF tactical + Gold DD boost 16",
        "path": ("..", "dynamic_port_opt", "result", "spy_gold_btc_bil_combined_etf_universe_current_best_curves.csv"),
        "column": "current_best country_only bucket8% top2 boost16",
        "family": "BEST_PARAM_S&P_PORT_OPT_ADVANCE",
    },
    {
        "label": "US stock only Dynamic HMM Copula [mean_variance] [with momentum] max10 PIT reselect",
        "path": ("..", "dynamic_port_opt", "result", "pit_reselect_step1_stock_only_momentum_objective_maxweight_curves_thb.csv"),
        "column": "US stock only Dynamic HMM Copula [mean_variance] [with momentum] max10 PIT reselect",
        "family": "PIT_RESELECT_BY_STEP_HANDOFF",
    },
    {
        "label": "Best stock sleeve + EQUITY/GOLD/BTC 55/40/5",
        "path": ("..", "dynamic_port_opt", "result", "pit_reselect_step2_1_from_step1_momentum_equity_gold_btc_bil_allocation_curves_thb.csv"),
        "column": "Best stock sleeve + EQUITY/GOLD/BTC 55/40/5",
        "family": "PIT_RESELECT_BY_STEP_HANDOFF",
    },
    {
        "label": "Stocks+Gold+BTC+BIL one-model Static Copula [mean_variance] PIT reselect",
        "path": ("..", "dynamic_port_opt", "result", "pit_reselect_step2_2_from_step1_momentum_all_assets_with_bil_one_model_curves_thb.csv"),
        "column": "Stocks+Gold+BTC+BIL one-model Static Copula [mean_variance] PIT reselect",
        "family": "PIT_RESELECT_BY_STEP_HANDOFF",
    },
    {
        "label": "Stocks+Gold+BTC+BIL one-model capped Static Copula [mean_variance] PIT reselect",
        "path": ("..", "dynamic_port_opt", "result", "pit_reselect_step2_3_from_step1_momentum_all_assets_with_bil_capped_from_2_1_curves_thb.csv"),
        "column": "Stocks+Gold+BTC+BIL one-model capped Static Copula [mean_variance] PIT reselect",
        "family": "PIT_RESELECT_BY_STEP_HANDOFF",
    },
    {
        "label": "Mean Covariance Gold30 stock-cap sweep stockcap8 mom_63 + asset-level daily exposure",
        "path": ("..", "dynamic_port_opt", "result", "mean_covariance_stock_cap_sweep_daily_exposure_curves.csv"),
        "column": "Mean Covariance Gold30 stock-cap sweep stockcap8 mom_63 + asset-level daily exposure",
        "family": "PIT_RESELECT_BY_STEP_HANDOFF",
    },
    {
        "label": "One-model US cap 70% / TH cap 30% stockcap5 penalty0.02 assets50 + daily exposure",
        "path": ("result", "us_th_one_model_us70_th30_stockcap5_penalty002_assets50_weekday_june_curves_thb.csv"),
        "column": "One-model US cap 70% / TH cap 30% stockcap5 penalty0.02 assets50 + daily exposure",
        "family": "PIT_RESELECT_BY_STEP_HANDOFF",
    },
    {
        "label": "JP optimized min_vol_mom_tilt top10 cap15 weekly exposure with Gold DD252",
        "path": ("..", "dynamic_port_opt", "result", "us_th_jp_optimized_sleeve_sweep_curves_thb.csv"),
        "column": "JP optimized min_vol_mom_tilt top10 cap15% Stock60/Gold30/BTC10 Index signal leaves inactive equity in cash + weekly exposure all assets + gold drawdown 252d warn10 crash20",
        "family": "PIT_RESELECT_BY_STEP_HANDOFF",
    },
    {
        "label": "Best stock assets + Gold/BTC/BIL/IEF reoptimized Dynamic HMM Copula [US stock only] [mean_variance] max10 PIT reselect",
        "path": ("..", "dynamic_port_opt", "result", "pit_reselect_step2_4_from_step1_momentum_best_stock_assets_gold_btc_bil_ief_reoptimized_curves_thb.csv"),
        "column": "Best stock assets + Gold/BTC/BIL/IEF reoptimized Dynamic HMM Copula [US stock only] [mean_variance] max10 PIT reselect",
        "family": "PIT_RESELECT_BY_STEP_HANDOFF",
    },
    {
        "label": "S&P trend daily exposure on reoptimized PIT portfolio",
        "path": ("..", "dynamic_port_opt", "result", "pit_reselect_step2_5_daily_exposure_on_step2_4_curves_thb.csv"),
        "column": "S&P trend MA300 below0.50",
        "family": "PIT_RESELECT_BY_STEP_HANDOFF",
    },
]

HANDOFF_SUPPORT_FILES = [
    {
        "source": ("..", "dynamic_port_opt", "result", "spy_gold_btc_bil_combined_etf_universe_current_best_summary.csv"),
        "target": "spy_gold_btc_bil_combined_etf_universe_current_best_summary.csv",
        "description": "Country ETF tactical + Gold DD boost 16 sweep summary copied for standalone review.",
    },
    {
        "source": ("..", "dynamic_port_opt", "result", "spy_gold_btc_bil_combined_etf_universe_current_best_selection_history.csv"),
        "target": "spy_gold_btc_bil_combined_etf_universe_current_best_selection_history.csv",
        "description": "Country ETF tactical + Gold DD boost 16 monthly selected ETF history copied for standalone review.",
    },
    {
        "source": ("..", "dynamic_port_opt", "result", "pit_reselect_step2_5_latest_effective_security_weights_thb.csv"),
        "target": "pit_reselect_step2_5_latest_effective_security_weights_thb.csv",
        "description": "Latest effective security weights for PIT daily exposure strategy.",
    },
    {
        "source": ("..", "dynamic_port_opt", "result", "pit_reselect_step2_5_latest_effective_sleeve_weights_thb.csv"),
        "target": "pit_reselect_step2_5_latest_effective_sleeve_weights_thb.csv",
        "description": "Latest effective sleeve weights for PIT daily exposure strategy.",
    },
    {
        "source": ("..", "dynamic_port_opt", "result", "mean_covariance_stock_cap_sweep_daily_exposure_curves.csv"),
        "target": "mean_covariance_stock_cap_sweep_daily_exposure_curves.csv",
        "description": "Step 2.3b-4 mean-covariance Gold30 stock-cap 8 asset-level daily-exposure curves copied for standalone review.",
    },
    {
        "source": ("..", "dynamic_port_opt", "result", "mean_covariance_stock_cap_sweep_daily_exposure_summary.csv"),
        "target": "mean_covariance_stock_cap_sweep_daily_exposure_summary.csv",
        "description": "Step 2.3b-4 mean-covariance Gold30 stock-cap sweep summary copied for standalone review.",
    },
    {
        "source": ("..", "dynamic_port_opt", "result", "mean_covariance_gold30_asset_daily_latest_effective_weights.csv"),
        "target": "mean_covariance_gold30_asset_daily_latest_effective_weights.csv",
        "description": "Step 2.3b handoff recommended latest effective security weights.",
    },
    {
        "source": ("..", "dynamic_port_opt", "result", "mean_covariance_gold30_asset_daily_recheck_today_meta.csv"),
        "target": "mean_covariance_gold30_asset_daily_recheck_today_meta.csv",
        "description": "Step 2.3b latest effective weight metadata.",
    },
    {
        "source": ("..", "dynamic_port_opt", "result", "mean_covariance_gold30_asset_daily_recheck_today_sleeve_weights.csv"),
        "target": "mean_covariance_gold30_asset_daily_recheck_today_sleeve_weights.csv",
        "description": "Step 2.3b latest effective sleeve weights.",
    },
    {
        "source": ("..", "dynamic_port_opt", "result", "us_th_tactical_perf_momentum_gold_crash_protection_sweep_thb.csv"),
        "target": "us_th_tactical_perf_momentum_gold_crash_protection_sweep_thb.csv",
        "description": "US/TH Tactical Final Best Sharpe summary copied for standalone review.",
    },
    {
        "source": ("..", "dynamic_port_opt", "result", "us_th_tactical_perf_momentum_gold_crash_protection_sweep_curves_thb.csv"),
        "target": "us_th_tactical_perf_momentum_gold_crash_protection_sweep_curves_thb.csv",
        "description": "US/TH Tactical Final Best Sharpe curves copied for standalone review.",
    },
    {
        "source": ("..", "dynamic_port_opt", "result", "us_th_tactical_perf_momentum_gold_crash_protection_sweep_weight_history_thb.csv"),
        "target": "us_th_tactical_perf_momentum_gold_crash_protection_sweep_weight_history_thb.csv",
        "description": "US/TH Tactical Final Best Sharpe historical effective sleeve weights.",
    },
    {
        "source": ("..", "dynamic_port_opt", "result", "us_th_tactical_perf_momentum_one_model_gold30_btc10_th_signal_asym_group_cap_grid_us70_80_th30_40_summary_thb.csv"),
        "target": "us_th_tactical_perf_momentum_one_model_gold30_btc10_th_signal_asym_group_cap_grid_us70_80_th30_40_summary_thb.csv",
        "description": "One-model US70/TH30 asymmetric group-cap grid summary copied for standalone review.",
    },
    {
        "source": ("..", "dynamic_port_opt", "result", "us_th_tactical_perf_momentum_one_model_gold30_btc10_th_signal_asym_group_cap_grid_us70_80_th30_40_curves_thb.csv"),
        "target": "us_th_tactical_perf_momentum_one_model_gold30_btc10_th_signal_asym_group_cap_grid_us70_80_th30_40_curves_thb.csv",
        "description": "One-model US70/TH30 asymmetric group-cap grid curves copied for standalone review.",
    },
    {
        "source": ("..", "dynamic_port_opt", "result", "us_th_tactical_perf_momentum_one_model_gold30_btc10_th_signal_asym_group_cap_grid_us70_80_th30_40_grouped_weight_history_thb.csv"),
        "target": "us_th_tactical_perf_momentum_one_model_gold30_btc10_th_signal_asym_group_cap_grid_us70_80_th30_40_grouped_weight_history_thb.csv",
        "description": "One-model US70/TH30 asymmetric group-cap grid grouped weight history.",
    },
    {
        "source": ("..", "dynamic_port_opt", "result", "us_th_one_model_us70_th30_concentration_sweep_summary_thb.csv"),
        "target": "us_th_one_model_us70_th30_concentration_sweep_summary_thb.csv",
        "description": "US70/TH30 one-model stockcap5 penalty0.02 assets50 concentration sweep summary.",
    },
    {
        "source": ("..", "dynamic_port_opt", "result", "us_th_one_model_us70_th30_concentration_sweep_curves_thb.csv"),
        "target": "us_th_one_model_us70_th30_concentration_sweep_curves_thb.csv",
        "description": "US70/TH30 one-model stockcap5 penalty0.02 assets50 concentration sweep curves.",
    },
    {
        "source": ("..", "dynamic_port_opt", "result", "us_th_one_model_us70_th30_concentration_sweep_latest_weights_thb.csv"),
        "target": "us_th_one_model_us70_th30_concentration_sweep_latest_weights_thb.csv",
        "description": "US70/TH30 one-model stockcap5 penalty0.02 assets50 concentration sweep latest historical effective weights.",
    },
    {
        "source": ("..", "dynamic_port_opt", "result", "us_th_one_model_us70_th30_concentration_sweep_concentration_history_thb.csv"),
        "target": "us_th_one_model_us70_th30_concentration_sweep_concentration_history_thb.csv",
        "description": "US70/TH30 one-model stockcap5 penalty0.02 assets50 concentration diagnostics.",
    },
    {
        "source": ("result", "us_th_tactical_perf_momentum_603010_latest_effective_security_weights_thb.csv"),
        "target": "us_th_tactical_perf_momentum_603010_latest_effective_security_weights_thb.csv",
        "description": "Latest effective weights for Tactical TH/Gold/BTC 60/30/10 asset-level daily exposure.",
    },
    {
        "source": ("..", "dynamic_port_opt", "result", "us_th_jp_optimized_sleeve_sweep_focus_summary_thb.csv"),
        "target": "us_th_jp_optimized_sleeve_sweep_focus_summary_thb.csv",
        "description": "US/TH/JP JP-optimized sleeve selected-candidate summary.",
    },
    {
        "source": ("..", "dynamic_port_opt", "result", "us_th_jp_optimized_sleeve_sweep_curves_thb.csv"),
        "target": "us_th_jp_optimized_sleeve_sweep_curves_thb.csv",
        "description": "US/TH/JP JP-optimized sleeve curves.",
    },
    {
        "source": ("..", "dynamic_port_opt", "result", "us_th_jp_optimized_sleeve_sweep_weight_history_thb.csv"),
        "target": "us_th_jp_optimized_sleeve_sweep_weight_history_thb.csv",
        "description": "US/TH/JP JP-optimized sleeve top-level weight history.",
    },
    {
        "source": ("..", "dynamic_port_opt", "result", "us_th_jp_optimized_sleeve_sweep_jp_internal_weight_history.csv"),
        "target": "us_th_jp_optimized_sleeve_sweep_jp_internal_weight_history.csv",
        "description": "US/TH/JP JP-optimized sleeve internal Japan stock weights.",
    },
]


def _repo_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _source_path_info(parts: tuple[str, ...], target: str | None = None) -> dict[str, str | Path]:
    original = _repo_path(*parts)
    filename = Path(*parts).name
    candidates = [
        ("result", _repo_path("result", filename)),
        ("precomputed", OUT_DIR / (target or filename)),
        ("configured", original),
    ]
    for source_type, candidate in candidates:
        if candidate.exists():
            return {
                "path": candidate,
                "source_type": source_type,
                "configured_path": str(Path(*parts)),
                "actual_path": _display_path(candidate),
            }
    return {
        "path": original,
        "source_type": "missing",
        "configured_path": str(Path(*parts)),
        "actual_path": str(Path(*parts)),
    }


def _is_legacy_lookahead_series(name: str) -> bool:
    return name in LOOKAHEAD_SERIES_NAMES or any(name.startswith(prefix) for prefix in LOOKAHEAD_SERIES_PREFIXES)


def _read_curve_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    frame.index.name = "Date"
    return frame.sort_index()


def _read_curve_column(path: Path, column: str) -> pd.Series:
    frame = _read_curve_csv(path)
    if column not in frame.columns:
        raise KeyError(f"{column!r} was not found in {path}")
    return frame[column].dropna().astype(float).sort_index()


def _copy_support_files() -> list[dict[str, str]]:
    copied: list[dict[str, str]] = []
    for item in HANDOFF_SUPPORT_FILES:
        source_info = _source_path_info(item["source"], str(item["target"]))
        source = source_info["path"]
        target = OUT_DIR / str(item["target"])
        if not source.exists():
            continue
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        copied.append(
            {
                "source": str(source_info["actual_path"]),
                "configured_source": str(source_info["configured_path"]),
                "source_type": str(source_info["source_type"]),
                "target": str(target.relative_to(PROJECT_ROOT)),
                "description": str(item["description"]),
            }
        )
    return copied


def _mirror_handoff_curve_sources() -> list[dict[str, str]]:
    mirrored: list[dict[str, str]] = []
    for source in HANDOFF_CURVE_SOURCES:
        configured = _repo_path(*source["path"])
        target = OUT_DIR / Path(*source["path"]).name
        if target.exists() or not configured.exists():
            continue
        shutil.copy2(configured, target)
        mirrored.append(
            {
                "strategy": str(source["label"]),
                "source": str(Path(*source["path"])),
                "target": str(target.relative_to(PROJECT_ROOT)),
            }
        )
    return mirrored


def _latest_sp_trend_exposure() -> dict[str, object]:
    overlay_path = PROJECT_ROOT / "data" / "cache" / "dynamic_factor_copula" / "overlay_compare_prices.parquet"
    if not overlay_path.exists():
        return {}
    overlay = pd.read_parquet(overlay_path).sort_index()
    if "SPY" not in overlay.columns:
        return {}
    price = overlay["SPY"].dropna().astype(float).sort_index()
    if price.empty:
        return {}
    ma_period = 300
    below_exposure = 0.50
    min_periods = max(20, int(ma_period * 0.20))
    ma = price.rolling(ma_period, min_periods=min_periods).mean()
    close_signal = pd.Series(1.0, index=price.index, dtype=float)
    close_signal.loc[price < ma] = below_exposure
    close_signal.loc[ma.isna()] = 1.0
    effective = close_signal.shift(1).ffill().fillna(1.0)
    effective_date = effective.index.max()
    source_candidates = close_signal.index[close_signal.index < effective_date]
    source_date = source_candidates.max() if len(source_candidates) else effective_date
    return {
        "exposure": float(effective.loc[effective_date]),
        "effective_date": pd.Timestamp(effective_date).date().isoformat(),
        "source_close_date": pd.Timestamp(source_date).date().isoformat(),
        "latest_cache_trading_date": pd.Timestamp(price.index.max()).date().isoformat(),
        "rule": f"S&P trend MA{ma_period} below{below_exposure:.2f}",
    }


def _refresh_latest_effective_weight_files() -> None:
    exposure = _latest_sp_trend_exposure()
    if not exposure:
        return
    exposure_value = float(exposure["exposure"])
    security_path = OUT_DIR / "pit_reselect_step2_5_latest_effective_security_weights_thb.csv"
    if security_path.exists():
        security = pd.read_csv(security_path)
        if not security.empty:
            if "Raw Step 2.4 Weight" not in security.columns:
                old_exposure = pd.to_numeric(security.get("Daily Exposure", 1.0), errors="coerce").replace(0.0, np.nan)
                security["Raw Step 2.4 Weight"] = pd.to_numeric(security["Effective Weight"], errors="coerce").div(old_exposure).fillna(0.0)
            raw_weight = pd.to_numeric(security["Raw Step 2.4 Weight"], errors="coerce").fillna(0.0)
            security["Daily Exposure"] = exposure_value
            security["Effective Weight"] = raw_weight * exposure_value
            security["Effective Weight %"] = security["Effective Weight"] * 100.0
            security["Raw Step 2.4 Weight %"] = raw_weight * 100.0
            security["Last Exposure Date"] = str(exposure["effective_date"])
            security["Signal Source Close Date"] = str(exposure["source_close_date"])
            security["Latest Cache Trading Date"] = str(exposure["latest_cache_trading_date"])
            security["Exposure Source Column"] = "S&P trend"
            security["Daily Exposure Variant"] = "S&P trend"
            security["Latest Cache Note"] = (
                "Daily exposure is recalculated locally from the latest cached SPY close. "
                "Because the rule is lag-1, the effective exposure date uses the prior close signal."
            )
            security.to_csv(security_path, index=False)

    sleeve_path = OUT_DIR / "pit_reselect_step2_5_latest_effective_sleeve_weights_thb.csv"
    if sleeve_path.exists():
        sleeve = pd.read_csv(sleeve_path)
        if not sleeve.empty:
            if "Raw Step 2.4 Weight" not in sleeve.columns:
                sleeve["Raw Step 2.4 Weight"] = pd.to_numeric(sleeve.get("Effective Weight", 0.0), errors="coerce").fillna(0.0)
            raw_weight = pd.to_numeric(sleeve["Raw Step 2.4 Weight"], errors="coerce").fillna(0.0)
            sleeve["Effective Weight"] = raw_weight * exposure_value
            sleeve["Effective Weight %"] = sleeve["Effective Weight"] * 100.0
            sleeve["Date"] = str(exposure["effective_date"])
            sleeve["Signal Source Close Date"] = str(exposure["source_close_date"])
            sleeve["Daily Exposure"] = exposure_value
            sleeve["Daily Exposure Variant"] = "S&P trend"
            sleeve["Exposure Source Column"] = "S&P trend"
            sleeve.to_csv(sleeve_path, index=False)


def _curve_from_returns(returns: pd.Series) -> pd.Series:
    clean = returns.dropna()
    curve = pd.Series(np.nan, index=returns.index, dtype=float)
    if clean.empty:
        return curve
    curve.loc[clean.index] = INITIAL_VALUE * (1.0 + clean.fillna(0.0)).cumprod()
    return curve


def _returns_from_curve(curve: pd.Series) -> pd.Series:
    return curve.astype(float).pct_change(fill_method=None).fillna(0.0)


def _align_return_starts(returns: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    first_dates = []
    for column in returns.columns:
        valid = returns[column].dropna()
        if not valid.empty:
            first_dates.append(valid.index.min())
    if not first_dates:
        return returns, ""
    common_start = max(first_dates)
    aligned = returns.loc[returns.index >= common_start].copy()
    if not aligned.empty:
        aligned.iloc[0] = 0.0
    return aligned, pd.Timestamp(common_start).date().isoformat()


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


def _read_source_metrics(paths: list[Path]) -> dict[str, dict[str, object]]:
    source_metrics: dict[str, dict[str, object]] = {}
    metric_columns = [
        "Total Return",
        "CAGR",
        "Annual Vol",
        "Sharpe",
        "Sortino",
        "Max Drawdown",
        "Hit Rate",
        "Start",
        "End",
    ]
    for path in paths:
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if "Strategy" in frame.columns:
            frame = frame.set_index("Strategy")
        elif frame.columns[0].startswith("Unnamed"):
            frame = frame.set_index(frame.columns[0])
        else:
            continue
        for strategy, row in frame.iterrows():
            if _is_legacy_lookahead_series(str(strategy)):
                continue
            source_metrics.setdefault(str(strategy), {
                column: row[column]
                for column in metric_columns
                if column in row.index and pd.notna(row[column])
            })
    return source_metrics


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
            pd.Series(np.where(spy_thb < ma200, 0.50, 1.0), index=spy_thb.index),
            pd.Series(np.where(drawdown <= -0.08, 0.35, 1.0), index=spy_thb.index),
            pd.Series(np.where(drawdown <= -0.15, 0.15, 1.0), index=spy_thb.index),
            pd.Series(np.where(vix.reindex(spy_thb.index).ffill() >= 28.0, 0.35, 1.0), index=spy_thb.index),
            pd.Series(np.where(vix.reindex(spy_thb.index).ffill() >= 35.0, 0.15, 1.0), index=spy_thb.index),
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
        active = drop_duplicate_share_classes_available(
            (ticker for ticker in _active_members(intervals, rebalance_date) if ticker in prices.columns),
            prices.columns,
        )
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
    spy_exposure = _sp_daily_exposure(spy_thb, vix).shift(1).ffill().fillna(1.0)
    gold_exposure = _trend_exposure(gold_thb, 0.25).shift(1).ffill().fillna(1.0)
    btc_exposure = _trend_exposure(btc_thb, 0.00).shift(1).ffill().fillna(1.0)

    sleeve_returns = pd.DataFrame(
        {
            "SPY": spy_thb.pct_change(fill_method=None).fillna(0.0),
            "SPY_DAILY_EXPOSURE": spy_thb.pct_change(fill_method=None).fillna(0.0) * spy_exposure,
            "GOLD": gold_thb.pct_change(fill_method=None).fillna(0.0),
            "GOLD_DAILY_EXPOSURE": gold_thb.pct_change(fill_method=None).fillna(0.0) * gold_exposure,
            "BTC": btc_thb.pct_change(fill_method=None).fillna(0.0),
            "BTC_DAILY_EXPOSURE": btc_thb.pct_change(fill_method=None).fillna(0.0) * btc_exposure,
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
        "S&P Gold BTC 80/10/10": _quarterly_rebalanced_returns(
            sleeve_returns[["SPY", "GOLD", "BTC"]],
            {"SPY": 0.80, "GOLD": 0.10, "BTC": 0.10},
        ),
        "S&P BTC 85/0/15": _quarterly_rebalanced_returns(
            sleeve_returns[["SPY", "BTC"]],
            {"SPY": 0.85, "BTC": 0.15},
        ),
        "S&P Gold BTC daily exposure 60/30/10": _quarterly_rebalanced_returns(
            sleeve_returns[["SPY_DAILY_EXPOSURE", "GOLD_DAILY_EXPOSURE", "BTC_DAILY_EXPOSURE"]],
            {"SPY_DAILY_EXPOSURE": 0.60, "GOLD_DAILY_EXPOSURE": 0.30, "BTC_DAILY_EXPOSURE": 0.10},
        ),
        "S&P Gold BTC daily exposure 80/10/10": _quarterly_rebalanced_returns(
            sleeve_returns[["SPY_DAILY_EXPOSURE", "GOLD_DAILY_EXPOSURE", "BTC_DAILY_EXPOSURE"]],
            {"SPY_DAILY_EXPOSURE": 0.80, "GOLD_DAILY_EXPOSURE": 0.10, "BTC_DAILY_EXPOSURE": 0.10},
        ),
        "S&P Gold BTC daily exposure 70/20/10": _quarterly_rebalanced_returns(
            sleeve_returns[["SPY_DAILY_EXPOSURE", "GOLD_DAILY_EXPOSURE", "BTC_DAILY_EXPOSURE"]],
            {"SPY_DAILY_EXPOSURE": 0.70, "GOLD_DAILY_EXPOSURE": 0.20, "BTC_DAILY_EXPOSURE": 0.10},
        ),
        "S&P BTC daily exposure 85/0/15": _quarterly_rebalanced_returns(
            sleeve_returns[["SPY_DAILY_EXPOSURE", "BTC_DAILY_EXPOSURE"]],
            {"SPY_DAILY_EXPOSURE": 0.85, "BTC_DAILY_EXPOSURE": 0.15},
        ),
    }

    if QQQ_COMBO_PRICE_CACHE.exists():
        qqq_combo_prices = pd.read_parquet(QQQ_COMBO_PRICE_CACHE).sort_index().ffill()
        qqq_combo_ret = qqq_combo_daily_returns(qqq_combo_prices).loc[overlay.index.min() : overlay.index.max()]
        strategy_returns[QQQ_COMBO_STRATEGY] = qqq_combo_ret

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
        _repo_path("result", "gold_btc_sp500_overlay", "equity_curves.csv"),
        _repo_path("..", "dynamic_port_opt", "result", "joint_confirm_603010_504d_1m_overlay_curves_thb.csv"),
        _repo_path("data", "precomputed", "us_th_tactical_perf_momentum_gold_btc_overlay_curves_thb.csv"),
        _repo_path("result", "us_th_side_trigger_reallocation_curves_thb.csv"),
        _repo_path("result", "strategy_b_weekly_exposure_test_curves.csv"),
        _repo_path("result", "us_th_stocks_only_vs_gold_btc_side_trigger_curves_thb.csv"),
        _repo_path("result", "us_th_best_asset_sweep_fee_realloc_curves_thb.csv"),
        _repo_path("..", "dynamic_port_opt", "result", "us_th_joint_model_curves_thb.csv"),
        _repo_path("..", "dynamic_port_opt", "result", "us_th_gold_btc_blended_curves_thb.csv"),
        _repo_path("..", "dynamic_port_opt", "result", "us_th_side_trigger_reallocation_curves_thb.csv"),
        _repo_path("..", "dynamic_port_opt", "result", "us_th_stocks_only_vs_gold_btc_side_trigger_curves_thb.csv"),
    ]
    curve_renames = {
        "S&P overlay + Gold/BTC": "S&P overlay + Gold/BTC 80/10/10",
        "Tactical TH/Gold/BTC 60/30/10 asset-level daily exposure (Tactical TH proxy_regime relative_return binary lb1 cap30 entry0% exit0% hold0 confirm1)": "Tactical TH/Gold/BTC 60/30/10 asset-level daily exposure",
    }
    for path in curve_sources:
        if not path.exists():
            continue
        curves = _read_curve_csv(path).loc[: overlay.index.max()]
        for column in curves.columns:
            name = curve_renames.get(column, column)
            if _is_legacy_lookahead_series(name):
                continue
            if _is_excluded_strategy_name(name):
                continue
            if name not in strategy_returns:
                strategy_returns[name] = _returns_from_curve(curves[column])

    mirrored_handoff_curve_sources = _mirror_handoff_curve_sources()

    handoff_sources_used: list[dict[str, str]] = []
    for source in HANDOFF_CURVE_SOURCES:
        source_info = _source_path_info(source["path"])
        path = source_info["path"]
        if not path.exists():
            continue
        curve = _read_curve_column(path, source["column"])
        strategy_returns[source["label"]] = _returns_from_curve(curve)
        handoff_sources_used.append(
            {
                "strategy": source["label"],
                "family": source["family"],
                "path": str(source_info["actual_path"]),
                "configured_path": str(source_info["configured_path"]),
                "source_type": str(source_info["source_type"]),
                "column": source["column"],
                "source_start": curve.index.min().date().isoformat(),
                "source_end": curve.index.max().date().isoformat(),
            }
        )

    returns = pd.DataFrame(strategy_returns).sort_index().loc[: overlay.index.max()]
    returns = returns.loc[:, [column for column in returns.columns if not _is_excluded_strategy_name(column)]]
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
    canonical_path = OUT_DIR / "us_th_tactical_perf_momentum_gold_btc_overlay_summary_thb.csv"
    canonical_label = "Tactical TH/Gold/BTC 60/30/10 asset-level daily exposure"
    if canonical_path.exists():
        canonical = pd.read_csv(canonical_path)
        canonical_row = canonical.loc[canonical["Strategy"].str.startswith(f"{canonical_label} (")].iloc[0]
        target = summary["Strategy"].eq(canonical_label)
        for column in ["Start", "End", "Total Return", "CAGR", "Annual Vol", "Sharpe", "Sortino", "Max Drawdown", "Hit Rate"]:
            summary.loc[target, column] = canonical_row[column]
    summary.to_csv(OUT_DIR / "streamlit_10y_strategy_summary.csv", index=False)

    support_files = _copy_support_files()
    _refresh_latest_effective_weight_files()

    metadata = {
        "generated_at": pd.Timestamp.now(tz="Asia/Bangkok").isoformat(),
        "data_start": returns.dropna(how="all").index.min().date().isoformat(),
        "raw_data_start": overlay.index.min().date().isoformat(),
        "data_end": overlay.index.max().date().isoformat(),
        "currency": "THB",
        "notes": [
            "This is a frozen deploy-friendly performance dataset.",
            "It stores strategy/sleeve return series, not raw stock-level cache.",
            "Latest rebalance weights are intentionally not included.",
            "Strategy return series keep their own available history; the app trims only the selected chart pair to a shared date window.",
        ],
        "handoff_curve_sources": handoff_sources_used,
        "mirrored_handoff_curve_sources": mirrored_handoff_curve_sources,
        "handoff_support_files": support_files,
        "strategy_descriptions": {
            QQQ_COMBO_STRATEGY: {
                "Strategy setup": [
                    "Base allocation: QQQ core 40%, global country/sector rotation 30%, defensive GLD/IEF/TLT 30%.",
                    "Universe: QQQ, country ETFs, US sector/industry/commodity ETFs, GLD, IEF, TLT, and BIL as cash proxy.",
                    "Selection rules: rotation sleeve reselects monthly top 3 ETFs by 6-month momentum, only among ETFs above MA200.",
                    "Optimizer/model: rotation and defensive sleeves use 126-day inverse-vol risk parity; QQQ core is fixed.",
                    "Objective: durable risk-adjusted return with lower drawdown than QQQ buy-and-hold, net of 17 bps turnover cost.",
                    "Rebalance schedule: monthly rotation selection; structural sleeve weights stay 40/30/30.",
                    "Caps: sleeve caps are fixed at Core 40%, Rotation 30%, Defensive 30%; no extra single-ETF cap beyond sleeve construction.",
                    "Latest-weight source: standalone refresh from this repo's yfinance-backed qqq_combo_gtaa cache, not static files from dynamic_port_opt or webull_api.",
                ],
                "Daily exposure rules": [
                    "Uses daily exposure overlay on every holding.",
                    "Signal timing: close signal is shifted lag-1, so today's exposure uses the prior session signal for next-session execution.",
                    "Signal: each asset uses its own MA200; asset is risk-on when close is above MA200.",
                    "Threshold: below MA200 reduces that asset sleeve slice to 0% exposure.",
                    "Reduced exposure goes to BIL via Cash / Reduced Exposure.",
                ],
            }
        },
        "files": {
            "returns": "data/precomputed/streamlit_10y_strategy_returns.parquet",
            "curves": "data/precomputed/streamlit_10y_strategy_curves.parquet",
            "sleeves": "data/precomputed/streamlit_10y_sleeve_returns.parquet",
            "summary": "data/precomputed/streamlit_10y_strategy_summary.csv",
        },
    }
    (OUT_DIR / "streamlit_10y_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print("\nWrote precomputed dataset to data/precomputed")


if __name__ == "__main__":
    main()
