from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DYNAMIC_ROOT = ROOT.parent / "dynamic_port_opt"
for path in [DYNAMIC_ROOT / "scripts", DYNAMIC_ROOT / "src"]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_cap5_10y_june as june  # type: ignore  # noqa: E402
import run_us_th_one_model_us70_th30_concentration_sweep as base  # type: ignore  # noqa: E402

STRATEGY = "One-model US cap 70% / TH cap 30% stockcap5 penalty0.02 assets50 + daily exposure"
CASE_LABEL = "Stock cap 5%"
RESULT_PREFIX = "us_th_one_model_us70_th30_stockcap5_penalty002_assets50_weekday_june"
INITIAL_VALUE = base.INITIAL_VALUE
OUT_DIRS = [ROOT / "result", ROOT / "data" / "precomputed"]
YFINANCE_CACHE_DIR = ROOT / ".yfinance_cache_weekday_june"


def _metrics(curve: pd.Series) -> dict[str, object]:
    clean = curve.dropna().astype(float)
    row = base.compute_port_opt_style_metrics(clean, risk_free_rate=base.RISK_FREE_RATE).to_dict()
    row.update(
        {
            "Strategy": STRATEGY,
            "Case": CASE_LABEL,
            "Start": clean.index.min().date().isoformat(),
            "End": clean.index.max().date().isoformat(),
            "US Group Cap": base.US_GROUP_CAP,
            "TH Group Cap": base.TH_GROUP_CAP,
            "Stock Cap": 0.05,
            "Concentration Penalty": 0.02,
            "US Assets": 50,
            "TH Assets": 50,
            "Gold Cap": base.GOLD_CAP,
            "BTC Cap": base.BTC_CAP,
            "Exposure": "daily exposure",
        }
    )
    return row


def _period_compare(curve: pd.Series) -> pd.DataFrame:
    end = curve.dropna().index.max()
    rows: list[dict[str, object]] = []
    for period, years in [("Full period", None), ("10Y", 10), ("5Y", 5), ("3Y", 3), ("1Y", 1), ("2026 YTD", None)]:
        if period == "2026 YTD":
            start = pd.Timestamp("2026-01-01")
        else:
            start = curve.index.min() if years is None else end - pd.DateOffset(years=years)
        sample = curve.dropna().loc[lambda s: s.index >= start]
        if len(sample) < 2:
            continue
        row = base.compute_port_opt_style_metrics(sample, risk_free_rate=base.RISK_FREE_RATE).to_dict()
        row.update(
            {
                "Period": period,
                "Strategy": STRATEGY,
                "Start": sample.index.min().date().isoformat(),
                "End": sample.index.max().date().isoformat(),
                "Observations": int(sample.shape[0]),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    YFINANCE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    june.yf.set_tz_cache_location(str(YFINANCE_CACHE_DIR))

    p2, v2, b2, vp2, us_all, th_all = june._extend()
    keep = p2.index[p2.index.weekday < 5]
    p2 = p2.reindex(keep).ffill()
    v2 = v2.reindex(keep).fillna(0.0)
    b2 = b2.reindex(keep).ffill()
    vp2 = vp2.reindex(keep).ffill()
    _, th_signal = base._load_tactical_th_signal(p2.index)

    case = base.SweepCase(CASE_LABEL, 0.05, 0.02, 50, 50)
    raw_curve, raw_weights, selected, returns = base._run_case(case, p2, v2, b2, vp2, us_all, th_all, th_signal)
    exposure = base._proxy_exposure(raw_curve.index)
    effective = base._apply_proxy_exposure(raw_weights, exposure)
    asset_cols = [column for column in effective.columns if column in returns.columns]
    gross_returns = returns.reindex(raw_curve.index)[asset_cols].fillna(0.0).mul(effective[asset_cols], axis=1).sum(axis=1)
    curve = base.curve_from_returns(gross_returns, initial=INITIAL_VALUE).rename(STRATEGY)

    latest_date = effective.dropna(how="all").index.max()
    latest = effective.loc[latest_date].dropna()
    latest = latest[latest.abs() > 1e-12].sort_values(ascending=False)
    latest_df = latest.rename("Effective Weight").reset_index().rename(columns={"index": "Asset"})
    latest_df["Date"] = pd.Timestamp(latest_date).date().isoformat()
    latest_df["Strategy"] = STRATEGY
    latest_df["Case"] = CASE_LABEL

    effective_out = effective.copy().reset_index(names="Date")
    effective_out.insert(1, "Strategy", STRATEGY)
    effective_out.insert(2, "Case", CASE_LABEL)

    summary = pd.DataFrame([_metrics(curve)])
    periods = _period_compare(curve)
    curves = curve.to_frame()
    selected = selected.copy()
    if not selected.empty:
        selected["Strategy"] = STRATEGY
        selected["Case"] = CASE_LABEL

    for out_dir in OUT_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out_dir / f"{RESULT_PREFIX}_summary_thb.csv", index=False)
        periods.to_csv(out_dir / f"{RESULT_PREFIX}_period_compare_thb.csv", index=False)
        curves.to_csv(out_dir / f"{RESULT_PREFIX}_curves_thb.csv")
        latest_df.to_csv(out_dir / f"{RESULT_PREFIX}_latest_weights_thb.csv", index=False)
        effective_out.to_csv(out_dir / f"{RESULT_PREFIX}_effective_weights_thb.csv", index=False)
        selected.to_csv(out_dir / f"{RESULT_PREFIX}_universe_history_thb.csv", index=False)

    print(summary[["Strategy", "Start", "End", "CAGR", "Sharpe", "Max Drawdown"]].to_string(index=False))
    print(periods[["Period", "Start", "End", "CAGR", "Sharpe", "Max Drawdown"]].to_string(index=False))
    print(latest_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()