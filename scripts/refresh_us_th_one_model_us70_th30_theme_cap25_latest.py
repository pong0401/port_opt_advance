from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import refresh_us_th_tactical_one_model_us70_th30_latest as refresh
from share_class_utils import drop_duplicate_share_classes_available


THEME_CAP = 0.25
STRICT_AI_TECH_BUCKET = {
    "AAPL",
    "AMD",
    "GOOG",
    "GOOGL",
    "INTC",
    "MU",
    "NVDA",
    "QCOM",
    "TXN",
}

refresh.STRATEGY = "One-model US cap 70% / TH cap 30% stockcap5 penalty0.02 assets50 AI-tech cap 25% + daily exposure"
refresh.CASE = "US cap 70% / TH cap 30% stockcap5 penalty0.02 assets50 AI-tech cap 25%"
refresh.RESULT_PREFIX = "us_th_one_model_us70_th30_stockcap5_penalty002_assets50_ai_tech_cap25"
refresh.STOCK_CAP = 0.05
refresh.US_ASSETS = 50
refresh.TH_ASSETS = 50
refresh.CONCENTRATION_PENALTY = 0.02

_original_write_outputs = refresh._write_outputs


def _optimize_one_model_with_theme_cap(
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

    features = refresh.compute_feature_table(
        train_returns,
        benchmark.pct_change(fill_method=None).reindex(train_returns.index),
        vol_proxy.pct_change(fill_method=None).reindex(train_returns.index),
        prices.reindex(train_returns.index)[selected],
        include_momentum_features=True,
        feature_flags=refresh.FEATURE_FLAGS,
    )
    momentum = refresh.build_momentum_signal(features, mode="mom_63").reindex(selected)
    mu = momentum.fillna(momentum.median() if momentum.notna().any() else 0.0).to_numpy(dtype=float)
    if len(mu):
        mu = np.clip(mu, np.nanpercentile(mu, 10), np.nanpercentile(mu, 90))

    cov = train_returns.cov().reindex(index=selected, columns=selected).fillna(0.0)
    cov_matrix = cov.to_numpy(dtype=float)
    caps = pd.Series(refresh.STOCK_CAP, index=selected, dtype=float)
    caps.loc[[asset for asset in selected if asset == "GC=F"]] = refresh.GOLD_CAP
    caps.loc[[asset for asset in selected if asset == "BTC-USD"]] = refresh.BTC_CAP
    if float(caps.sum()) < 1.0 - 1e-12:
        raise RuntimeError("One-model caps are infeasible; caps sum below 100%.")

    x0 = caps / caps.sum()
    bounds = [(0.0, float(caps.loc[asset])) for asset in selected]
    constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1.0}]
    us_idx = [i for i, asset in enumerate(selected) if asset in us_assets]
    th_idx = [i for i, asset in enumerate(selected) if asset in th_assets]
    theme_idx = [i for i, asset in enumerate(selected) if asset in STRICT_AI_TECH_BUCKET]
    if us_idx:
        constraints.append({"type": "ineq", "fun": lambda x, idx=us_idx: refresh.US_GROUP_CAP - float(np.sum(x[idx]))})
    if th_idx:
        constraints.append({"type": "ineq", "fun": lambda x, idx=th_idx: refresh.TH_GROUP_CAP - float(np.sum(x[idx]))})
    if theme_idx:
        constraints.append({"type": "ineq", "fun": lambda x, idx=theme_idx: THEME_CAP - float(np.sum(x[idx]))})

    def objective(x: np.ndarray) -> float:
        variance = float(x @ cov_matrix @ x)
        expected = float(mu @ x)
        concentration = float(np.sum(np.square(x)))
        return 0.5 * refresh.RISK_AVERSION * variance - expected + refresh.CONCENTRATION_PENALTY * concentration

    result = minimize(objective, x0=x0.to_numpy(dtype=float), bounds=bounds, constraints=constraints, method="SLSQP")
    if not result.success:
        weights = x0.copy()
    else:
        weights = pd.Series(result.x, index=selected).clip(lower=0.0)
    return (weights / weights.sum()).sort_values(ascending=False)


def _write_outputs_with_theme_meta(security: pd.DataFrame, sleeve: pd.DataFrame, meta: pd.DataFrame) -> None:
    theme_assets = sorted(STRICT_AI_TECH_BUCKET)
    if not meta.empty:
        meta = meta.copy()
        effective = pd.to_numeric(security.get("Effective Weight", 0.0), errors="coerce").fillna(0.0)
        raw = pd.to_numeric(security.get("Raw Optimizer Weight", 0.0), errors="coerce").fillna(0.0)
        asset = security.get("Asset", pd.Series(dtype=str)).astype(str)
        theme_mask = asset.isin(STRICT_AI_TECH_BUCKET)
        meta["AI-Tech Theme Cap"] = THEME_CAP
        meta["AI-Tech Bucket"] = ", ".join(theme_assets)
        meta["Latest AI-Tech Effective Weight"] = float(effective.loc[theme_mask].sum())
        meta["Latest AI-Tech Raw Optimizer Weight"] = float(raw.loc[theme_mask].sum())
    _original_write_outputs(security, sleeve, meta)


refresh._optimize_one_model = _optimize_one_model_with_theme_cap
refresh._write_outputs = _write_outputs_with_theme_meta


if __name__ == "__main__":
    refresh.main()
