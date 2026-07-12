from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "data" / "cache" / "dynamic_factor_copula"
YF_CACHE_DIR = CACHE_DIR / ".yfinance"
PRICE_CACHE = CACHE_DIR / "qqq_combo_gtaa_latest_prices.parquet"

STRATEGY = "QQQ + global rotation + defensive GTAA 40/30/30"
RESULT_PREFIX = "qqq_global_rotation_defensive_gtaa_403030"

START_DATE = "2000-01-01"
CORE = "QQQ"
CASH = "BIL"
COUNTRIES = (
    "EWA", "EWC", "EWG", "EWH", "EWI", "EWJ", "EWL", "EWM", "EWN", "EWP",
    "EWQ", "EWS", "EWD", "EWU", "EWW", "EWY", "EWZ", "EWT", "EEM", "EFA",
    "TUR", "THD", "NORW", "EPOL", "ARGT", "MCHI", "GREK", "INDA",
)
SECTORS = (
    "SMH", "XLK", "XLE", "XLF", "XLV", "XLI", "XLP", "XLU", "XLB", "XLY",
    "IGV", "SOXX", "VNQ", "KRE", "IYT", "GDX", "DBC",
)
DEFENSIVE = ("GLD", "IEF", "TLT")
ROTATION = COUNTRIES + SECTORS
PRICE_TICKERS = tuple(dict.fromkeys([CORE, *ROTATION, *DEFENSIVE, CASH]))

CORE_WEIGHT = 0.40
ROTATION_WEIGHT = 0.30
DEFENSIVE_WEIGHT = 0.30
TOP_N = 3
MOMENTUM_DAYS = 126
MA_DAYS = 200
RISK_PARITY_VOL_DAYS = 126
FEE_BPS = 17.0
MIN_DISPLAY_WEIGHT = 0.01


def close_series(raw: pd.DataFrame, ticker: str) -> pd.Series:
    if raw.empty:
        return pd.Series(dtype=float, name=ticker)
    close = raw["Close"] if "Close" in raw.columns else raw.get("Adj Close", pd.Series(dtype=float))
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.dropna().astype(float).rename(ticker)
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close


def download_close_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    frames: list[pd.Series] = []
    for ticker in tickers:
        raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False, threads=False)
        close = close_series(raw, ticker)
        if close.empty:
            print(f"Warning: no fresh rows for {ticker}")
            continue
        frames.append(close)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1).sort_index()


def load_prices(refresh: bool = True) -> pd.DataFrame:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    YF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(YF_CACHE_DIR))
    existing = pd.read_parquet(PRICE_CACHE).sort_index() if PRICE_CACHE.exists() else pd.DataFrame()
    if not refresh:
        if existing.empty:
            raise RuntimeError(f"{PRICE_CACHE} does not exist. Run the latest refresh first.")
        existing.index = pd.to_datetime(existing.index)
        return existing.reindex(columns=PRICE_TICKERS)
    if not existing.empty:
        existing.index = pd.to_datetime(existing.index)
        existing = existing.loc[:, ~existing.columns.duplicated(keep="last")]
        start = (pd.Timestamp(existing.index.max()).date() - timedelta(days=10)).isoformat()
    else:
        start = START_DATE
    end = (pd.Timestamp.today().date() + timedelta(days=1)).isoformat()
    update = download_close_prices(list(PRICE_TICKERS), start, end)
    if update.empty and existing.empty:
        raise RuntimeError("No rows returned from yfinance for QQQ combo GTAA latest refresh.")
    if not update.empty:
        cutoff = pd.Timestamp(update.index.min())
        historical = existing.loc[existing.index < cutoff] if not existing.empty else pd.DataFrame()
        prices = pd.concat([historical, update]).sort_index()
    else:
        prices = existing.sort_index()
    prices = prices[~prices.index.duplicated(keep="last")].reindex(columns=PRICE_TICKERS)
    prices.to_parquet(PRICE_CACHE)
    return prices


def daily_gate(prices: pd.DataFrame) -> pd.DataFrame:
    clean = prices.sort_index().ffill()
    sma = clean.rolling(MA_DAYS, min_periods=MA_DAYS).mean()
    return (clean > sma).shift(1).fillna(0.0).astype(float)


def monthly_topn(prices: pd.DataFrame) -> dict[pd.Period, list[str]]:
    clean = prices.sort_index().ffill()
    sma = clean.rolling(MA_DAYS, min_periods=MA_DAYS).mean()
    momentum = clean / clean.shift(MOMENTUM_DAYS) - 1.0
    month_ends = clean.resample("ME").last().index
    selected: dict[pd.Period, list[str]] = {}
    for date in month_ends:
        available = [
            asset for asset in ROTATION
            if asset in clean.columns
            and date in clean.index
            and pd.notna(clean.at[date, asset])
            and pd.notna(sma.at[date, asset])
            and pd.notna(momentum.at[date, asset])
            and clean.at[date, asset] > sma.at[date, asset]
        ]
        selected[date.to_period("M")] = sorted(
            available, key=lambda asset: float(momentum.at[date, asset]), reverse=True
        )[:TOP_N]
    return selected


def always(prices: pd.DataFrame, assets: tuple[str, ...]) -> dict[pd.Period, list[str]]:
    month_ends = prices.sort_index().ffill().resample("ME").last().index
    return {date.to_period("M"): [asset for asset in assets if asset in prices.columns] for date in month_ends}


def sleeve_weights(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    gate: pd.DataFrame,
    monthly_holdings: dict[pd.Period, list[str]],
    weight_mode: str,
) -> pd.DataFrame:
    weights = pd.DataFrame(0.0, index=prices.index, columns=list(prices.columns))
    daily_vol = returns.rolling(RISK_PARITY_VOL_DAYS).std()
    month = prices.index.to_period("M")
    for period, holdings in monthly_holdings.items():
        days = prices.index[month == period]
        if not holdings or len(days) == 0:
            continue
        if weight_mode == "equal":
            sleeve_weight = {asset: 1.0 / len(holdings) for asset in holdings}
        else:
            inv_vol = {}
            for asset in holdings:
                vol = daily_vol[asset].reindex(days).iloc[0]
                inv_vol[asset] = 1.0 / float(vol) if pd.notna(vol) and float(vol) > 0 else 0.0
            total = sum(inv_vol.values()) or 1.0
            sleeve_weight = {asset: inv_vol[asset] / total for asset in holdings}
        for asset in holdings:
            weights.loc[days, asset] = sleeve_weight[asset] * gate[asset].reindex(days).fillna(0.0)
    return weights


def build_weight_matrices(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    clean = prices.sort_index().ffill().reindex(columns=PRICE_TICKERS)
    returns = clean.pct_change(fill_method=None)
    gate = daily_gate(clean)
    core_weights = sleeve_weights(clean, returns, gate, always(clean, (CORE,)), "equal")
    rotation_weights = sleeve_weights(clean, returns, gate, monthly_topn(clean), "riskparity")
    defensive_weights = sleeve_weights(clean, returns, gate, always(clean, DEFENSIVE), "riskparity")
    return core_weights, rotation_weights, defensive_weights, returns


def combined_weights(prices: pd.DataFrame) -> pd.DataFrame:
    core_weights, rotation_weights, defensive_weights, _returns = build_weight_matrices(prices)
    return CORE_WEIGHT * core_weights + ROTATION_WEIGHT * rotation_weights + DEFENSIVE_WEIGHT * defensive_weights


def daily_returns(prices: pd.DataFrame, fee_bps: float = FEE_BPS) -> pd.Series:
    weights = combined_weights(prices)
    returns = prices.sort_index().ffill().pct_change(fill_method=None).reindex(columns=weights.columns)
    cash_returns = prices[CASH].sort_index().ffill().pct_change(fill_method=None).reindex(weights.index).fillna(0.0)
    cash_weight = (1.0 - weights.sum(axis=1)).clip(lower=0.0)
    gross = (weights * returns).sum(axis=1) + cash_weight * cash_returns
    turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
    net = gross - turnover * fee_bps / 10_000.0
    warmup_start = prices.index.min() + pd.Timedelta(days=260)
    return net.loc[net.index >= warmup_start].fillna(0.0).rename(STRATEGY)


def sleeve_for_asset(asset: str) -> str:
    if asset == CORE:
        return "Core"
    if asset in ROTATION:
        return "Rotation"
    if asset in DEFENSIVE:
        return "Defensive"
    if asset == CASH:
        return "BIL"
    return "Other"


def latest_rebalance_date(prices: pd.DataFrame, as_of: pd.Timestamp) -> pd.Timestamp:
    eligible = prices.loc[:as_of].dropna(how="all")
    month_ends = eligible.groupby(eligible.index.to_period("M")).tail(1).index
    month_ends = month_ends[month_ends >= eligible.index.min() + pd.DateOffset(days=252)]
    if month_ends.empty:
        raise RuntimeError("Not enough history to find latest QQQ combo rebalance date.")
    return pd.Timestamp(month_ends.max())


def latest_security_weights(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    common = prices.dropna(subset=[asset for asset in (CORE, *DEFENSIVE, CASH) if asset in prices.columns])
    if common.empty:
        raise RuntimeError("No common latest close for QQQ combo required assets.")
    as_of = pd.Timestamp(common.index.max())
    clean = prices.loc[:as_of].sort_index().ffill().reindex(columns=PRICE_TICKERS)
    weights = combined_weights(clean)
    latest = weights.loc[as_of].fillna(0.0)
    cash_weight = max(0.0, 1.0 - float(latest.sum()))
    rebalance_date = latest_rebalance_date(clean, as_of)
    gate = daily_gate(clean).loc[as_of].fillna(0.0)
    selected_rotation = monthly_topn(clean).get(rebalance_date.to_period("M"), [])
    source_candidates = clean.index[clean.index < as_of]
    source_date = pd.Timestamp(source_candidates.max() if len(source_candidates) else as_of)

    rows: list[dict[str, object]] = []
    for asset, weight in latest[latest > 1e-12].sort_values(ascending=False).items():
        rows.append(
            {
                "Asset": asset,
                "Sleeve": sleeve_for_asset(asset),
                "Target Weight": float(weight),
                "Target Weight %": float(weight) * 100.0,
                "Daily Exposure": float(gate.get(asset, 0.0)),
                "Effective Weight": float(weight),
                "Effective Weight %": float(weight) * 100.0,
                "Date": as_of.date().isoformat(),
                "Last Exposure Date": as_of.date().isoformat(),
                "Last Rebalance Date": rebalance_date.date().isoformat(),
                "Signal Source Close Date": source_date.date().isoformat(),
                "Latest Cache Trading Date": as_of.date().isoformat(),
                "Strategy": STRATEGY,
                "Daily Exposure Variant": "Per-asset MA200 close signal, lag-1 next-session execution",
            }
        )
    if cash_weight > 1e-12:
        rows.append(
            {
                "Asset": "Cash / Reduced Exposure",
                "Sleeve": "Cash / Reduced Exposure",
                "Target Weight": 0.0,
                "Target Weight %": 0.0,
                "Daily Exposure": 1.0,
                "Effective Weight": cash_weight,
                "Effective Weight %": cash_weight * 100.0,
                "Date": as_of.date().isoformat(),
                "Last Exposure Date": as_of.date().isoformat(),
                "Last Rebalance Date": rebalance_date.date().isoformat(),
                "Signal Source Close Date": source_date.date().isoformat(),
                "Latest Cache Trading Date": as_of.date().isoformat(),
                "Strategy": STRATEGY,
                "Daily Exposure Variant": "BIL cash return for MA200-gated inactive slices",
            }
        )
    latest_df = pd.DataFrame(rows).sort_values("Effective Weight", ascending=False)
    display_df = latest_df.loc[
        (latest_df["Effective Weight"].astype(float) >= MIN_DISPLAY_WEIGHT)
        | latest_df["Asset"].eq("Cash / Reduced Exposure")
    ].copy()
    sleeve_df = (
        latest_df.groupby("Sleeve", as_index=False)
        .agg(
            {
                "Target Weight": "sum",
                "Target Weight %": "sum",
                "Effective Weight": "sum",
                "Effective Weight %": "sum",
                "Daily Exposure": "mean",
                "Date": "first",
                "Strategy": "first",
            }
        )
        .sort_values("Effective Weight", ascending=False)
    )
    meta = {
        "Strategy": STRATEGY,
        "Date": as_of.date().isoformat(),
        "Latest Cache Trading Date": as_of.date().isoformat(),
        "Last Rebalance Date": rebalance_date.date().isoformat(),
        "Strategy setup": (
            "Base allocation QQQ core 40%, global country/sector rotation 30%, defensive GLD/IEF/TLT 30%; "
            "rotation selects top 3 by 6-month momentum among ETFs above MA200; rotation and defensive sleeves use 126-day inverse-vol risk parity; "
            "monthly rebalance; no single-name cap beyond sleeve weights; latest weights recomputed from this repo's qqq_combo_gtaa price cache."
        ),
        "Daily exposure rules": (
            "Every holding uses its own MA200 close signal shifted by one session. If an asset is below MA200, that asset slice is reduced to 0% "
            "and the inactive allocation earns BIL via Cash / Reduced Exposure. Signals are evaluated daily between monthly rotation rebalances."
        ),
        "Rotation Universe": ", ".join(ROTATION),
        "Defensive Universe": ", ".join(DEFENSIVE),
        "Selected Rotation": ", ".join(selected_rotation),
        "Fee Bps": FEE_BPS,
        "Latest Weight Source": "Standalone refresh from this repo's current yfinance-backed qqq_combo_gtaa cache; no static latest-weight file is read from dynamic_port_opt or webull_api.",
        "Display Rule": "Latest-weight display hides asset rows below 1% except Cash / Reduced Exposure.",
    }
    return display_df, sleeve_df, meta
