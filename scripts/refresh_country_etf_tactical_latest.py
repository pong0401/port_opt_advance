from __future__ import annotations

from datetime import timedelta
from pathlib import Path
import json

import numpy as np
import pandas as pd
import yfinance as yf


ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "data" / "cache" / "dynamic_factor_copula"
YF_CACHE_DIR = CACHE_DIR / ".yfinance"
PRICE_CACHE = CACHE_DIR / "country_etf_tactical_latest_prices.parquet"
RESULT_PREFIX = "country_etf_tactical_gold_dd_boost16"
STRATEGY = "Country ETF tactical + Gold DD boost 16"

START_DATE = "2003-01-01"
BASE_TICKERS = ("SPY", "GC=F", "BTC-USD", "BIL", "IEF")
TICKER_TO_ASSET = {"GC=F": "Gold", "BTC-USD": "BTC"}
CORE_MOMENTUM_ETFS = ("SPMO", "MTUM", "SCHG", "XLK", "EWY", "EWJ", "INDA")
COUNTRY_ETFS = (
    "EWC", "EWA", "EWW", "EWZ", "ARGT", "ECH", "EPU", "GXG",
    "EWG", "EWU", "EWQ", "EWL", "EWP", "EWI", "EWN", "EWD", "EDEN", "EIRL", "NORW", "EPOL", "GREK", "TUR",
    "EWJ", "DXJ", "EWT", "EWY", "INDA", "EPI", "MCHI", "KWEB", "EWS", "EWM", "EIDO", "VNM", "THD", "ENZL",
    "PAK", "KSA", "QAT", "UAE", "EIS", "EZA", "EGPT",
)
PRICE_TICKERS = tuple(dict.fromkeys([*BASE_TICKERS, *CORE_MOMENTUM_ETFS, *COUNTRY_ETFS]))

BASE_CORE = {"SPY": 0.45, "Gold": 0.30, "BTC": 0.10, "BIL": 0.15}
BASE_BUCKET = 0.15
BASE_TOP_N = 1
COUNTRY_BUCKET = 0.08
COUNTRY_TOP_N = 2
GOLD_BOOST = 0.16
MIN_HISTORY_DAYS = 2520


def _close_series(raw: pd.DataFrame, ticker: str) -> pd.Series:
    if raw.empty:
        return pd.Series(dtype=float, name=ticker)
    close = raw["Close"] if "Close" in raw.columns else raw.get("Adj Close", pd.Series(dtype=float))
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.dropna().astype(float).rename(ticker)
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close


def _download_close_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    frames: list[pd.Series] = []
    for ticker in tickers:
        raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False, threads=False)
        close = _close_series(raw, ticker)
        if close.empty:
            print(f"Warning: no fresh rows for {ticker}")
            continue
        frames.append(close)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1).sort_index()


def load_prices() -> pd.DataFrame:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    YF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(YF_CACHE_DIR))

    existing = pd.read_parquet(PRICE_CACHE).sort_index() if PRICE_CACHE.exists() else pd.DataFrame()
    if not existing.empty:
        existing.index = pd.to_datetime(existing.index)
        existing = existing.loc[:, ~existing.columns.duplicated(keep="last")]
        start = (pd.Timestamp(existing.index.max()).date() - timedelta(days=10)).isoformat()
    else:
        start = START_DATE
    end = (pd.Timestamp.today().date() + timedelta(days=1)).isoformat()

    missing = [ticker for ticker in PRICE_TICKERS if ticker not in existing.columns or existing[ticker].dropna().empty]
    update_tickers = list(dict.fromkeys([*PRICE_TICKERS])) if missing else list(PRICE_TICKERS)
    update = _download_close_prices(update_tickers, start, end)
    if update.empty and existing.empty:
        raise RuntimeError("No rows returned from yfinance for country ETF tactical latest refresh.")

    if not update.empty:
        cutoff = pd.Timestamp(update.index.min())
        historical = existing.loc[existing.index < cutoff] if not existing.empty else pd.DataFrame()
        prices = pd.concat([historical, update]).sort_index()
    else:
        prices = existing.sort_index()
    prices = prices[~prices.index.duplicated(keep="last")]
    prices = prices.reindex(columns=PRICE_TICKERS)
    prices.to_parquet(PRICE_CACHE)
    return prices


def asset_prices(raw: pd.DataFrame) -> pd.DataFrame:
    renamed = raw.rename(columns=TICKER_TO_ASSET)
    cols = list(dict.fromkeys(["SPY", "Gold", "BTC", "BIL", "IEF", *CORE_MOMENTUM_ETFS, *COUNTRY_ETFS]))
    return renamed.loc[:, ~renamed.columns.duplicated(keep="last")].reindex(columns=cols).dropna(how="all")


def _lag_close_signal_to_next_session(signal: pd.Series, initial: float = 1.0) -> pd.Series:
    return signal.astype(float).sort_index().shift(1).ffill().fillna(initial)


def trend_exposure(price: pd.Series, ma_period: int, below_exposure: float) -> pd.Series:
    price = price.astype(float).sort_index().ffill()
    ma = price.rolling(ma_period, min_periods=max(20, int(ma_period * 0.20))).mean()
    signal = pd.Series(1.0, index=price.index, dtype=float)
    signal.loc[price < ma] = below_exposure
    signal.loc[ma.isna()] = 1.0
    return _lag_close_signal_to_next_session(signal, initial=1.0)


def gold_drawdown_exposure(price: pd.Series) -> pd.Series:
    price = price.astype(float).sort_index().ffill()
    dd = price / price.rolling(252, min_periods=63).max() - 1.0
    signal = pd.Series(1.0, index=price.index, dtype=float)
    signal.loc[dd <= -0.08] = 0.50
    signal.loc[dd <= -0.20] = 0.50
    return _lag_close_signal_to_next_session(signal, initial=1.0)


def spy_risk_off(prices: pd.DataFrame, date: pd.Timestamp) -> bool:
    spy = prices["SPY"].loc[:date].dropna()
    if len(spy) < 252:
        return False
    latest = float(spy.iloc[-1])
    ma200 = float(spy.iloc[-200:].mean())
    dd252 = latest / float(spy.iloc[-252:].max()) - 1.0
    return latest < ma200 or dd252 <= -0.08


def momentum_rank(prices: pd.DataFrame, date: pd.Timestamp, candidates: tuple[str, ...]) -> pd.DataFrame:
    hist = prices.loc[:date, list(candidates)].ffill()
    rows: list[dict[str, object]] = []
    for asset in candidates:
        series = hist[asset].dropna() if asset in hist else pd.Series(dtype=float)
        if len(series) < 253:
            rows.append({"ETF": asset, "pass": False, "score": np.nan})
            continue
        latest = float(series.iloc[-1])
        ret_1m = latest / float(series.iloc[-22]) - 1.0
        ret_3m = latest / float(series.iloc[-64]) - 1.0
        ret_6m = latest / float(series.iloc[-127]) - 1.0
        ret_12m = latest / float(series.iloc[-253]) - 1.0
        sma200 = float(series.iloc[-200:].mean())
        rows.append(
            {
                "ETF": asset,
                "ret_1m": ret_1m,
                "ret_3m": ret_3m,
                "ret_6m": ret_6m,
                "ret_12m": ret_12m,
                "pass": ret_3m > 0.0 and ret_6m > 0.0 and latest > sma200,
            }
        )
    ranks = pd.DataFrame(rows)
    score = pd.Series(0.0, index=ranks.index, dtype=float)
    for col, weight in {"ret_1m": 0.20, "ret_3m": 0.30, "ret_6m": 0.40, "ret_12m": 0.10}.items():
        score += weight * ranks[col].rank(pct=True).fillna(0.0)
    ranks["score"] = score
    return ranks.sort_values(["pass", "score"], ascending=[False, False]).reset_index(drop=True)


def funded_weights(weights: pd.Series, selected: list[str], bucket: float) -> pd.Series:
    out = weights.copy()
    if not selected:
        return out
    out["SPY"] = out.get("SPY", 0.0) - bucket
    if out["SPY"] < 0.20 - 1e-12:
        return pd.Series(dtype=float)
    for asset in selected:
        out[asset] = out.get(asset, 0.0) + bucket / len(selected)
    return out / out.sum()


def baseline_weights(prices: pd.DataFrame, date: pd.Timestamp) -> tuple[pd.Series, list[str]]:
    ranks = momentum_rank(prices, date, CORE_MOMENTUM_ETFS)
    selected = ranks.loc[ranks["pass"].fillna(False), "ETF"].head(BASE_TOP_N).astype(str).tolist()
    weights = funded_weights(pd.Series(BASE_CORE, dtype=float), selected, BASE_BUCKET)
    if weights.empty:
        return pd.Series(BASE_CORE, dtype=float), []
    return weights, selected


def apply_gold_boost(weights: pd.Series) -> pd.Series:
    out = weights.copy()
    actual = min(GOLD_BOOST, max(0.0, float(out.get("SPY", 0.0)) - 0.20))
    out["SPY"] = out.get("SPY", 0.0) - actual
    out["Gold"] = out.get("Gold", 0.0) + actual
    return out / out.sum()


def latest_rebalance_date(prices: pd.DataFrame, as_of: pd.Timestamp) -> pd.Timestamp:
    eligible = prices.loc[:as_of].dropna(how="all")
    month_ends = eligible.groupby(eligible.index.to_period("M")).tail(1).index
    month_ends = month_ends[month_ends >= eligible.index.min() + pd.DateOffset(days=252)]
    if month_ends.empty:
        raise RuntimeError("Not enough history to find latest monthly rebalance date.")
    return pd.Timestamp(month_ends.max())


def sleeve_for_asset(asset: str) -> str:
    if asset == "SPY":
        return "US Equity"
    if asset in COUNTRY_ETFS:
        return "Country ETF"
    if asset in CORE_MOMENTUM_ETFS:
        return "Momentum ETF"
    if asset == "Gold":
        return "Gold"
    if asset == "BTC":
        return "BTC"
    if asset == "BIL":
        return "BIL"
    return "Other"


def build_latest_outputs(prices: pd.DataFrame, raw_prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    required = ["SPY", "Gold", "BTC", "BIL"]
    common = raw_prices.dropna(subset=required)
    if common.empty:
        raise RuntimeError(f"No common latest close for {required}.")
    as_of = pd.Timestamp(common.index.max())
    prices = prices.loc[:as_of].ffill()
    rebalance_date = latest_rebalance_date(prices, as_of)
    usable_country = tuple(asset for asset in COUNTRY_ETFS if asset in prices and prices[asset].dropna().shape[0] >= MIN_HISTORY_DAYS)
    if not usable_country:
        raise RuntimeError("No country ETF has enough history for the latest refresh.")

    base, base_selected = baseline_weights(prices, rebalance_date)
    ranks = momentum_rank(prices, rebalance_date, usable_country)
    country_selected = ranks.loc[ranks["pass"].fillna(False), "ETF"].head(COUNTRY_TOP_N).astype(str).tolist()
    target = funded_weights(base, country_selected, COUNTRY_BUCKET)
    if target.empty:
        country_selected = []
        target = base
    boost_on = spy_risk_off(prices, rebalance_date)
    if boost_on:
        target = apply_gold_boost(target)

    exposure = pd.DataFrame(index=prices.index)
    exposure["SPY"] = trend_exposure(prices["SPY"], 300, 0.50)
    exposure["Gold"] = gold_drawdown_exposure(prices["Gold"])
    exposure["BTC"] = trend_exposure(prices["BTC"], 50, 0.00)
    latest_exposure = {
        "SPY": float(exposure["SPY"].reindex([as_of]).ffill().iloc[0]),
        "Gold": float(exposure["Gold"].reindex([as_of]).ffill().iloc[0]),
        "BTC": float(exposure["BTC"].reindex([as_of]).ffill().iloc[0]),
    }

    rows = []
    reduced = 0.0
    for asset, raw_weight in target.sort_values(ascending=False).items():
        factor = latest_exposure.get(asset, 1.0)
        effective = float(raw_weight) * factor
        reduced += float(raw_weight) - effective
        rows.append(
            {
                "Asset": asset,
                "Sleeve": sleeve_for_asset(asset),
                "Target Weight": float(raw_weight),
                "Target Weight %": float(raw_weight) * 100.0,
                "Daily Exposure": factor,
                "Effective Weight": effective,
                "Effective Weight %": effective * 100.0,
                "Date": as_of.date().isoformat(),
                "Last Exposure Date": as_of.date().isoformat(),
                "Last Rebalance Date": rebalance_date.date().isoformat(),
                "Signal Source Close Date": prices.index[prices.index < as_of].max().date().isoformat(),
                "Latest Cache Trading Date": as_of.date().isoformat(),
                "Strategy": STRATEGY,
                "Daily Exposure Variant": "SPY MA300, Gold DD252, BTC MA50, country ETFs full exposure",
            }
        )
    if reduced > 1e-12:
        rows.append(
            {
                "Asset": "Cash / Reduced Exposure",
                "Sleeve": "Cash / Reduced Exposure",
                "Target Weight": 0.0,
                "Target Weight %": 0.0,
                "Daily Exposure": 1.0,
                "Effective Weight": reduced,
                "Effective Weight %": reduced * 100.0,
                "Date": as_of.date().isoformat(),
                "Last Exposure Date": as_of.date().isoformat(),
                "Last Rebalance Date": rebalance_date.date().isoformat(),
                "Signal Source Close Date": prices.index[prices.index < as_of].max().date().isoformat(),
                "Latest Cache Trading Date": as_of.date().isoformat(),
                "Strategy": STRATEGY,
                "Daily Exposure Variant": "Reduced exposure cash residual",
            }
        )

    latest = pd.DataFrame(rows).sort_values("Effective Weight", ascending=False)
    sleeve = (
        latest.groupby("Sleeve", as_index=False)
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
        "Latest Cache Trading Date": as_of.date().isoformat(),
        "Last Rebalance Date": rebalance_date.date().isoformat(),
        "Base Mix": "SPY 45%, Gold 30%, BTC 10%, BIL 15%",
        "Base Momentum Selected": ",".join(base_selected),
        "Country ETF Universe": "country_only",
        "Country Bucket": COUNTRY_BUCKET,
        "Country Top N": COUNTRY_TOP_N,
        "Country ETF Selected": ",".join(country_selected),
        "Gold Boost": GOLD_BOOST,
        "Gold Boost Active": bool(boost_on),
        "SPY Exposure": latest_exposure["SPY"],
        "Gold Exposure": latest_exposure["Gold"],
        "BTC Exposure": latest_exposure["BTC"],
        "Latest Weight Source": "Standalone refresh from this repo's current yfinance-backed country ETF cache and overlay prices; no static latest-weight file is read from dynamic_port_opt.",
    }
    return latest, sleeve, meta


def write_outputs(latest: pd.DataFrame, sleeve: pd.DataFrame, meta: dict[str, object]) -> None:
    for output_dir in [ROOT / "result", ROOT / "data" / "precomputed"]:
        output_dir.mkdir(parents=True, exist_ok=True)
        latest.to_csv(output_dir / f"{RESULT_PREFIX}_latest_effective_weights.csv", index=False)
        sleeve.to_csv(output_dir / f"{RESULT_PREFIX}_latest_sleeve_weights.csv", index=False)
        (output_dir / f"{RESULT_PREFIX}_latest_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        pd.DataFrame([meta]).to_csv(output_dir / f"{RESULT_PREFIX}_latest_meta.csv", index=False)


def main() -> None:
    raw = load_prices()
    raw_prices = asset_prices(raw)
    prices = raw_prices.ffill()
    latest, sleeve, meta = build_latest_outputs(prices, raw_prices)
    write_outputs(latest, sleeve, meta)
    print(f"Updated {STRATEGY} latest weights through {meta['Latest Cache Trading Date']}")
    print(latest[["Asset", "Target Weight %", "Daily Exposure", "Effective Weight %"]].to_string(index=False))


if __name__ == "__main__":
    main()
