from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "dynamic_factor_copula"
CACHE_FILE = CACHE_DIR / "overlay_compare_prices.parquet"
YF_CACHE_DIR = CACHE_DIR / ".yfinance"
TICKERS = ["SPY", "GC=F", "BTC-USD", "^VIX", "USDTHB=X"]


def _close_frame(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            close = raw["Close"]
        elif "Adj Close" in raw.columns.get_level_values(0):
            close = raw["Adj Close"]
        else:
            raise RuntimeError(f"Unexpected yfinance columns: {raw.columns}")
    else:
        close = raw[["Close"]].copy()
        close.columns = [TICKERS[0]]
    return close.reindex(columns=TICKERS).sort_index()


def _download_close_prices(start: str, end: str) -> pd.DataFrame:
    frames = []
    for ticker in TICKERS:
        raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False, threads=False)
        if raw.empty:
            print(f"Warning: no fresh rows for {ticker}")
            continue
        if isinstance(raw.columns, pd.MultiIndex):
            close_data = raw["Close"].iloc[:, 0] if "Close" in raw.columns.get_level_values(0) else pd.Series(dtype=float)
        else:
            close_data = raw["Close"] if "Close" in raw.columns else pd.Series(dtype=float)
        close = close_data.dropna().rename(ticker).to_frame()
        if close.empty:
            print(f"Warning: no fresh rows for {ticker}")
            continue
        frames.append(close[[ticker]])
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1).reindex(columns=TICKERS).sort_index()


def _trend_exposure(price: pd.Series, below: float, ma_days: int = 200) -> tuple[float, pd.Timestamp, float, float]:
    clean = price.dropna().astype(float)
    ma = clean.rolling(ma_days, min_periods=40).mean()
    latest_date = clean.index.max()
    latest_price = float(clean.loc[latest_date])
    latest_ma = float(ma.loc[latest_date])
    exposure = below if latest_price < latest_ma else 1.0
    return exposure, latest_date, latest_price, latest_ma


def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    YF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(YF_CACHE_DIR))

    existing = pd.read_parquet(CACHE_FILE).sort_index() if CACHE_FILE.exists() else pd.DataFrame()
    if not existing.empty:
        start = pd.Timestamp(existing.index.max()).date() - timedelta(days=10)
    else:
        start = pd.Timestamp.today().date() - timedelta(days=420)
    end = pd.Timestamp.today().date() + timedelta(days=1)

    update = _download_close_prices(start.isoformat(), end.isoformat())
    if update.empty:
        raise RuntimeError("No rows returned from yfinance.")

    combined = pd.concat([existing, update]).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    combined = combined.reindex(columns=TICKERS).ffill()
    combined.to_parquet(CACHE_FILE)

    fx = combined["USDTHB=X"].ffill()
    spy_thb = combined["SPY"] * fx
    gold_thb = combined["GC=F"] * fx
    btc_thb = combined["BTC-USD"] * fx

    print("Updated data/cache/dynamic_factor_copula/overlay_compare_prices.parquet")
    print(f"Rows: {len(combined):,}")
    print(f"Date range: {combined.index.min().date()} -> {combined.index.max().date()}")
    for label, series, below in [
        ("Gold", gold_thb, 0.50),
        ("BTC", btc_thb, 0.00),
        ("SPY", spy_thb, 1.00),
    ]:
        exposure, dt, price, ma = _trend_exposure(series, below=below)
        print(f"{label}: {dt.date()} price_thb={price:,.2f} ma200={ma:,.2f} exposure={exposure:.0%}")


if __name__ == "__main__":
    main()
