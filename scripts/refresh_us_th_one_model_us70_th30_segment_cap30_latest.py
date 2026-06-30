from __future__ import annotations

from html.parser import HTMLParser

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import refresh_us_th_tactical_one_model_us70_th30_latest as refresh
from dynamic_factor_copula import (  # noqa: E402
    _parquet_column_names,
    load_cached_market_data,
    load_set100_membership_intervals,
    load_sp500_membership_intervals,
)
from share_class_utils import drop_duplicate_share_classes_available


SEGMENT_CAP = 0.30
US_SEGMENT_FILE = refresh.ROOT / "data" / "us_segment.csv"
TH_SEGMENT_FILE = refresh.ROOT / "data" / "set100_segment.xls"
TH_TO_US_SEGMENT = {
    "Agro & Food Industry": "Consumer Staples",
    "Consumer Products": "Consumer Discretionary",
    "Financials": "Financials",
    "Industrial": "Industrials",
    "Industrials": "Industrials",
    "Property & Construction": "Real Estate",
    "Resources": "Energy",
    "Services": "Consumer Discretionary",
    "Technology": "Information Technology",
}

refresh.STRATEGY = (
    "One-model US cap 70% / TH cap 30% stockcap5 penalty0.02 assets50 "
    "US segment cap 30% + daily exposure"
)
refresh.CASE = "US cap 70% / TH cap 30% stockcap5 penalty0.02 assets50 US segment cap 30%"
refresh.RESULT_PREFIX = "us_th_one_model_us70_th30_stockcap5_penalty002_assets50_segment_cap30"
refresh.STOCK_CAP = 0.05
refresh.US_ASSETS = 50
refresh.TH_ASSETS = 50
refresh.CONCENTRATION_PENALTY = 0.02

def _available_cached_columns(path) -> set[str]:
    return set(_parquet_column_names(str(path))) if path.exists() else set()


def _cache_us_th_panel() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str], list[str], pd.DataFrame, pd.Timestamp, str]:
    paths = refresh.default_paths(refresh.ROOT)
    source_cols = _available_cached_columns(paths.source_cache_root / "prices.parquet")
    extra_cols = _available_cached_columns(paths.local_cache_root / "extra_prices.parquet")
    available = source_cols | extra_cols
    sp500_intervals = load_sp500_membership_intervals(paths)
    set100_intervals = load_set100_membership_intervals(paths)
    all_us = [
        ticker
        for ticker in sp500_intervals["ticker"].dropna().astype(str).drop_duplicates()
        if ticker in available
    ]
    all_us = drop_duplicate_share_classes_available(all_us, all_us)
    all_th = [
        ticker
        for ticker in set100_intervals["ticker"].dropna().astype(str).drop_duplicates()
        if ticker in available
    ]
    cached = load_cached_market_data(paths, tickers=list(dict.fromkeys(all_us + all_th + ["SPY", "^VIX", "^SET.BK"])))
    stock_prices = cached["prices"].sort_index().ffill()
    stock_volumes = cached["volumes"].sort_index().fillna(0.0)
    overlay_path = paths.local_cache_root / "overlay_compare_prices.parquet"
    if not overlay_path.exists():
        raise RuntimeError(f"Missing overlay cache: {overlay_path}")
    overlay = pd.read_parquet(overlay_path).sort_index().ffill()
    extra_prices = paths.local_cache_root / "extra_prices.parquet"
    if extra_prices.exists():
        extra = pd.read_parquet(extra_prices, columns=["^SET.BK"]).sort_index().ffill()
        overlay["^SET.BK"] = extra["^SET.BK"]
    required = ["SPY", "^VIX", "GC=F", "BTC-USD", "USDTHB=X", "^SET.BK"]
    missing = [column for column in required if column not in overlay.columns]
    if missing:
        raise RuntimeError(f"Cache-backed latest refresh missing overlay columns: {missing}")
    full_index = stock_prices.index.union(overlay.index).sort_values()
    stock_prices = stock_prices.reindex(full_index).ffill()
    stock_volumes = stock_volumes.reindex(full_index).fillna(0.0)
    overlay = overlay.reindex(full_index).ffill()
    common = overlay[required].dropna().index.intersection(stock_prices.dropna(how="all").index)
    if common.empty:
        raise RuntimeError("Cache-backed latest refresh found no common stock/overlay date.")
    as_of = pd.Timestamp(common.max())
    fx = overlay["USDTHB=X"].ffill()
    us_cols = [ticker for ticker in all_us if ticker in stock_prices.columns]
    th_cols = [ticker for ticker in all_th if ticker in stock_prices.columns]
    prices = pd.concat([stock_prices[us_cols].mul(fx, axis=0), stock_prices[th_cols]], axis=1).sort_index().ffill()
    prices = prices.loc[refresh.START_DATE:as_of]
    volumes = stock_volumes.reindex(prices.index).reindex(columns=prices.columns).fillna(0.0)
    benchmark = (overlay["SPY"] * fx).reindex(prices.index).ffill().rename("benchmark")
    vol_proxy = overlay["^VIX"].reindex(prices.index).ffill().rename("vol_proxy")
    signal_overlay = overlay.reindex(prices.index).ffill()
    return prices, volumes, benchmark, vol_proxy, us_cols, th_cols, signal_overlay, as_of, refresh.START_DATE

_original_write_outputs = refresh._write_outputs
_latest_raw_segment_weights: dict[str, float] = {}


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "tr":
            self._row = []
        elif tag.lower() == "td" and self._row is not None:
            self._cell = []

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == "td" and self._row is not None and self._cell is not None:
            self._row.append(" ".join(part.strip() for part in self._cell if part.strip()))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if self._row:
                self.rows.append(self._row)
            self._row = None


def _us_segment_table() -> pd.DataFrame:
    segment = pd.read_csv(US_SEGMENT_FILE)
    required = {"ticker", "segment"}
    if not required.issubset(segment.columns):
        raise RuntimeError(f"{US_SEGMENT_FILE} must contain columns: {sorted(required)}")
    segment = segment.copy()
    segment["ticker"] = segment["ticker"].astype(str).str.upper()
    segment["segment"] = segment["segment"].astype(str).str.strip()
    return segment.loc[segment["ticker"].ne("") & segment["segment"].ne("")]


def _th_segment_table() -> pd.DataFrame:
    parser = _TableParser()
    parser.feed(TH_SEGMENT_FILE.read_text(encoding="utf-8", errors="ignore"))
    header_idx = next(
        (idx for idx, row in enumerate(parser.rows) if row and row[0].strip().casefold() == "symbol"),
        None,
    )
    if header_idx is None:
        raise RuntimeError(f"{TH_SEGMENT_FILE} must contain a Symbol header row")
    header = [cell.strip() for cell in parser.rows[header_idx]]
    rows = parser.rows[header_idx + 1 :]
    frame = pd.DataFrame([row[: len(header)] for row in rows if len(row) >= len(header)], columns=header)
    required = {"Symbol", "Industry"}
    if not required.issubset(frame.columns):
        raise RuntimeError(f"{TH_SEGMENT_FILE} must contain columns: {sorted(required)}")
    frame = frame.rename(columns={"Symbol": "ticker", "Industry": "segment"})
    frame["ticker"] = frame["ticker"].astype(str).str.strip().str.upper() + ".BK"
    frame["segment"] = frame["segment"].astype(str).str.strip()
    return frame.loc[
        frame["ticker"].ne(".BK") & frame["segment"].ne("") & frame["segment"].ne("-"),
        ["ticker", "segment"],
    ]


def _normalized_segment(segment: str | None, market: str) -> str | None:
    if segment is None or pd.isna(segment):
        return None
    value = str(segment).strip()
    if not value or value.lower() == "nan" or value == "-":
        return None
    return TH_TO_US_SEGMENT.get(value, value) if market == "th" else value


def _segment_maps() -> tuple[dict[str, str], dict[str, str]]:
    us_table = _us_segment_table()
    th_table = _th_segment_table()
    us = dict(zip(us_table["ticker"], us_table["segment"]))
    th = dict(zip(th_table["ticker"], th_table["segment"]))
    return us, th


def _segment_groups(selected: list[str], us_assets: set[str], th_assets: set[str]) -> dict[str, list[int]]:
    us_segments, _th_segments = _segment_maps()
    groups: dict[str, list[int]] = {}
    for idx, asset in enumerate(selected):
        ticker = str(asset).upper()
        if ticker in us_assets:
            segment = _normalized_segment(us_segments.get(ticker), "us")
        else:
            segment = None
        if segment:
            groups.setdefault(segment, []).append(idx)
    return groups


def _segment_weights(weights: pd.Series, us_assets: set[str], th_assets: set[str]) -> dict[str, float]:
    us_segments, _th_segments = _segment_maps()
    out: dict[str, float] = {}
    for asset, weight in weights.items():
        ticker = str(asset).upper()
        if ticker in us_assets:
            segment = _normalized_segment(us_segments.get(ticker), "us")
        else:
            segment = None
        if segment:
            out[segment] = out.get(segment, 0.0) + float(weight)
    return out


def _optimize_one_model_with_segment_cap(
    train_returns: pd.DataFrame,
    benchmark: pd.Series,
    vol_proxy: pd.Series,
    prices: pd.DataFrame,
    us_assets: set[str],
    th_assets: set[str],
) -> pd.Series:
    global _latest_raw_segment_weights

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
    if us_idx:
        constraints.append({"type": "ineq", "fun": lambda x, idx=us_idx: refresh.US_GROUP_CAP - float(np.sum(x[idx]))})
    if th_idx:
        constraints.append({"type": "ineq", "fun": lambda x, idx=th_idx: refresh.TH_GROUP_CAP - float(np.sum(x[idx]))})
    for segment, idx in _segment_groups(selected, us_assets, th_assets).items():
        constraints.append({"type": "ineq", "fun": lambda x, idx=idx: SEGMENT_CAP - float(np.sum(x[idx]))})

    def objective(x: np.ndarray) -> float:
        variance = float(x @ cov_matrix @ x)
        expected = float(mu @ x)
        concentration = float(np.sum(np.square(x)))
        return 0.5 * refresh.RISK_AVERSION * variance - expected + refresh.CONCENTRATION_PENALTY * concentration

    result = minimize(objective, x0=x0.to_numpy(dtype=float), bounds=bounds, constraints=constraints, method="SLSQP")
    weights = pd.Series(result.x, index=selected).clip(lower=0.0) if result.success else x0.copy()
    weights = (weights / weights.sum()).sort_values(ascending=False)
    _latest_raw_segment_weights = _segment_weights(weights, us_assets, th_assets)
    return weights


def _write_outputs_with_segment_meta(security: pd.DataFrame, sleeve: pd.DataFrame, meta: pd.DataFrame) -> None:
    if not meta.empty:
        meta = meta.copy()
        meta["Segment Cap"] = SEGMENT_CAP
        meta["Segment Cap Mode"] = "US-only segment cap"
        meta["Latest Max US Segment Raw Weight"] = (
            max(_latest_raw_segment_weights.values()) if _latest_raw_segment_weights else 0.0
        )
        meta["Latest US Segment Weights"] = "; ".join(
            f"{segment}: {weight:.2%}" for segment, weight in sorted(_latest_raw_segment_weights.items())
        )
        meta["Segment Source"] = "data/us_segment.csv; TH segments are not capped by this variant"
        meta["Latest Weight Source"] = "Standalone cache-backed refresh from this repo data/cache and segment files; no static latest-weight file is read from dynamic_port_opt."
    _original_write_outputs(security, sleeve, meta)


refresh._fresh_us_th_panel = _cache_us_th_panel
refresh._optimize_one_model = _optimize_one_model_with_segment_cap
refresh._write_outputs = _write_outputs_with_segment_meta


if __name__ == "__main__":
    refresh.main()
