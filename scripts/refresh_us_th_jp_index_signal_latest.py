from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import yfinance as yf


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dynamic_factor_copula import default_paths  # noqa: E402
from dynamic_factor_copula import load_cached_market_data  # noqa: E402
from refresh_us_th_tactical_final_best_latest import (  # noqa: E402
    START_DATE,
    TH_ASSETS,
    US_ASSETS,
    _active_members,
    _all_available_members,
    _optimize_latest_sleeve,
)


RESULT_PREFIX = "us_th_jp_optimized_minvol_top10_cap15"
JP_SLEEVE_MODE = "JP optimized min_vol_mom_tilt top10 cap15%"
WEEKLY_STRATEGY = (
    "JP optimized min_vol_mom_tilt top10 cap15% Stock60/Gold30/BTC10 Index signal leaves inactive equity in cash "
    "+ weekly exposure all assets + gold drawdown 252d warn10 crash20"
)
EQUITY_BUDGET = 0.60
GOLD_WEIGHT = 0.30
BTC_WEIGHT = 0.10
JP_INDEX_PROXY = "13060"
JP_ASSETS = 10
JP_CAP = 0.15
JP_OBJECTIVE = "min_vol_mom_tilt"
JP_LOOKBACK_DAYS = 120
JP_INITIAL_TRAIN_DAYS = 40
JP_CONCENTRATION_PENALTY = 0.01
EPSILON = 1e-12
JP_YAHOO_LOOKBACK_DAYS = 550
YF_CACHE_DIR = ROOT / "data" / "cache" / "dynamic_factor_copula" / ".yfinance"


def _load_overlay() -> pd.DataFrame:
    paths = default_paths(ROOT)
    overlay = pd.read_parquet(paths.local_cache_root / "overlay_compare_prices.parquet").sort_index().ffill()
    extra_prices = paths.local_cache_root / "extra_prices.parquet"
    if extra_prices.exists():
        extra = pd.read_parquet(extra_prices).sort_index()
        if "^SET.BK" in extra.columns:
            overlay["^SET.BK"] = extra["^SET.BK"]
    required = ["SPY", "^SET.BK", "GC=F", "BTC-USD", "USDTHB=X"]
    missing = [column for column in required if column not in overlay.columns]
    if missing:
        raise RuntimeError(f"Missing required overlay columns: {missing}")
    return overlay[required].ffill()


def _jquants_code_to_yahoo(code: str) -> str:
    clean = str(code).strip()
    if clean.endswith("0") and len(clean) >= 5:
        clean = clean[:-1]
    return f"{clean}.T"


def _load_japan_universe() -> pd.DataFrame:
    paths = default_paths(ROOT)
    universe_file = paths.local_cache_root / "japan_pit_universe_history.parquet"
    if not universe_file.exists():
        raise FileNotFoundError(f"Missing Japan PIT universe file: {universe_file}")

    universe = pd.read_parquet(universe_file)
    universe["entry_date"] = pd.to_datetime(universe["entry_date"], errors="coerce")
    universe["Code"] = universe["Code"].astype(str).str.strip()
    return universe


def _latest_japan_universe_codes(universe: pd.DataFrame, as_of: pd.Timestamp | None = None) -> tuple[list[str], str]:
    rows = universe.dropna(subset=["entry_date"]).copy()
    if as_of is not None:
        rows = rows.loc[rows["entry_date"] <= as_of]
    rows = rows.sort_values(["entry_date", "rank"])
    if rows.empty:
        return [], ""
    latest_entry = rows["entry_date"].max()
    codes = rows.loc[rows["entry_date"].eq(latest_entry), "Code"].dropna().astype(str).str.strip().tolist()
    return codes[:JP_ASSETS], pd.Timestamp(latest_entry).date().isoformat()


def _download_yahoo_close(ticker_map: dict[str, str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if not ticker_map:
        return pd.DataFrame()
    YF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        yf.set_tz_cache_location(str(YF_CACHE_DIR))
    except Exception:
        pass

    frames: list[pd.Series] = []
    for code, ticker in ticker_map.items():
        raw = yf.download(
            ticker,
            start=start.date().isoformat(),
            end=(end + pd.Timedelta(days=1)).date().isoformat(),
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        if raw.empty:
            continue
        close = raw["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close = close.dropna().astype(float).rename(code)
        frames.append(close)
    if not frames:
        return pd.DataFrame()
    prices = pd.concat(frames, axis=1).sort_index().ffill()
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    return prices


def _load_japan_prices_from_yahoo(universe: pd.DataFrame, overlay_end: pd.Timestamp) -> tuple[pd.DataFrame, str, list[str]]:
    selected_codes, universe_entry_date = _latest_japan_universe_codes(universe, overlay_end)
    wanted = list(dict.fromkeys(selected_codes + [JP_INDEX_PROXY]))
    ticker_map = {code: _jquants_code_to_yahoo(code) for code in wanted}
    start = overlay_end - pd.Timedelta(days=JP_YAHOO_LOOKBACK_DAYS)
    prices = _download_yahoo_close(ticker_map, start=start, end=overlay_end)
    missing = [code for code in wanted if code not in prices.columns or prices[code].dropna().empty]
    if selected_codes and not any(code in prices.columns and prices[code].dropna().shape[0] >= JP_INITIAL_TRAIN_DAYS for code in selected_codes):
        raise RuntimeError("Yahoo Finance returned no usable Japan stock prices for latest JP universe.")
    return prices, universe_entry_date, missing


def _load_japan_name_map(as_of: pd.Timestamp) -> dict[str, str]:
    paths = default_paths(ROOT)
    master_file = paths.local_cache_root / "japan_master_history.parquet"
    if not master_file.exists():
        return {}
    master = pd.read_parquet(master_file)
    if master.empty or "Code" not in master.columns:
        return {}
    master["Code"] = master["Code"].astype(str).str.strip()
    date_col = "signal_date" if "signal_date" in master.columns else "Date"
    if date_col in master.columns:
        master[date_col] = pd.to_datetime(master[date_col], errors="coerce")
        master = master.loc[master[date_col].le(as_of) | master[date_col].isna()].sort_values(date_col)
    name_col = "CoNameEn" if "CoNameEn" in master.columns else "CoName"
    if name_col not in master.columns:
        return {}
    latest = master.dropna(subset=["Code"]).drop_duplicates("Code", keep="last")
    names = latest.set_index("Code")[name_col].fillna("").astype(str).str.strip()
    fallback = latest.set_index("Code").get("CoName", pd.Series(dtype=str)).fillna("").astype(str).str.strip()
    output: dict[str, str] = {}
    for code, name in names.items():
        clean_name = name or str(fallback.get(code, ""))
        output[str(code)] = clean_name or str(code)
    return output


def _japan_signal_price(prices: pd.DataFrame, universe: pd.DataFrame) -> pd.Series:
    if JP_INDEX_PROXY in prices.columns and prices[JP_INDEX_PROXY].dropna().shape[0] >= 40:
        return prices[JP_INDEX_PROXY].rename("JP index proxy")
    selected = sorted(universe["Code"].dropna().astype(str).str.strip().unique().tolist())
    return prices.reindex(columns=[ticker for ticker in selected if ticker in prices.columns]).mean(axis=1).rename("JP PIT proxy signal")


def _momentum_signal(train_returns: pd.DataFrame) -> pd.Series:
    if train_returns.empty:
        return pd.Series(dtype=float)
    lookback = train_returns.tail(min(63, len(train_returns)))
    signal = (1.0 + lookback).prod() - 1.0
    return signal.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _fallback_equal(assets: list[str]) -> pd.Series:
    if not assets:
        return pd.Series(dtype=float)
    return pd.Series(1.0 / len(assets), index=assets, dtype=float)


def _optimized_japan_weights(train_returns: pd.DataFrame, assets: list[str]) -> pd.Series:
    usable = train_returns.reindex(columns=assets).dropna(
        axis=1,
        thresh=max(JP_INITIAL_TRAIN_DAYS, int(len(train_returns) * 0.75)),
    )
    usable = usable.dropna(how="all")
    if usable.shape[1] < 2 or usable.shape[0] < JP_INITIAL_TRAIN_DAYS or usable.shape[1] * JP_CAP < 1.0 - 1e-9:
        return _fallback_equal(assets)
    cov = usable.cov().fillna(0.0)
    cov = 0.80 * cov + 0.20 * pd.DataFrame(np.diag(np.diag(cov)), index=cov.index, columns=cov.columns)
    momentum = _momentum_signal(usable)
    assets_index = cov.index
    cov_matrix = cov.to_numpy(dtype=float)
    mu = momentum.reindex(assets_index).fillna(momentum.median() if not momentum.empty else 0.0).to_numpy()
    mu = np.clip(mu, np.nanpercentile(mu, 10), np.nanpercentile(mu, 90)) if len(mu) else mu
    caps = pd.Series(JP_CAP, index=assets_index, dtype=float)
    if float(caps.sum()) < 1.0 - EPSILON:
        return _fallback_equal(assets)
    x0 = caps / caps.sum()
    bounds = [(0.0, float(caps.loc[asset])) for asset in assets_index]
    constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1.0}]

    def objective(x: np.ndarray) -> float:
        variance = float(x @ cov_matrix @ x)
        expected = float(mu @ x)
        concentration = float(np.sum(np.square(x)))
        if JP_OBJECTIVE == "min_vol_mom_tilt":
            return variance - 0.20 * expected + JP_CONCENTRATION_PENALTY * concentration
        raise ValueError(f"Unsupported JP objective: {JP_OBJECTIVE}")

    result = minimize(objective, x0=x0.to_numpy(dtype=float), bounds=bounds, constraints=constraints, method="SLSQP")
    if not result.success:
        weights = x0
    else:
        weights = pd.Series(result.x, index=assets_index).clip(lower=0.0)
        weights = weights / weights.sum()
    return weights.reindex(assets).fillna(0.0)


def _latest_japan_internal_weights(
    prices: pd.DataFrame,
    universe: pd.DataFrame,
    as_of: pd.Timestamp,
) -> tuple[pd.Series, str]:
    eligible, universe_entry_date = _latest_japan_universe_codes(universe, as_of)
    eligible = [ticker for ticker in eligible if ticker in prices.columns]
    if not eligible:
        return pd.Series(dtype=float), ""
    returns = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    train_end_pos = returns.index.searchsorted(as_of, side="right")
    train_start_pos = max(0, train_end_pos - JP_LOOKBACK_DAYS)
    train_returns = returns.iloc[train_start_pos:train_end_pos].fillna(0.0).reindex(columns=eligible)
    weights = _optimized_japan_weights(train_returns, eligible)
    return weights.sort_index(), as_of.date().isoformat()


def _latest_us_th_internal_weights(overlay: pd.DataFrame, as_of: pd.Timestamp) -> tuple[pd.Series, pd.Series, str, str]:
    paths = default_paths(ROOT)
    us_all = _all_available_members("us")
    th_all = _all_available_members("th")
    us_active = _active_members("us", as_of, us_all)
    th_active = _active_members("th", as_of, th_all)
    cached = load_cached_market_data(paths, tickers=list(dict.fromkeys(us_active + th_active)))
    fx = overlay["USDTHB=X"].reindex(cached["prices"].index).ffill()

    us_prices = cached["prices"].reindex(columns=us_active).mul(fx, axis=0).loc[START_DATE:as_of].ffill()
    th_prices = cached["prices"].reindex(columns=th_active).loc[START_DATE:as_of].ffill()
    us_volumes = cached["volumes"].reindex(us_prices.index).reindex(columns=us_active).fillna(0.0)
    th_volumes = cached["volumes"].reindex(th_prices.index).reindex(columns=th_active).fillna(0.0)
    benchmark_us = (overlay["SPY"] * overlay["USDTHB=X"]).reindex(us_prices.index).ffill().rename("benchmark")
    benchmark_th = overlay["^SET.BK"].reindex(th_prices.index).ffill().rename("benchmark")
    vol_proxy_us = pd.read_parquet(paths.local_cache_root / "overlay_compare_prices.parquet").sort_index()["^VIX"].reindex(us_prices.index).ffill().rename("vol_proxy")
    vol_proxy_th = pd.Series(0.0, index=th_prices.index, name="vol_proxy")

    us_weights, us_internal_date, _, _ = _optimize_latest_sleeve(
        us_prices,
        us_volumes,
        benchmark_us,
        vol_proxy_us,
        us_active,
        US_ASSETS,
        as_of,
    )
    th_weights, th_internal_date, _, _ = _optimize_latest_sleeve(
        th_prices,
        th_volumes,
        benchmark_th,
        vol_proxy_th,
        th_active,
        TH_ASSETS,
        as_of,
    )
    return us_weights, th_weights, us_internal_date.date().isoformat(), th_internal_date.date().isoformat()


def _score(price: pd.Series, ma_period: int) -> pd.Series:
    price = price.astype(float).sort_index().ffill()
    trend = (price > price.rolling(ma_period, min_periods=max(40, ma_period // 3)).mean()).astype(float)
    momentum = (price.pct_change(63, fill_method=None) > 0.0).astype(float)
    return ((trend + momentum) / 2.0).shift(1).ffill().fillna(0.0).clip(0.0, 1.0)


def _close_trend_exposure(price: pd.Series, ma_period: int, below_exposure: float) -> pd.Series:
    price = price.astype(float).sort_index().ffill()
    ma = price.rolling(ma_period, min_periods=max(20, int(ma_period * 0.20))).mean()
    exposure = pd.Series(1.0, index=price.index, dtype=float)
    exposure.loc[price < ma] = below_exposure
    exposure.loc[ma.isna()] = 1.0
    return exposure.shift(1).ffill().fillna(1.0).clip(0.0, 1.0)


def _gold_drawdown_exposure(price: pd.Series) -> pd.Series:
    price = price.astype(float).sort_index().ffill()
    rolling_high = price.rolling(252, min_periods=63).max()
    drawdown = price / rolling_high - 1.0
    panic_ma = price.rolling(200, min_periods=50).mean()
    panic_mom = price.pct_change(63, fill_method=None)
    active = 1.0
    values: list[float] = []
    for date, dd in drawdown.items():
        panic = (
            pd.notna(dd)
            and dd <= -0.30
            and pd.notna(panic_ma.loc[date])
            and price.loc[date] < panic_ma.loc[date]
            and pd.notna(panic_mom.loc[date])
            and panic_mom.loc[date] < 0.0
        )
        if pd.isna(dd):
            active = 1.0
        elif panic:
            active = 0.0
        elif dd <= -0.20:
            active = 0.50
        elif dd <= -0.10:
            active = min(active, 0.75)
        elif dd >= -0.05:
            active = 1.0
        values.append(active)
    return pd.Series(values, index=drawdown.index, name="Gold Daily Exposure").shift(1).ffill().fillna(1.0).clip(0.0, 1.0)


def _weekly_exposure(exposure: pd.DataFrame) -> pd.DataFrame:
    weekly = exposure.resample("W-FRI").last()
    return weekly.reindex(exposure.index).ffill().fillna(1.0).clip(0.0, 1.0)


def _source_close_date(index: pd.Index, effective_date: pd.Timestamp) -> str:
    dates = pd.DatetimeIndex(index[index < effective_date])
    source = dates.max() if len(dates) else effective_date
    return pd.Timestamp(source).date().isoformat()


def _security_rows(
    strategy: str,
    raw_sleeve: pd.Series,
    effective_sleeve: pd.Series,
    exposure: pd.Series,
    us_internal: pd.Series,
    th_internal: pd.Series,
    jp_internal: pd.Series,
    jp_name_map: dict[str, str],
    as_of: pd.Timestamp,
    source_date: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for sleeve, weights in [("US Equity", us_internal), ("TH Equity", th_internal)]:
        for asset, internal_weight in weights.items():
            target_weight = float(internal_weight * raw_sleeve.get(sleeve, 0.0))
            rows.append(
                {
                    "Asset": asset,
                    "Ticker": asset,
                    "Sleeve": sleeve,
                    "Internal Weight": float(internal_weight),
                    "Raw Sleeve Weight": float(raw_sleeve.get(sleeve, 0.0)),
                    "Target Weight": target_weight,
                    "Effective Weight": float(internal_weight * effective_sleeve.get(sleeve, 0.0)),
                    "Daily Exposure": float(exposure.get(sleeve, 1.0)),
                }
            )
    for asset, internal_weight in jp_internal.items():
        asset_code = str(asset)
        asset_name = jp_name_map.get(asset_code, asset_code)
        target_weight = float(internal_weight * raw_sleeve.get("JP Equity", 0.0))
        rows.append(
            {
                "Asset": asset_name,
                "Ticker": asset_code,
                "Sleeve": "JP Equity",
                "Internal Weight": float(internal_weight),
                "Raw Sleeve Weight": float(raw_sleeve.get("JP Equity", 0.0)),
                "Target Weight": target_weight,
                "Effective Weight": float(internal_weight * effective_sleeve.get("JP Equity", 0.0)),
                "Daily Exposure": float(exposure.get("JP Equity", 1.0)),
            }
        )
    for asset, sleeve in [("GC=F", "Gold"), ("BTC-USD", "BTC"), ("Cash / Reduced Exposure", "Cash / Reduced Exposure")]:
        target_weight = float(raw_sleeve.get(sleeve, 0.0))
        rows.append(
            {
                "Asset": asset,
                "Ticker": asset,
                "Sleeve": sleeve,
                "Internal Weight": 1.0,
                "Raw Sleeve Weight": float(raw_sleeve.get(sleeve, 0.0)),
                "Target Weight": target_weight,
                "Effective Weight": float(effective_sleeve.get(sleeve, 0.0)),
                "Daily Exposure": float(exposure.get(sleeve, 1.0)),
            }
        )
    frame = pd.DataFrame(rows)
    frame["Target Weight %"] = frame["Target Weight"].mul(100.0)
    frame["Effective Weight %"] = frame["Effective Weight"].mul(100.0)
    frame["Date"] = as_of.date().isoformat()
    frame["Strategy"] = strategy
    frame["Last Exposure Date"] = as_of.date().isoformat()
    frame["Signal Source Close Date"] = source_date
    keep = frame["Effective Weight"].abs().gt(1e-12) | frame["Target Weight"].abs().gt(1e-12)
    return frame.loc[keep].sort_values(["Effective Weight", "Target Weight"], ascending=False)


def _build_outputs(
    strategy: str,
    raw_sleeve: pd.Series,
    exposure: pd.Series,
    us_internal: pd.Series,
    th_internal: pd.Series,
    jp_internal: pd.Series,
    jp_name_map: dict[str, str],
    as_of: pd.Timestamp,
    source_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    asset_cols = ["US Equity", "TH Equity", "JP Equity", "Gold", "BTC"]
    effective_sleeve = raw_sleeve.reindex(asset_cols).fillna(0.0).mul(exposure.reindex(asset_cols).fillna(1.0))
    effective_sleeve["Cash / Reduced Exposure"] = max(0.0, 1.0 - float(effective_sleeve.sum()))
    security = _security_rows(
        strategy,
        raw_sleeve,
        effective_sleeve,
        exposure,
        us_internal,
        th_internal,
        jp_internal,
        jp_name_map,
        as_of,
        source_date,
    )
    sleeve = effective_sleeve.rename("Effective Weight").reset_index().rename(columns={"index": "Sleeve"})
    sleeve["Raw Sleeve Weight"] = sleeve["Sleeve"].map(raw_sleeve).fillna(0.0)
    sleeve["Daily Exposure"] = sleeve["Sleeve"].map(exposure).fillna(1.0)
    sleeve["Effective Weight %"] = sleeve["Effective Weight"].mul(100.0)
    sleeve["Date"] = as_of.date().isoformat()
    sleeve["Strategy"] = strategy
    return security, sleeve.sort_values("Effective Weight", ascending=False)


def _write_outputs(weekly_security: pd.DataFrame, weekly_sleeve: pd.DataFrame, meta: pd.DataFrame) -> None:
    for output_dir in [ROOT / "result", ROOT / "data" / "precomputed"]:
        output_dir.mkdir(parents=True, exist_ok=True)
        weekly_security.to_csv(output_dir / f"{RESULT_PREFIX}_weekly_latest_effective_weights_thb.csv", index=False)
        weekly_sleeve.to_csv(output_dir / f"{RESULT_PREFIX}_weekly_latest_sleeve_weights_thb.csv", index=False)
        meta.to_csv(output_dir / f"{RESULT_PREFIX}_latest_meta.csv", index=False)
        payload = meta.iloc[0].to_dict() if not meta.empty else {}
        payload["calculated_at"] = pd.Timestamp.now(tz="Asia/Bangkok").isoformat()
        (output_dir / f"{RESULT_PREFIX}_latest_meta.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def main() -> None:
    overlay = _load_overlay()
    overlay_end = pd.Timestamp(overlay.dropna().index.max())
    jp_universe = _load_japan_universe()
    jp_prices, jp_universe_entry_date, jp_missing_yahoo = _load_japan_prices_from_yahoo(jp_universe, overlay_end)
    jp_signal = _japan_signal_price(jp_prices, jp_universe)
    common_end = min(overlay_end, pd.Timestamp(jp_prices.dropna(how="all").index.max()))
    as_of = pd.Timestamp(common_end)

    signal_prices = pd.DataFrame(
        {
            "US Equity": overlay["SPY"],
            "TH Equity": overlay["^SET.BK"],
            "JP Equity": jp_signal,
            "Gold": overlay["GC=F"],
            "BTC": overlay["BTC-USD"],
        }
    ).sort_index().ffill().loc[:as_of]

    scores = pd.Series(
        {
            "US Equity": float(_score(signal_prices["US Equity"], 300).loc[:as_of].iloc[-1]),
            "TH Equity": float(_score(signal_prices["TH Equity"], 200).loc[:as_of].iloc[-1]),
            "JP Equity": float(_score(signal_prices["JP Equity"], 120).loc[:as_of].iloc[-1]),
        },
        dtype=float,
    )
    raw_sleeve = pd.Series(
        {
            "US Equity": EQUITY_BUDGET / 3.0 * scores["US Equity"],
            "TH Equity": EQUITY_BUDGET / 3.0 * scores["TH Equity"],
            "JP Equity": EQUITY_BUDGET / 3.0 * scores["JP Equity"],
            "Gold": GOLD_WEIGHT,
            "BTC": BTC_WEIGHT,
        },
        dtype=float,
    )
    raw_sleeve["Cash / Reduced Exposure"] = max(0.0, 1.0 - float(raw_sleeve.sum()))

    daily_exposure = pd.DataFrame(
        {
            "US Equity": _close_trend_exposure(signal_prices["US Equity"], 300, 0.50),
            "TH Equity": _close_trend_exposure(signal_prices["TH Equity"], 200, 0.00),
            "JP Equity": _close_trend_exposure(signal_prices["JP Equity"], 120, 0.00),
            "Gold": _gold_drawdown_exposure(signal_prices["Gold"]),
            "BTC": _close_trend_exposure(signal_prices["BTC"], 50, 0.00),
        }
    ).ffill().fillna(1.0).clip(0.0, 1.0)
    weekly_exposure = _weekly_exposure(daily_exposure)
    weekly_latest = weekly_exposure.loc[:as_of].iloc[-1]
    jp_internal, jp_internal_date = _latest_japan_internal_weights(jp_prices, jp_universe, as_of)
    jp_name_map = _load_japan_name_map(as_of)
    us_internal, th_internal, us_internal_date, th_internal_date = _latest_us_th_internal_weights(overlay, as_of)
    source_date = _source_close_date(signal_prices.index, as_of)

    weekly_security, weekly_sleeve = _build_outputs(
        WEEKLY_STRATEGY,
        raw_sleeve,
        weekly_latest,
        us_internal,
        th_internal,
        jp_internal,
        jp_name_map,
        as_of,
        source_date,
    )

    gold_high = signal_prices["Gold"].rolling(252, min_periods=63).max()
    gold_dd = float(signal_prices["Gold"].loc[as_of] / gold_high.loc[as_of] - 1.0)
    meta = pd.DataFrame(
        [
            {
                "Date": as_of.date().isoformat(),
                "Strategy": WEEKLY_STRATEGY,
                "JP Sleeve Mode": JP_SLEEVE_MODE,
                "Model": "Index-signal sleeve allocation; inactive equity remains cash",
                "Base Allocation": "Equity 60%; Gold 30%; BTC 10%",
                "Universe": "US PIT optimized sleeve, TH PIT optimized sleeve, JP PIT optimized sleeve, Gold, BTC",
                "JP Optimizer": "min_vol_mom_tilt; top 10 PIT names; max 15% internal weight; 120 trading-day covariance lookback; minimum 40 training days; 63-day momentum tilt; concentration penalty 0.01; fallback equal weight for insufficient history.",
                "Japan Price Source": "Yahoo Finance daily adjusted close, downloaded at refresh time only for latest JP top10 universe plus 1306.T proxy.",
                "Japan Name Source": "J-Quants API equity master cached locally at data/cache/dynamic_factor_copula/japan_master_history.parquet.",
                "Equity Signal Rule": "US SPY MA300 + mom63; TH SET MA200 + mom63; JP Nikkei/proxy MA120 + mom63; scores shifted by one session",
                "Weekly Exposure": "Samples the already-lagged daily exposure on W-FRI and forward-fills. US SPY MA300 below50%; TH SET MA200 below0%; JP MA120 below0%; Gold DD252 warn-10%->75%, crash-20%->50%, panic-30% + below MA200 + mom63<0 -> 0%, recover-5%; BTC MA50 below0%.",
                "US Score": float(scores["US Equity"]),
                "TH Score": float(scores["TH Equity"]),
                "JP Score": float(scores["JP Equity"]),
                "Gold DD252": gold_dd,
                "US Internal Weight Date": us_internal_date,
                "TH Internal Weight Date": th_internal_date,
                "JP Internal Weight Date": jp_internal_date,
                "JP Universe Entry Date": jp_universe_entry_date,
                "Japan Yahoo Price End": pd.Timestamp(jp_prices.dropna(how="all").index.max()).date().isoformat(),
                "Japan Yahoo Missing Codes": ",".join(jp_missing_yahoo),
                "Overlay Cache End": pd.Timestamp(overlay.dropna().index.max()).date().isoformat(),
                "Latest Weight Source": "Standalone refresh from this repo's current data/cache and Yahoo Finance latest JP prices; no static latest-weight file is read from dynamic_port_opt. Effective date is capped at the common latest date across Yahoo JP prices and overlay assets.",
            }
        ]
    )

    _write_outputs(weekly_security, weekly_sleeve, meta)
    print(meta.to_string(index=False))
    print("\nWeekly sleeve weights")
    print(weekly_sleeve.to_string(index=False, float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()
