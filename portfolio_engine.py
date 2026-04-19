from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import yfinance as yf


TRADING_DAYS = 252
DEFAULT_BENCHMARK = "SPY"
DEFAULT_VOL_PROXY = "^VIX"
THAILAND_BENCHMARK = "^SET.BK"
THAILAND_VOL_PROXY = ""
EPSILON = 1e-12
SNAPSHOT_RULE_VERSION = "sp500_pit_top_liquidity_v1"
SNAPSHOT_DIR = Path(__file__).resolve().parent / "data" / "universe_snapshots"
US_LIQUID_LEADERS_SNAPSHOT_FILE = SNAPSHOT_DIR / "us_liquid_leaders.csv"
THAI_STOCK_DIR = Path(__file__).resolve().parent / "data" / "thai_stock"
SET100_INTERVAL_FILE = THAI_STOCK_DIR / "set100_ticker_start_end.csv"
SP500_REPO_DIR = Path(__file__).resolve().parent.parent / "sp500"
SP500_CURRENT_FILE = SP500_REPO_DIR / "sp500.csv"
SP500_TICKER_INTERVAL_FILE = SP500_REPO_DIR / "sp500_ticker_start_end.csv"
THAILAND_SET100_GROUP = "Thailand SET100"

PRESET_UNIVERSES: Dict[str, List[str]] = {
    "US Liquid Leaders": [
        "AAPL",
        "MSFT",
        "NVDA",
        "AMZN",
        "GOOGL",
        "META",
        "BRK-B",
        "LLY",
        "AVGO",
        "JPM",
        "XOM",
        "V",
        "COST",
        "UNH",
        "PG",
        "MA",
        "HD",
        "WMT",
        "NFLX",
        "ABBV",
        "KO",
        "PEP",
        "MRK",
        "JNJ",
        "CVX",
        "ORCL",
        "BAC",
        "AMD",
        "CRM",
        "CSCO",
        "TMO",
        "ACN",
        "IBM",
        "PM",
        "ABT",
        "LIN",
        "GE",
        "CAT",
        "MCD",
        "WFC",
        "DIS",
        "QCOM",
        "TXN",
        "INTU",
        "NOW",
        "AMAT",
        "GS",
        "MS",
        "RTX",
        "SCHW",
        "SPGI",
        "BKNG",
        "ISRG",
        "ADBE",
        "C",
        "LOW",
        "DHR",
        "BLK",
        "PFE",
        "HON",
        "ORLY",
        "DE",
        "ADI",
        "MMC",
        "VRTX",
        "USB",
        "AMGN",
        "CMCSA",
        "GILD",
        "TJX",
        "MDT",
        "MO",
        "NKE",
        "LRCX",
        "ELV",
        "PANW",
        "ADP",
        "T",
        "COP",
        "AMT",
    ],
    "Quality ETFs": [
        "QUAL",
        "SCHD",
        "USMV",
        "VIG",
        "DGRO",
        "DGRW",
        "NOBL",
        "SPHQ",
        "IUSG",
        "IUSV",
        "VTV",
        "VUG",
        "VYM",
        "FNDX",
        "FQAL",
        "MOAT",
        "RPV",
        "RPG",
        "EFG",
        "EFV",
        "IVE",
        "IVW",
    ],
    "Commodity ETFs": [
        "GLD",
        "IAU",
        "SGOL",
        "SLV",
        "SIVR",
        "DBC",
        "PDBC",
        "GSG",
        "COMT",
        "DBA",
        "FTGC",
        "GCC",
        "USCI",
        "RJI",
        "BCI",
        "CMDY",
        "JJG",
        "CORN",
        "WEAT",
        "SOYB",
    ],
    "Energy ETFs": [
        "XLE",
        "VDE",
        "IYE",
        "OIH",
        "XOP",
        "USO",
        "UNG",
        "AMLP",
        "MLPX",
        "PXE",
        "PXJ",
        "IEZ",
        "RYE",
        "FENY",
        "ICLN",
        "TAN",
        "QCLN",
        "PBW",
        "ACES",
        "LIT",
    ],
    "Gold-Silver Diversified": [
        "GLD",
        "IAU",
        "SGOL",
        "SLV",
        "SIVR",
        "GDX",
        "GDXJ",
        "SIL",
        "SILJ",
        "FNV",
        "WPM",
        "NEM",
        "AEM",
        "GOLD",
        "PAAS",
        "AGI",
        "KGC",
        "HL",
        "MAG",
        "CDE",
    ],
    "US Defensive": [
        "XLU",
        "XLP",
        "XLV",
        "VIG",
        "SCHD",
        "DGRO",
        "JNJ",
        "PG",
        "KO",
        "PEP",
        "WMT",
        "MCD",
        "CL",
        "GIS",
        "KMB",
        "MRK",
        "ABBV",
        "DUK",
        "SO",
        "ED",
    ],
    "US Small & Mid": [
        "IWM",
        "IJH",
        "MDY",
        "VB",
        "VBR",
        "IJR",
        "AVUV",
        "VXF",
        "VTWO",
        "SCHA",
        "VBK",
        "VOE",
        "VOT",
        "IJK",
        "IJS",
        "MDYG",
        "IJT",
        "SPSM",
        "RWJ",
        "TNA",
    ],
    "Sector Rotation": [
        "XLB",
        "XLE",
        "XLF",
        "XLI",
        "XLK",
        "XLP",
        "XLRE",
        "XLU",
        "XLV",
        "XLY",
        "XLC",
        "SMH",
        "IGV",
        "KRE",
        "XBI",
        "ITB",
        "XHB",
        "PAVE",
        "TAN",
        "ICLN",
    ],
    "Fixed Income": [
        "SHY",
        "IEF",
        "TLT",
        "TIP",
        "LQD",
        "HYG",
        "BND",
        "BIL",
        "IEI",
        "VGIT",
        "GOVT",
        "IGSB",
        "MINT",
        "SPSB",
        "VCSH",
        "VCIT",
        "JNK",
        "EMB",
        "BWX",
        "TIPX",
    ],
    "Real Assets": [
        "GLD",
        "DBC",
        "USO",
        "UNG",
        "DBA",
        "VNQ",
        "REET",
        "PDBC",
        "GSG",
        "COMT",
        "IAU",
        "SLV",
        "COPX",
        "WOOD",
        "MOO",
        "GCC",
        "FTGC",
        "TIP",
        "VNQI",
        "REM",
    ],
    "Global Macro": [
        "SPY",
        "QQQ",
        "EFA",
        "EEM",
        "EWJ",
        "EWY",
        "FXI",
        "TLT",
        "IEF",
        "HYG",
        "GLD",
        "USO",
        "UUP",
        "VNQ",
        "BIL",
        "TIP",
        "DBA",
        "VWO",
        "LQD",
        "DBC",
    ],
    "International Equities": [
        "EFA",
        "EEM",
        "VEA",
        "VWO",
        "EWJ",
        "EWY",
        "INDA",
        "FXI",
        "EWZ",
        "EWW",
        "EWU",
        "EWG",
        "EWC",
        "EWH",
        "EWT",
        "EIDO",
        "EPOL",
        "EZA",
        "TUR",
        "ARGT",
    ],
    "Crisis Hedges": [
        "GLD",
        "IAU",
        "TLT",
        "IEF",
        "TIP",
        "UUP",
        "FXF",
        "BIL",
        "SHY",
        "GOVT",
        "VGSH",
        "STIP",
        "ZROZ",
        "EDV",
        "BTAL",
        "DBMF",
        "KMLM",
        "SWAN",
        "PFIX",
        "SGOV",
    ],
    THAILAND_SET100_GROUP: [],
}


@dataclass
class AlphaConfig:
    top_n: int = 30
    target_holdings: int = 20
    liquidity_cut: int = 100
    min_history_ratio: float = 0.85
    min_avg_dollar_volume_millions: float = 0.0
    use_fundamental_factors: bool = False
    use_historical_eligibility: bool = True
    min_listing_days: int = 252 * 2
    momentum_weight: float = 0.35
    quality_weight: float = 0.20
    value_weight: float = 0.20
    growth_weight: float = 0.15
    low_vol_weight: float = 0.10


@dataclass
class ConstructionConfig:
    method: str = "Max Sharpe"
    min_weight: float = 0.0
    max_weight: float = 0.25
    risk_free_rate: float = 0.03
    covariance_shrinkage: float = 0.25
    alpha_strength: float = 0.20
    target_volatility: float = 0.18


@dataclass
class RiskConfig:
    use_trend_filter: bool = True
    use_regime_filter: bool = True
    max_drawdown_stop: float = 0.12
    bull_exposure: float = 1.00
    neutral_exposure: float = 0.65
    bear_exposure: float = 0.25


@dataclass
class ImplementationConfig:
    lookback_months: int = 12
    rebalance_months: int = 1
    transaction_cost_bps: float = 10.0
    slippage_bps: float = 2.0


@dataclass
class GovernanceConfig:
    assumed_aum_usd: float = 1_000_000.0
    adv_participation_limit: float = 0.05
    liquidation_days: int = 5


def sanitize_tickers(tickers: Iterable[str]) -> List[str]:
    clean: List[str] = []
    seen = set()
    for ticker in tickers:
        value = str(ticker).strip().upper()
        if value and value not in seen:
            clean.append(value)
            seen.add(value)
    return clean


def parse_ticker_text(text: str) -> List[str]:
    if not text:
        return []
    normalized = text.replace("\n", ",").replace(" ", ",")
    return sanitize_tickers(part for part in normalized.split(",") if part)


def normalize_symbol(symbol: object) -> str:
    value = str(symbol).strip().upper()
    if not value or value == "NAN":
        return ""
    return value.replace(".", "-")


def normalize_set_symbol(symbol: object) -> str:
    value = str(symbol).strip().upper()
    if not value or value == "NAN":
        return ""
    if not any(char.isalnum() for char in value):
        return ""
    if value.endswith(".BK"):
        return value
    return f"{value}.BK"


@lru_cache(maxsize=1)
def load_current_sp500_tickers() -> List[str]:
    if not SP500_CURRENT_FILE.exists():
        return []
    frame = pd.read_csv(SP500_CURRENT_FILE)
    symbol_column = "Symbol" if "Symbol" in frame.columns else "ticker"
    if symbol_column not in frame.columns:
        return []
    return sanitize_tickers(normalize_symbol(symbol) for symbol in frame[symbol_column].tolist())


@lru_cache(maxsize=1)
def load_sp500_membership_intervals() -> pd.DataFrame:
    if not SP500_TICKER_INTERVAL_FILE.exists():
        return pd.DataFrame(columns=["ticker", "start_date", "end_date"])

    intervals = pd.read_csv(SP500_TICKER_INTERVAL_FILE)
    required = {"ticker", "start_date", "end_date"}
    if not required.issubset(intervals.columns):
        return pd.DataFrame(columns=["ticker", "start_date", "end_date"])

    cleaned = intervals.loc[:, ["ticker", "start_date", "end_date"]].copy()
    cleaned["ticker"] = cleaned["ticker"].map(normalize_symbol)
    cleaned = cleaned.loc[cleaned["ticker"] != ""].copy()
    cleaned["start_date"] = pd.to_datetime(cleaned["start_date"], errors="coerce")
    cleaned["end_date"] = pd.to_datetime(cleaned["end_date"], errors="coerce")
    cleaned = cleaned.dropna(subset=["start_date"]).sort_values(["start_date", "ticker"]).reset_index(drop=True)
    return cleaned


def get_sp500_members_as_of(as_of_date: pd.Timestamp) -> List[str]:
    intervals = load_sp500_membership_intervals()
    if intervals.empty:
        return []

    timestamp = pd.Timestamp(as_of_date).normalize()
    members = intervals.loc[
        (intervals["start_date"] <= timestamp)
        & (intervals["end_date"].isna() | (intervals["end_date"] >= timestamp)),
        "ticker",
    ].tolist()
    return sanitize_tickers(members)


@lru_cache(maxsize=1)
def load_set100_membership_intervals() -> pd.DataFrame:
    if not SET100_INTERVAL_FILE.exists():
        return pd.DataFrame(columns=["ticker", "start_date", "end_date"])

    intervals = pd.read_csv(SET100_INTERVAL_FILE)
    required = {"ticker", "start_date", "end_date"}
    if not required.issubset(intervals.columns):
        return pd.DataFrame(columns=["ticker", "start_date", "end_date"])

    cleaned = intervals.loc[:, ["ticker", "start_date", "end_date"]].copy()
    cleaned["ticker"] = cleaned["ticker"].map(normalize_set_symbol)
    cleaned = cleaned.loc[cleaned["ticker"] != ""].copy()
    cleaned["start_date"] = pd.to_datetime(cleaned["start_date"], errors="coerce")
    cleaned["end_date"] = pd.to_datetime(cleaned["end_date"], errors="coerce")
    cleaned = cleaned.dropna(subset=["start_date", "end_date"]).sort_values(["start_date", "ticker"]).reset_index(drop=True)
    return cleaned


def get_set100_members_as_of(as_of_date: pd.Timestamp) -> List[str]:
    intervals = load_set100_membership_intervals()
    if intervals.empty:
        return []

    timestamp = pd.Timestamp(as_of_date).normalize()
    members = intervals.loc[
        (intervals["start_date"] <= timestamp)
        & (intervals["end_date"] >= timestamp),
        "ticker",
    ].tolist()
    return sanitize_tickers(members)


def load_all_set100_tickers() -> List[str]:
    intervals = load_set100_membership_intervals()
    if intervals.empty:
        return []
    return sanitize_tickers(intervals["ticker"].tolist())


def infer_market_reference(selected_groups: Sequence[str], selected_tickers: Sequence[str]) -> Dict[str, str]:
    return {
        "benchmark": DEFAULT_BENCHMARK,
        "vol_proxy": DEFAULT_VOL_PROXY,
    }


current_sp500_tickers = load_current_sp500_tickers()
if current_sp500_tickers:
    PRESET_UNIVERSES["US Liquid Leaders"] = current_sp500_tickers

all_set100_tickers = load_all_set100_tickers()
if all_set100_tickers:
    PRESET_UNIVERSES[THAILAND_SET100_GROUP] = all_set100_tickers


def _extract_field(frame: pd.DataFrame, field: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    if isinstance(frame.columns, pd.MultiIndex):
        if field in frame.columns.get_level_values(0):
            out = frame[field].copy()
        else:
            return pd.DataFrame()
    else:
        out = frame.copy()
    if isinstance(out, pd.Series):
        out = out.to_frame()
    out.columns = [str(col).upper() for col in out.columns]
    return out.sort_index()


def download_market_data(
    tickers: Sequence[str],
    start_date: str,
    end_date: str,
    benchmark: str = DEFAULT_BENCHMARK,
    vol_proxy: str = DEFAULT_VOL_PROXY,
) -> Dict[str, pd.DataFrame]:
    universe = sanitize_tickers(tickers)
    if not universe:
        raise ValueError("No tickers were provided.")

    request = sanitize_tickers(list(universe) + [benchmark, vol_proxy])
    raw = yf.download(
        request,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
        threads=True,
        group_by="column",
    )
    if raw.empty:
        raise ValueError("yfinance returned no data for the selected period.")

    close = _extract_field(raw, "Close").ffill()
    volume = _extract_field(raw, "Volume").fillna(0.0)

    missing = [ticker for ticker in universe if ticker not in close.columns]
    available = [ticker for ticker in universe if ticker in close.columns]
    if not available:
        raise ValueError("No selected tickers returned valid price history.")

    prices = close[available].copy().dropna(how="all")
    volumes = volume.reindex(prices.index).reindex(columns=available).fillna(0.0)
    benchmark_series = close[benchmark].dropna() if benchmark in close.columns else pd.Series(dtype=float)
    vol_proxy_series = close[vol_proxy].dropna() if vol_proxy in close.columns else pd.Series(dtype=float)

    return {
        "prices": prices,
        "volumes": volumes,
        "benchmark": benchmark_series,
        "benchmark_symbol": benchmark,
        "vol_proxy": vol_proxy_series,
        "vol_proxy_symbol": vol_proxy,
        "missing": pd.DataFrame({"Ticker": missing}),
    }


def fetch_fundamentals(tickers: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for ticker in sanitize_tickers(tickers):
        try:
            info = yf.Ticker(ticker).info
        except Exception:
            info = {}
        rows.append(
            {
                "ticker": ticker,
                "quote_type": str(info.get("quoteType", "")).upper(),
                "sector": str(info.get("sector", "")).strip(),
                "industry": str(info.get("industry", "")).strip(),
                "trailing_pe": _safe_number(info.get("trailingPE")),
                "price_to_book": _safe_number(info.get("priceToBook")),
                "return_on_equity": _safe_number(info.get("returnOnEquity")),
                "operating_margin": _safe_number(info.get("operatingMargins")),
                "earnings_growth": _safe_number(info.get("earningsGrowth")),
                "revenue_growth": _safe_number(info.get("revenueGrowth")),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("ticker")


def _safe_number(value: object) -> float:
    try:
        if value is None:
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def _cross_sectional_zscore(series: pd.Series) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if clean.dropna().empty:
        return pd.Series(0.0, index=series.index)
    std = clean.std(ddof=0)
    if pd.isna(std) or std < EPSILON:
        return pd.Series(0.0, index=series.index)
    return ((clean - clean.mean()) / std).fillna(0.0)


def ensure_snapshot_dir() -> None:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)


def snapshot_date_key(as_of_date: pd.Timestamp) -> str:
    return pd.Timestamp(as_of_date).normalize().strftime("%Y-%m-%d")


def load_us_liquid_leaders_snapshot(as_of_date: pd.Timestamp) -> pd.DataFrame:
    if not US_LIQUID_LEADERS_SNAPSHOT_FILE.exists():
        return pd.DataFrame()

    snapshot = pd.read_csv(US_LIQUID_LEADERS_SNAPSHOT_FILE)
    if snapshot.empty:
        return snapshot
    key = snapshot_date_key(as_of_date)
    current = snapshot.loc[
        (snapshot["universe_name"] == "US Liquid Leaders")
        & (snapshot["rebalance_date"] == key)
        & (snapshot["selection_rule_version"] == SNAPSHOT_RULE_VERSION)
    ].copy()
    if current.empty:
        return current

    for column in ["avg_dollar_volume_m", "liquidity_rank"]:
        current[column] = pd.to_numeric(current[column], errors="coerce")
    if "membership_count" in current.columns:
        current["membership_count"] = pd.to_numeric(current["membership_count"], errors="coerce")
    for column in ["lookback_start", "lookback_end", "first_valid_date", "rebalance_date"]:
        current[column] = pd.to_datetime(current[column], errors="coerce")
    current["snapshot_source"] = "csv"
    return current.sort_values("liquidity_rank")


def save_us_liquid_leaders_snapshot(snapshot_rows: pd.DataFrame) -> None:
    if snapshot_rows.empty:
        return

    ensure_snapshot_dir()
    output = snapshot_rows.copy()
    if "snapshot_source" in output.columns:
        output = output.drop(columns=["snapshot_source"])
    if US_LIQUID_LEADERS_SNAPSHOT_FILE.exists():
        existing = pd.read_csv(US_LIQUID_LEADERS_SNAPSHOT_FILE)
        existing = existing.loc[
            ~(
                (existing["universe_name"] == "US Liquid Leaders")
                & (existing["rebalance_date"] == output["rebalance_date"].iloc[0])
                & (existing["selection_rule_version"] == SNAPSHOT_RULE_VERSION)
            )
        ].copy()
        output = pd.concat([existing, output], ignore_index=True)
    output.to_csv(US_LIQUID_LEADERS_SNAPSHOT_FILE, index=False)


def build_us_liquid_leaders_snapshot(
    prices_window: pd.DataFrame,
    volumes_window: pd.DataFrame,
    full_prices: pd.DataFrame,
    as_of_date: pd.Timestamp,
) -> pd.DataFrame:
    pit_members = get_sp500_members_as_of(as_of_date)
    if pit_members:
        universe = [ticker for ticker in pit_members if ticker in prices_window.columns]
        membership_source = "sp500_ticker_start_end"
    else:
        universe = [ticker for ticker in PRESET_UNIVERSES["US Liquid Leaders"] if ticker in prices_window.columns]
        membership_source = "current_sp500_fallback"
    if not universe:
        return pd.DataFrame()

    working_prices = prices_window[universe].ffill()
    working_volumes = volumes_window.reindex(working_prices.index).reindex(columns=universe).fillna(0.0)
    avg_dollar_volume_m = (working_prices * working_volumes).mean() / 1_000_000.0
    first_valid_dates = full_prices[universe].apply(lambda series: series.first_valid_index())

    snapshot = pd.DataFrame(
        {
            "universe_name": "US Liquid Leaders",
            "rebalance_date": snapshot_date_key(as_of_date),
            "ticker": avg_dollar_volume_m.index,
            "liquidity_rank": avg_dollar_volume_m.rank(ascending=False, method="dense"),
            "avg_dollar_volume_m": avg_dollar_volume_m.values,
            "lookback_start": prices_window.index.min(),
            "lookback_end": prices_window.index.max(),
            "first_valid_date": first_valid_dates.values,
            "membership_source": membership_source,
            "membership_count": len(universe),
            "selection_rule_version": SNAPSHOT_RULE_VERSION,
        }
    ).sort_values(["liquidity_rank", "ticker"])
    save_us_liquid_leaders_snapshot(snapshot)
    snapshot["snapshot_source"] = "computed"
    return snapshot


def get_or_create_us_liquid_leaders_snapshot(
    prices_window: pd.DataFrame,
    volumes_window: pd.DataFrame,
    full_prices: pd.DataFrame,
    as_of_date: pd.Timestamp,
) -> pd.DataFrame:
    current = load_us_liquid_leaders_snapshot(as_of_date)
    if not current.empty:
        return current
    return build_us_liquid_leaders_snapshot(prices_window, volumes_window, full_prices, as_of_date)


def apply_us_liquid_leaders_snapshot(
    prices_window: pd.DataFrame,
    volumes_window: pd.DataFrame,
    full_prices: pd.DataFrame,
    as_of_date: pd.Timestamp,
    alpha_cfg: AlphaConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    us_liquid_tickers = [ticker for ticker in prices_window.columns if ticker in PRESET_UNIVERSES["US Liquid Leaders"]]
    if not us_liquid_tickers:
        return prices_window, volumes_window, pd.DataFrame()

    snapshot = get_or_create_us_liquid_leaders_snapshot(
        prices_window=prices_window[us_liquid_tickers],
        volumes_window=volumes_window.reindex(columns=us_liquid_tickers),
        full_prices=full_prices.reindex(columns=us_liquid_tickers),
        as_of_date=as_of_date,
    )
    liquidity_cut = max(int(alpha_cfg.liquidity_cut), 1)
    filtered_snapshot = snapshot.sort_values("liquidity_rank").head(liquidity_cut).copy()
    ordered_tickers = [
        ticker
        for ticker in filtered_snapshot["ticker"].tolist()
        if ticker in prices_window.columns
    ]
    if alpha_cfg.use_historical_eligibility:
        cutoff = pd.Timestamp(as_of_date) - pd.Timedelta(days=int(alpha_cfg.min_listing_days))
        eligible_from_snapshot = filtered_snapshot.loc[
            pd.to_datetime(filtered_snapshot["first_valid_date"], errors="coerce") <= cutoff, "ticker"
        ].tolist()
        ordered_tickers = [ticker for ticker in ordered_tickers if ticker in eligible_from_snapshot]

    other_tickers = [ticker for ticker in prices_window.columns if ticker not in us_liquid_tickers]
    final_columns = ordered_tickers + other_tickers
    return (
        prices_window.reindex(columns=final_columns),
        volumes_window.reindex(columns=final_columns),
        filtered_snapshot,
    )


def apply_thailand_set100_snapshot(
    prices_window: pd.DataFrame,
    volumes_window: pd.DataFrame,
    full_prices: pd.DataFrame,
    as_of_date: pd.Timestamp,
    alpha_cfg: AlphaConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    thai_tickers = [
        ticker
        for ticker in prices_window.columns
        if ticker in PRESET_UNIVERSES.get(THAILAND_SET100_GROUP, [])
    ]
    if not thai_tickers:
        return prices_window, volumes_window, pd.DataFrame()

    pit_members = get_set100_members_as_of(as_of_date)
    universe = [ticker for ticker in thai_tickers if ticker in pit_members]
    if not universe:
        other_tickers = [ticker for ticker in prices_window.columns if ticker not in thai_tickers]
        return prices_window.reindex(columns=other_tickers), volumes_window.reindex(columns=other_tickers), pd.DataFrame()

    working_prices = prices_window[universe].ffill()
    working_volumes = volumes_window.reindex(working_prices.index).reindex(columns=universe).fillna(0.0)
    avg_dollar_volume_m = (working_prices * working_volumes).mean() / 1_000_000.0
    first_valid_dates = full_prices[universe].apply(lambda series: series.first_valid_index())

    snapshot = pd.DataFrame(
        {
            "universe_name": THAILAND_SET100_GROUP,
            "rebalance_date": snapshot_date_key(as_of_date),
            "ticker": avg_dollar_volume_m.index,
            "liquidity_rank": avg_dollar_volume_m.rank(ascending=False, method="dense"),
            "avg_dollar_volume_m": avg_dollar_volume_m.values,
            "lookback_start": prices_window.index.min(),
            "lookback_end": prices_window.index.max(),
            "first_valid_date": first_valid_dates.values,
            "membership_source": "set100_ticker_start_end",
            "membership_count": len(universe),
            "selection_rule_version": "set100_pit_top_liquidity_v1",
            "snapshot_source": "computed",
        }
    ).sort_values(["liquidity_rank", "ticker"])

    liquidity_cut = max(int(alpha_cfg.liquidity_cut), 1)
    filtered_snapshot = snapshot.head(liquidity_cut).copy()
    ordered_tickers = [
        ticker
        for ticker in filtered_snapshot["ticker"].tolist()
        if ticker in prices_window.columns
    ]
    if alpha_cfg.use_historical_eligibility:
        cutoff = pd.Timestamp(as_of_date) - pd.Timedelta(days=int(alpha_cfg.min_listing_days))
        eligible_from_snapshot = filtered_snapshot.loc[
            pd.to_datetime(filtered_snapshot["first_valid_date"], errors="coerce") <= cutoff, "ticker"
        ].tolist()
        ordered_tickers = [ticker for ticker in ordered_tickers if ticker in eligible_from_snapshot]

    other_tickers = [ticker for ticker in prices_window.columns if ticker not in thai_tickers]
    final_columns = ordered_tickers + other_tickers
    return (
        prices_window.reindex(columns=final_columns),
        volumes_window.reindex(columns=final_columns),
        filtered_snapshot,
    )


def apply_point_in_time_universe_filters(
    prices_window: pd.DataFrame,
    volumes_window: pd.DataFrame,
    full_prices: pd.DataFrame,
    as_of_date: pd.Timestamp,
    alpha_cfg: AlphaConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    working_prices = prices_window
    working_volumes = volumes_window
    snapshots: List[pd.DataFrame] = []

    for filter_fn in (apply_us_liquid_leaders_snapshot, apply_thailand_set100_snapshot):
        working_prices, working_volumes, snapshot = filter_fn(
            prices_window=working_prices,
            volumes_window=working_volumes,
            full_prices=full_prices,
            as_of_date=as_of_date,
            alpha_cfg=alpha_cfg,
        )
        if not snapshot.empty:
            snapshots.append(snapshot)

    if not snapshots:
        return working_prices, working_volumes, pd.DataFrame()
    combined = pd.concat(snapshots, ignore_index=True)
    return working_prices, working_volumes, combined


def filter_historical_universe(
    prices_window: pd.DataFrame,
    full_prices: pd.DataFrame,
    as_of_date: pd.Timestamp,
    alpha_cfg: AlphaConfig,
) -> pd.DataFrame:
    if prices_window.empty or not alpha_cfg.use_historical_eligibility:
        return prices_window

    cutoff = pd.Timestamp(as_of_date) - pd.Timedelta(days=int(alpha_cfg.min_listing_days))
    eligible_columns: List[str] = []
    for column in prices_window.columns:
        first_valid = full_prices[column].first_valid_index() if column in full_prices.columns else None
        if first_valid is None or pd.isna(first_valid):
            continue
        if pd.Timestamp(first_valid) <= cutoff:
            eligible_columns.append(column)
    return prices_window.loc[:, eligible_columns]


def compute_alpha_table(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    alpha_cfg: AlphaConfig,
    fundamentals: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    if prices.empty:
        return pd.DataFrame()

    coverage = prices.notna().mean()
    avg_dollar_volume = (prices * volumes.reindex_like(prices).fillna(0.0)).mean() / 1_000_000.0
    eligible = coverage >= alpha_cfg.min_history_ratio
    if alpha_cfg.min_avg_dollar_volume_millions > 0:
        eligible &= avg_dollar_volume >= alpha_cfg.min_avg_dollar_volume_millions

    working = prices.loc[:, eligible[eligible].index].copy()
    if working.empty:
        return pd.DataFrame()
    working = working.ffill().dropna(axis=1, how="any")
    if working.empty:
        return pd.DataFrame()

    returns = working.pct_change(fill_method=None).dropna()
    if returns.empty:
        return pd.DataFrame()

    lookbacks = {
        "momentum_3m": 63,
        "momentum_6m": 126,
        "momentum_12m": 252,
    }
    raw: Dict[str, pd.Series] = {}
    for name, days in lookbacks.items():
        if len(working) > days:
            raw[name] = working.iloc[-1] / working.iloc[-days - 1] - 1.0
        else:
            raw[name] = returns.mean() * min(days, len(returns))

    volatility = returns.std() * np.sqrt(TRADING_DAYS)
    current_drawdown = working.iloc[-1] / working.cummax().iloc[-1] - 1.0
    trend_ratio = working.iloc[-1] / working.rolling(200, min_periods=40).mean().iloc[-1] - 1.0

    signal_frame = pd.DataFrame(raw)
    signal_frame["volatility"] = volatility
    signal_frame["current_drawdown"] = current_drawdown
    signal_frame["trend_ratio"] = trend_ratio
    signal_frame["coverage"] = coverage.reindex(signal_frame.index)
    signal_frame["avg_dollar_volume_m"] = avg_dollar_volume.reindex(signal_frame.index).fillna(0.0)

    momentum_blend = (
        0.2 * signal_frame["momentum_3m"]
        + 0.35 * signal_frame["momentum_6m"]
        + 0.45 * signal_frame["momentum_12m"]
    )
    quality_signal = pd.Series(0.0, index=signal_frame.index)
    value_signal = pd.Series(0.0, index=signal_frame.index)
    growth_signal = pd.Series(0.0, index=signal_frame.index)
    quality_available = pd.Series(False, index=signal_frame.index)
    value_available = pd.Series(False, index=signal_frame.index)
    growth_available = pd.Series(False, index=signal_frame.index)

    if alpha_cfg.use_fundamental_factors and fundamentals is not None and not fundamentals.empty:
        aligned = fundamentals.reindex(signal_frame.index)
        excluded_types = {"ETF", "MUTUALFUND", "INDEX", "CRYPTOCURRENCY", "CURRENCY", "FUTURE"}
        quote_type = aligned.get("quote_type", pd.Series("", index=aligned.index)).fillna("").astype(str).str.upper()
        allow_fundamentals = ~quote_type.isin(excluded_types)

        raw_quality = aligned[["return_on_equity", "operating_margin"]].mean(axis=1, skipna=True)
        raw_value = pd.concat(
            [
                1.0 / aligned["trailing_pe"].replace(0.0, np.nan),
                1.0 / aligned["price_to_book"].replace(0.0, np.nan),
            ],
            axis=1,
        ).mean(axis=1, skipna=True)
        raw_growth = aligned[["earnings_growth", "revenue_growth"]].mean(axis=1, skipna=True)

        quality_available = allow_fundamentals & raw_quality.notna()
        value_available = allow_fundamentals & raw_value.notna()
        growth_available = allow_fundamentals & raw_growth.notna()

        quality_signal = raw_quality.where(quality_available, np.nan)
        value_signal = raw_value.where(value_available, np.nan)
        growth_signal = raw_growth.where(growth_available, np.nan)
        signal_frame = signal_frame.join(aligned)

    signal_frame["momentum_score"] = _cross_sectional_zscore(momentum_blend)
    signal_frame["quality_score"] = _cross_sectional_zscore(quality_signal)
    signal_frame["value_score"] = _cross_sectional_zscore(value_signal)
    signal_frame["growth_score"] = _cross_sectional_zscore(growth_signal)
    signal_frame["low_vol_score"] = _cross_sectional_zscore(-signal_frame["volatility"])
    signal_frame["trend_score"] = _cross_sectional_zscore(signal_frame["trend_ratio"])

    component_scores = pd.DataFrame(
        {
            "momentum": signal_frame["momentum_score"],
            "quality": signal_frame["quality_score"],
            "value": signal_frame["value_score"],
            "growth": signal_frame["growth_score"],
            "low_vol": signal_frame["low_vol_score"],
            "trend": signal_frame["trend_score"],
        },
        index=signal_frame.index,
    )
    component_weights = pd.DataFrame(
        {
            "momentum": alpha_cfg.momentum_weight,
            "quality": alpha_cfg.quality_weight,
            "value": alpha_cfg.value_weight,
            "growth": alpha_cfg.growth_weight,
            "low_vol": alpha_cfg.low_vol_weight,
            "trend": 0.10,
        },
        index=signal_frame.index,
    )
    component_available = pd.DataFrame(
        {
            "momentum": True,
            "quality": quality_available if alpha_cfg.use_fundamental_factors else False,
            "value": value_available if alpha_cfg.use_fundamental_factors else False,
            "growth": growth_available if alpha_cfg.use_fundamental_factors else False,
            "low_vol": True,
            "trend": True,
        },
        index=signal_frame.index,
    )
    active_weights = component_weights.where(component_available, 0.0)
    active_weight_sum = active_weights.sum(axis=1).replace(0.0, np.nan)
    composite = (component_scores.mul(active_weights).sum(axis=1) / active_weight_sum).fillna(0.0)

    signal_frame["fundamental_allowed"] = component_available[["quality", "value", "growth"]].any(axis=1)
    signal_frame["active_factor_weight"] = active_weight_sum.fillna(0.0)
    signal_frame["composite_score"] = composite
    signal_frame["rank"] = signal_frame["composite_score"].rank(ascending=False, method="dense")
    return signal_frame.sort_values("composite_score", ascending=False)


def _sector_cap_for_target_holdings(target_holdings: int) -> int:
    if target_holdings <= 6:
        return 2
    if target_holdings <= 15:
        return 3
    return 4


def select_diversified_holdings(
    candidate_table: pd.DataFrame,
    candidate_prices: pd.DataFrame,
    target_holdings: int,
    fundamentals: Optional[pd.DataFrame] = None,
) -> List[str]:
    if candidate_table.empty or target_holdings <= 0:
        return []

    ordered = [ticker for ticker in candidate_table.index.tolist() if ticker in candidate_prices.columns]
    if not ordered:
        return []

    target = min(int(target_holdings), len(ordered))
    sector_cap = _sector_cap_for_target_holdings(target)

    sector_map: Dict[str, str] = {}
    if fundamentals is not None and not fundamentals.empty and "sector" in fundamentals.columns:
        sectors = fundamentals.reindex(ordered)["sector"].fillna("").astype(str).str.strip()
        sector_map = {
            ticker: sector
            for ticker, sector in sectors.items()
            if sector
        }

    clean_prices = candidate_prices.reindex(columns=ordered).ffill()
    corr = clean_prices.pct_change(fill_method=None).dropna().corr() if len(clean_prices) >= 3 else pd.DataFrame()

    selected: List[str] = []
    sector_counts: Dict[str, int] = {}
    correlation_thresholds = (0.80, 0.88, 0.94, 1.01)

    for threshold in correlation_thresholds:
        for ticker in ordered:
            if ticker in selected:
                continue

            sector = sector_map.get(ticker, "")
            if sector and sector_counts.get(sector, 0) >= sector_cap:
                continue

            if selected and not corr.empty and ticker in corr.index:
                pairwise = corr.loc[ticker, selected].abs().dropna()
                if not pairwise.empty and float(pairwise.max()) > threshold:
                    continue

            selected.append(ticker)
            if sector:
                sector_counts[sector] = sector_counts.get(sector, 0) + 1
            if len(selected) >= target:
                return selected

    for enforce_sector_cap in (True, False):
        for ticker in ordered:
            if ticker in selected:
                continue
            sector = sector_map.get(ticker, "")
            if enforce_sector_cap and sector and sector_counts.get(sector, 0) >= sector_cap:
                continue
            selected.append(ticker)
            if sector:
                sector_counts[sector] = sector_counts.get(sector, 0) + 1
            if len(selected) >= target:
                return selected

    return selected


def detect_regime(
    benchmark_prices: pd.Series,
    universe_prices: pd.DataFrame,
    vol_proxy_prices: Optional[pd.Series],
) -> Dict[str, float | str]:
    benchmark = benchmark_prices.dropna().copy()
    if benchmark.empty:
        return {
            "regime": "Neutral",
            "score": 0.5,
            "trend_score": 0.5,
            "breadth_score": 0.5,
            "vol_score": 0.5,
            "drawdown_score": 0.5,
        }

    trend_ma = benchmark.rolling(200, min_periods=40).mean().iloc[-1]
    trend_score = 1.0 if benchmark.iloc[-1] > trend_ma else 0.0

    breadth_prices = universe_prices.ffill().dropna(axis=1, how="all")
    if breadth_prices.empty:
        breadth_score = 0.5
    else:
        breadth_ma = breadth_prices.rolling(200, min_periods=40).mean().iloc[-1]
        breadth_score = float((breadth_prices.iloc[-1] > breadth_ma).mean())

    rolling_peak = benchmark.cummax().iloc[-1]
    drawdown = benchmark.iloc[-1] / rolling_peak - 1.0
    if drawdown > -0.08:
        drawdown_score = 1.0
    elif drawdown > -0.15:
        drawdown_score = 0.5
    else:
        drawdown_score = 0.0

    vol_score = 0.5
    if vol_proxy_prices is not None and not vol_proxy_prices.dropna().empty:
        vol_now = float(vol_proxy_prices.dropna().iloc[-1])
        if vol_now < 20:
            vol_score = 1.0
        elif vol_now < 28:
            vol_score = 0.5
        else:
            vol_score = 0.0

    score = float(np.mean([trend_score, breadth_score, vol_score, drawdown_score]))
    if score >= 0.70:
        regime = "Bull"
    elif score >= 0.40:
        regime = "Neutral"
    else:
        regime = "Bear"

    return {
        "regime": regime,
        "score": score,
        "trend_score": trend_score,
        "breadth_score": breadth_score,
        "vol_score": vol_score,
        "drawdown_score": drawdown_score,
    }


def estimate_capital_market_inputs(
    prices: pd.DataFrame,
    alpha_scores: Optional[pd.Series],
    construction_cfg: ConstructionConfig,
) -> tuple[pd.Series, pd.DataFrame]:
    clean_prices = prices.ffill().dropna(axis=1, how="any")
    returns = clean_prices.pct_change(fill_method=None).dropna()
    if returns.empty:
        raise ValueError("Not enough return history to estimate optimization inputs.")

    mu = returns.mean() * TRADING_DAYS
    if alpha_scores is not None and not alpha_scores.empty:
        alpha_tilt = _cross_sectional_zscore(alpha_scores.reindex(mu.index)).fillna(0.0)
        mu = mu + construction_cfg.alpha_strength * 0.05 * alpha_tilt

    cov = returns.cov() * TRADING_DAYS
    diagonal = np.diag(np.diag(cov.values))
    shrunk = (1.0 - construction_cfg.covariance_shrinkage) * cov.values + construction_cfg.covariance_shrinkage * diagonal
    cov = pd.DataFrame(shrunk, index=cov.index, columns=cov.columns)
    return mu, cov


def _normalize_weights(weights: np.ndarray, min_weight: float, max_weight: float) -> np.ndarray:
    clipped = np.clip(weights, min_weight, max_weight)
    total = clipped.sum()
    if total <= EPSILON:
        return np.repeat(1.0 / len(clipped), len(clipped))
    return clipped / total


def optimize_weights(
    prices: pd.DataFrame,
    construction_cfg: ConstructionConfig,
    alpha_scores: Optional[pd.Series] = None,
) -> Dict[str, object]:
    mu, cov = estimate_capital_market_inputs(prices, alpha_scores, construction_cfg)
    assets = list(mu.index)
    n_assets = len(assets)
    bounds = [(construction_cfg.min_weight, construction_cfg.max_weight) for _ in assets]
    guess = np.repeat(1.0 / n_assets, n_assets)
    guess = _normalize_weights(guess, construction_cfg.min_weight, construction_cfg.max_weight)
    constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1.0}]

    cov_matrix = cov.values
    mu_vector = mu.values

    def portfolio_volatility(weights: np.ndarray) -> float:
        return float(np.sqrt(np.maximum(weights @ cov_matrix @ weights, EPSILON)))

    def portfolio_return(weights: np.ndarray) -> float:
        return float(weights @ mu_vector)

    method_name = construction_cfg.method

    if method_name == "Equal Weight":
        final_weights = pd.Series(np.repeat(1.0 / n_assets, n_assets), index=assets)
    else:
        if method_name == "Max Sharpe":
            objective = lambda x: -(
                (portfolio_return(x) - construction_cfg.risk_free_rate)
                / np.maximum(portfolio_volatility(x), EPSILON)
            )
        elif method_name == "Min Volatility":
            objective = lambda x: portfolio_volatility(x)
        elif method_name == "Risk Parity":
            def objective(x: np.ndarray) -> float:
                port_vol = portfolio_volatility(x)
                marginal = cov_matrix @ x / np.maximum(port_vol, EPSILON)
                contribution = x * marginal
                target = port_vol / n_assets
                return float(np.sum((contribution - target) ** 2))
        else:
            raise ValueError(f"Unsupported optimization method: {method_name}")

        result = minimize(
            objective,
            guess,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-9, "disp": False},
        )

        if not result.success:
            fallback = np.repeat(1.0 / n_assets, n_assets)
            final_weights = pd.Series(_normalize_weights(fallback, construction_cfg.min_weight, construction_cfg.max_weight), index=assets)
        else:
            final_weights = pd.Series(
                _normalize_weights(result.x, construction_cfg.min_weight, construction_cfg.max_weight),
                index=assets,
            )

    expected_return = float(final_weights.values @ mu_vector)
    expected_volatility = float(np.sqrt(np.maximum(final_weights.values @ cov_matrix @ final_weights.values, EPSILON)))
    sharpe = (expected_return - construction_cfg.risk_free_rate) / np.maximum(expected_volatility, EPSILON)
    return {
        "weights": final_weights.sort_values(ascending=False),
        "expected_return": expected_return,
        "expected_volatility": expected_volatility,
        "expected_sharpe": float(sharpe),
        "mu": mu,
        "cov": cov,
    }


def calculate_performance_metrics(portfolio_curve: pd.DataFrame, risk_free_rate: float) -> pd.DataFrame:
    if portfolio_curve.empty or "PortValue" not in portfolio_curve.columns:
        return pd.DataFrame()

    values = portfolio_curve["PortValue"].dropna().astype(float)
    if len(values) < 2:
        return pd.DataFrame()

    returns = values.pct_change().dropna()
    total_return = values.iloc[-1] / values.iloc[0] - 1.0
    years = max(len(values) / TRADING_DAYS, 1.0 / TRADING_DAYS)
    cagr = values.iloc[-1] ** (1.0 / years) / values.iloc[0] ** (1.0 / years) - 1.0
    annual_vol = returns.std(ddof=0) * np.sqrt(TRADING_DAYS)
    sharpe = ((returns.mean() * TRADING_DAYS) - risk_free_rate) / np.maximum(annual_vol, EPSILON)
    downside = returns[returns < 0].std(ddof=0) * np.sqrt(TRADING_DAYS)
    sortino = ((returns.mean() * TRADING_DAYS) - risk_free_rate) / np.maximum(downside, EPSILON)
    running_peak = values.cummax()
    drawdown = values / running_peak - 1.0
    max_drawdown = float(drawdown.min())
    hit_rate = float((returns > 0).mean())

    metrics = {
        "Total Return": total_return,
        "CAGR": cagr,
        "Annual Volatility": float(annual_vol),
        "Sharpe": float(sharpe),
        "Sortino": float(sortino),
        "Max Drawdown": max_drawdown,
        "Hit Rate": hit_rate,
    }
    return pd.DataFrame({"Metric": list(metrics.keys()), "Value": list(metrics.values())})


def simulate_retirement_paths_from_returns(
    sampled_returns: np.ndarray,
    initial_portfolio: float,
    monthly_withdrawal: float,
    annual_inflation: float = 0.03,
    monthly_income: float = 0.0,
) -> Dict[str, object]:
    periods, num_scenarios = sampled_returns.shape
    paths = np.zeros((periods + 1, num_scenarios), dtype=float)
    gross_withdrawals = np.zeros((periods + 1, num_scenarios), dtype=float)
    net_withdrawals = np.zeros((periods + 1, num_scenarios), dtype=float)
    paths[0, :] = float(initial_portfolio)

    inflation_step = (1 + annual_inflation) ** (1 / 12) - 1 if annual_inflation > -1 else 0.0
    current_withdrawal = float(monthly_withdrawal)
    current_income = float(monthly_income)

    for period in range(1, periods + 1):
        grown = paths[period - 1, :] * (1 + sampled_returns[period - 1, :])
        gross = np.full(num_scenarios, current_withdrawal, dtype=float)
        net = np.maximum(gross - current_income, 0.0)
        ending = np.maximum(grown - net, 0.0)

        gross_withdrawals[period, :] = gross
        net_withdrawals[period, :] = net
        paths[period, :] = ending
        current_withdrawal *= 1 + inflation_step
        current_income *= 1 + inflation_step

    simulated_paths = pd.DataFrame(paths)
    withdrawal_df = pd.DataFrame(gross_withdrawals)
    final_values = simulated_paths.iloc[-1]
    survival_rate = float((final_values > 0).mean())

    return {
        "simulated_paths": simulated_paths,
        "gross_withdrawals": withdrawal_df,
        "final_values": final_values,
        "survival_rate": survival_rate,
        "sampled_returns": sampled_returns,
    }


def block_bootstrap_monthly_returns(
    monthly_returns: pd.Series,
    periods: int,
    num_scenarios: int,
    block_size: int = 12,
    seed: int = 42,
) -> np.ndarray:
    clean = pd.Series(monthly_returns).dropna().astype(float)
    if clean.empty:
        raise ValueError("Monthly returns are required for retirement simulation.")

    values = clean.to_numpy()
    block_size = max(int(block_size), 1)
    if len(values) <= block_size:
        blocks = [values.copy()]
    else:
        blocks = [values[i : i + block_size] for i in range(len(values) - block_size + 1)]

    rng = np.random.default_rng(seed)
    samples = np.zeros((periods, num_scenarios), dtype=float)
    for scenario in range(num_scenarios):
        path_parts: List[np.ndarray] = []
        length = 0
        while length < periods:
            chosen = blocks[int(rng.integers(0, len(blocks)))]
            path_parts.append(chosen)
            length += len(chosen)
        samples[:, scenario] = np.concatenate(path_parts)[:periods]
    return samples


def simulate_retirement_paths(
    monthly_returns: pd.Series,
    initial_portfolio: float,
    monthly_withdrawal: float,
    years: int,
    annual_inflation: float = 0.03,
    monthly_income: float = 0.0,
    num_scenarios: int = 1000,
    block_size: int = 12,
    seed: int = 42,
) -> Dict[str, object]:
    periods = max(int(years * 12), 1)
    sampled_returns = block_bootstrap_monthly_returns(
        monthly_returns=monthly_returns,
        periods=periods,
        num_scenarios=num_scenarios,
        block_size=block_size,
        seed=seed,
    )
    return simulate_retirement_paths_from_returns(
        sampled_returns=sampled_returns,
        initial_portfolio=initial_portfolio,
        monthly_withdrawal=monthly_withdrawal,
        annual_inflation=annual_inflation,
        monthly_income=monthly_income,
    )


def monte_carlo_monthly_returns(
    annual_cagr: float,
    annual_volatility: float,
    periods: int,
    num_scenarios: int,
    seed: int = 42,
) -> np.ndarray:
    monthly_mean = (1 + annual_cagr) ** (1 / 12) - 1 if annual_cagr > -1 else -1.0
    monthly_std = max(float(annual_volatility), 0.0) / np.sqrt(12)
    rng = np.random.default_rng(seed)
    return rng.normal(loc=monthly_mean, scale=monthly_std, size=(periods, num_scenarios))


def simulate_retirement_paths_monte_carlo(
    annual_cagr: float,
    annual_volatility: float,
    initial_portfolio: float,
    monthly_withdrawal: float,
    years: int,
    annual_inflation: float = 0.03,
    monthly_income: float = 0.0,
    num_scenarios: int = 1000,
    seed: int = 42,
) -> Dict[str, object]:
    periods = max(int(years * 12), 1)
    sampled_returns = monte_carlo_monthly_returns(
        annual_cagr=annual_cagr,
        annual_volatility=annual_volatility,
        periods=periods,
        num_scenarios=num_scenarios,
        seed=seed,
    )
    return simulate_retirement_paths_from_returns(
        sampled_returns=sampled_returns,
        initial_portfolio=initial_portfolio,
        monthly_withdrawal=monthly_withdrawal,
        annual_inflation=annual_inflation,
        monthly_income=monthly_income,
    )


def find_sustainable_monthly_withdrawal(
    monthly_returns: pd.Series,
    initial_portfolio: float,
    years: int,
    annual_inflation: float = 0.03,
    monthly_income: float = 0.0,
    target_success_rate: float = 0.9,
    num_scenarios: int = 1000,
    block_size: int = 12,
    seed: int = 42,
    iterations: int = 24,
) -> Dict[str, object]:
    periods = max(int(years * 12), 1)
    lower = 0.0
    upper = max(initial_portfolio / periods * 2.0, initial_portfolio * 0.02)

    best_result: Optional[Dict[str, object]] = None
    for _ in range(12):
        trial = simulate_retirement_paths(
            monthly_returns=monthly_returns,
            initial_portfolio=initial_portfolio,
            monthly_withdrawal=upper,
            years=years,
            annual_inflation=annual_inflation,
            monthly_income=monthly_income,
            num_scenarios=num_scenarios,
            block_size=block_size,
            seed=seed,
        )
        if trial["survival_rate"] < target_success_rate:
            break
        lower = upper
        best_result = trial
        upper *= 1.5

    for _ in range(iterations):
        mid = (lower + upper) / 2.0
        trial = simulate_retirement_paths(
            monthly_returns=monthly_returns,
            initial_portfolio=initial_portfolio,
            monthly_withdrawal=mid,
            years=years,
            annual_inflation=annual_inflation,
            monthly_income=monthly_income,
            num_scenarios=num_scenarios,
            block_size=block_size,
            seed=seed,
        )
        if trial["survival_rate"] >= target_success_rate:
            lower = mid
            best_result = trial
        else:
            upper = mid

    if best_result is None:
        best_result = simulate_retirement_paths(
            monthly_returns=monthly_returns,
            initial_portfolio=initial_portfolio,
            monthly_withdrawal=0.0,
            years=years,
            annual_inflation=annual_inflation,
            monthly_income=monthly_income,
            num_scenarios=num_scenarios,
            block_size=block_size,
            seed=seed,
        )

    return {
        "monthly_withdrawal": lower,
        "result": best_result,
    }


def find_sustainable_monthly_withdrawal_monte_carlo(
    annual_cagr: float,
    annual_volatility: float,
    initial_portfolio: float,
    years: int,
    annual_inflation: float = 0.03,
    monthly_income: float = 0.0,
    target_success_rate: float = 0.9,
    num_scenarios: int = 1000,
    seed: int = 42,
    iterations: int = 24,
) -> Dict[str, object]:
    periods = max(int(years * 12), 1)
    lower = 0.0
    upper = max(initial_portfolio / periods * 2.0, initial_portfolio * 0.02)

    best_result: Optional[Dict[str, object]] = None
    for _ in range(12):
        trial = simulate_retirement_paths_monte_carlo(
            annual_cagr=annual_cagr,
            annual_volatility=annual_volatility,
            initial_portfolio=initial_portfolio,
            monthly_withdrawal=upper,
            years=years,
            annual_inflation=annual_inflation,
            monthly_income=monthly_income,
            num_scenarios=num_scenarios,
            seed=seed,
        )
        if trial["survival_rate"] < target_success_rate:
            break
        lower = upper
        best_result = trial
        upper *= 1.5

    for _ in range(iterations):
        mid = (lower + upper) / 2.0
        trial = simulate_retirement_paths_monte_carlo(
            annual_cagr=annual_cagr,
            annual_volatility=annual_volatility,
            initial_portfolio=initial_portfolio,
            monthly_withdrawal=mid,
            years=years,
            annual_inflation=annual_inflation,
            monthly_income=monthly_income,
            num_scenarios=num_scenarios,
            seed=seed,
        )
        if trial["survival_rate"] >= target_success_rate:
            lower = mid
            best_result = trial
        else:
            upper = mid

    if best_result is None:
        best_result = simulate_retirement_paths_monte_carlo(
            annual_cagr=annual_cagr,
            annual_volatility=annual_volatility,
            initial_portfolio=initial_portfolio,
            monthly_withdrawal=0.0,
            years=years,
            annual_inflation=annual_inflation,
            monthly_income=monthly_income,
            num_scenarios=num_scenarios,
            seed=seed,
        )

    return {
        "monthly_withdrawal": lower,
        "result": best_result,
    }


def build_static_portfolio_curve(
    prices: pd.DataFrame,
    invested_weights: pd.Series,
    initial_value: float = 10_000.0,
) -> pd.DataFrame:
    clean_prices = prices.loc[:, invested_weights.index].ffill().dropna()
    normalized = clean_prices / clean_prices.iloc[0]
    cash_weight = float(max(1.0 - invested_weights.sum(), 0.0))
    values = (normalized.mul(invested_weights, axis=1).sum(axis=1) + cash_weight) * initial_value
    return pd.DataFrame({"PortValue": values})


def run_one_shot_optimization(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    benchmark: pd.Series,
    vol_proxy: pd.Series,
    alpha_cfg: AlphaConfig,
    construction_cfg: ConstructionConfig,
    risk_cfg: RiskConfig,
    fundamentals: Optional[pd.DataFrame] = None,
) -> Dict[str, object]:
    snap_prices, snap_volumes, liquidity_snapshot = apply_point_in_time_universe_filters(
        prices_window=prices,
        volumes_window=volumes,
        full_prices=prices,
        as_of_date=prices.index.max(),
        alpha_cfg=alpha_cfg,
    )
    eligible_prices = filter_historical_universe(snap_prices, prices, prices.index.max(), alpha_cfg)
    eligible_volumes = snap_volumes.reindex(eligible_prices.index).reindex(columns=eligible_prices.columns)
    alpha_table = compute_alpha_table(eligible_prices, eligible_volumes, alpha_cfg, fundamentals=fundamentals)
    if alpha_table.empty:
        raise ValueError("No eligible assets after applying universe and data quality filters.")

    if not liquidity_snapshot.empty:
        alpha_table = alpha_table.join(
            liquidity_snapshot.set_index("ticker")[["liquidity_rank", "avg_dollar_volume_m"]],
            how="left",
            rsuffix="_snapshot",
        )

    candidate_table = alpha_table.head(alpha_cfg.top_n).copy()
    selected = select_diversified_holdings(
        candidate_table=candidate_table,
        candidate_prices=eligible_prices.reindex(columns=candidate_table.index),
        target_holdings=alpha_cfg.target_holdings,
        fundamentals=fundamentals,
    )
    candidate_table["selected_for_portfolio"] = candidate_table.index.isin(selected)
    alpha_table["in_candidate_set"] = alpha_table.index.isin(candidate_table.index)
    alpha_table["selected_for_portfolio"] = alpha_table.index.isin(selected)
    selected_prices = prices[selected].ffill().dropna()
    optimization = optimize_weights(
        selected_prices,
        construction_cfg=construction_cfg,
        alpha_scores=alpha_table.loc[selected, "composite_score"],
    )
    benchmark_window = benchmark.reindex(selected_prices.index).dropna()
    vol_window = vol_proxy.reindex(selected_prices.index).dropna() if not vol_proxy.empty else None
    regime_info = detect_regime(benchmark_window, prices.reindex(selected_prices.index), vol_window)

    exposure = 1.0
    if risk_cfg.use_regime_filter:
        if regime_info["regime"] == "Bull":
            exposure = risk_cfg.bull_exposure
        elif regime_info["regime"] == "Neutral":
            exposure = risk_cfg.neutral_exposure
        else:
            exposure = risk_cfg.bear_exposure

    if risk_cfg.use_trend_filter and regime_info["trend_score"] == 0.0:
        exposure = min(exposure, risk_cfg.neutral_exposure)

    exposure = float(np.clip(exposure, 0.0, 1.0))
    invested_weights = optimization["weights"] * exposure
    portfolio_curve = build_static_portfolio_curve(selected_prices, invested_weights)
    metrics = calculate_performance_metrics(portfolio_curve, construction_cfg.risk_free_rate)

    final_weights = optimization["weights"].rename("Target Weight").to_frame()
    final_weights["Invested Weight"] = invested_weights
    cash_weight = float(max(1.0 - invested_weights.sum(), 0.0))
    if cash_weight > 0:
        final_weights.loc["CASH", "Target Weight"] = 0.0
        final_weights.loc["CASH", "Invested Weight"] = cash_weight

    factor_exposure = (
        alpha_table.loc[selected, ["momentum_score", "quality_score", "value_score", "growth_score", "low_vol_score"]]
        .mul(optimization["weights"], axis=0)
        .sum()
        .rename("Portfolio Exposure")
        .to_frame()
    )

    return {
        "alpha_table": alpha_table,
        "candidate_table": candidate_table,
        "selected_assets": selected,
        "optimization": optimization,
        "weights": final_weights,
        "portfolio_curve": portfolio_curve,
        "metrics": metrics,
        "regime": regime_info,
        "effective_exposure": exposure,
        "factor_exposure": factor_exposure,
        "liquidity_snapshot": liquidity_snapshot,
        "liquidity_universe_size": len(liquidity_snapshot) if not liquidity_snapshot.empty else len(eligible_prices.columns),
    }


def _month_end_rebalance_dates(prices: pd.DataFrame, lookback_months: int, rebalance_months: int) -> List[pd.Timestamp]:
    month_ends = prices.groupby(prices.index.to_period("M")).tail(1).index.sort_values()
    if len(month_ends) <= lookback_months:
        return []
    return list(month_ends[lookback_months::rebalance_months])


def run_forward_test(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    benchmark: pd.Series,
    vol_proxy: pd.Series,
    alpha_cfg: AlphaConfig,
    construction_cfg: ConstructionConfig,
    risk_cfg: RiskConfig,
    implementation_cfg: ImplementationConfig,
    governance_cfg: GovernanceConfig,
    fundamentals: Optional[pd.DataFrame] = None,
    progress_callback: Optional[Callable[[int, int, pd.Timestamp], None]] = None,
) -> Dict[str, object]:
    rebalance_dates = _month_end_rebalance_dates(
        prices=prices,
        lookback_months=implementation_cfg.lookback_months,
        rebalance_months=implementation_cfg.rebalance_months,
    )
    if len(rebalance_dates) < 1:
        raise ValueError("Not enough data to run the requested rolling forward test.")

    all_assets = list(prices.columns)
    previous_weights = pd.Series(0.0, index=all_assets)
    equity = 10_000.0
    curve_rows: List[Dict[str, object]] = []
    weight_rows: List[Dict[str, object]] = []
    rebalance_rows: List[Dict[str, object]] = []

    total_rebalances = len(rebalance_dates)
    for idx, rebalance_date in enumerate(rebalance_dates):
        if progress_callback is not None:
            progress_callback(idx + 1, total_rebalances, rebalance_date)
        next_date = rebalance_dates[idx + 1] if idx + 1 < len(rebalance_dates) else prices.index[-1]
        train_start = rebalance_date - pd.DateOffset(months=implementation_cfg.lookback_months)

        train_prices = prices.loc[(prices.index > train_start) & (prices.index <= rebalance_date)]
        train_volumes = volumes.reindex(train_prices.index)
        if train_prices.empty:
            continue

        snap_train_prices, snap_train_volumes, liquidity_snapshot = apply_point_in_time_universe_filters(
            prices_window=train_prices,
            volumes_window=train_volumes,
            full_prices=prices,
            as_of_date=rebalance_date,
            alpha_cfg=alpha_cfg,
        )
        eligible_train_prices = filter_historical_universe(snap_train_prices, prices, rebalance_date, alpha_cfg)
        eligible_train_volumes = snap_train_volumes.reindex(columns=eligible_train_prices.columns)
        alpha_table = compute_alpha_table(eligible_train_prices, eligible_train_volumes, alpha_cfg, fundamentals=fundamentals)
        if alpha_table.empty:
            continue

        candidate_table = alpha_table.head(alpha_cfg.top_n).copy()
        selected = select_diversified_holdings(
            candidate_table=candidate_table,
            candidate_prices=eligible_train_prices.reindex(columns=candidate_table.index),
            target_holdings=alpha_cfg.target_holdings,
            fundamentals=fundamentals,
        )
        selected_train = train_prices[selected].ffill().dropna()
        if selected_train.empty or len(selected_train) < 20:
            continue

        optimization = optimize_weights(
            selected_train,
            construction_cfg=construction_cfg,
            alpha_scores=alpha_table.loc[selected, "composite_score"],
        )
        benchmark_window = benchmark.reindex(train_prices.index).dropna()
        vol_window = vol_proxy.reindex(train_prices.index).dropna() if not vol_proxy.empty else None
        regime_info = detect_regime(
            benchmark_window,
            train_prices,
            vol_window,
        )

        exposure = 1.0
        if risk_cfg.use_regime_filter:
            if regime_info["regime"] == "Bull":
                exposure = risk_cfg.bull_exposure
            elif regime_info["regime"] == "Neutral":
                exposure = risk_cfg.neutral_exposure
            else:
                exposure = risk_cfg.bear_exposure
        if risk_cfg.use_trend_filter and regime_info["trend_score"] == 0.0:
            exposure = min(exposure, risk_cfg.neutral_exposure)

        expected_vol = float(optimization["expected_volatility"])
        if expected_vol > EPSILON and construction_cfg.target_volatility > 0:
            exposure = min(exposure, construction_cfg.target_volatility / expected_vol)
        exposure = float(np.clip(exposure, 0.0, 1.0))

        target_weights = pd.Series(0.0, index=all_assets)
        target_weights.loc[selected] = optimization["weights"] * exposure
        turnover = float((target_weights - previous_weights).abs().sum())
        trading_cost_pct = turnover * (implementation_cfg.transaction_cost_bps + implementation_cfg.slippage_bps) / 10_000.0
        equity *= max(1.0 - trading_cost_pct, 0.0)

        period_prices = prices.loc[(prices.index > rebalance_date) & (prices.index <= next_date), selected]
        period_returns = period_prices.pct_change(fill_method=None).fillna(0.0)
        if period_returns.empty:
            previous_weights = target_weights
            continue

        peak = equity
        stopped = False
        for date, row in period_returns.iterrows():
            if not stopped:
                portfolio_return = float(row.fillna(0.0) @ (optimization["weights"] * exposure))
                equity *= 1.0 + portfolio_return
                peak = max(peak, equity)
                drawdown = equity / peak - 1.0
                if risk_cfg.max_drawdown_stop > 0 and drawdown <= -risk_cfg.max_drawdown_stop:
                    stopped = True
            curve_rows.append({"Date": date, "PortValue": equity})

        cash_weight = float(1.0 - target_weights.sum())
        for asset, weight in target_weights[target_weights > 0].sort_values(ascending=False).items():
            weight_rows.append({"Date": rebalance_date, "Asset": asset, "Weight": float(weight)})
        if cash_weight > 0:
            weight_rows.append({"Date": rebalance_date, "Asset": "CASH", "Weight": cash_weight})

        selected_adv = (train_prices[selected] * train_volumes[selected]).mean() / 1_000_000.0
        capacity = estimate_capacity_limit(
            weights=optimization["weights"],
            avg_dollar_volume_m=selected_adv,
            governance_cfg=governance_cfg,
        )

        rebalance_rows.append(
            {
                "Date": rebalance_date,
                "Regime": regime_info["regime"],
                "Regime Score": regime_info["score"],
                "Liquidity Universe Size": len(liquidity_snapshot) if not liquidity_snapshot.empty else len(eligible_train_prices.columns),
                "Alpha Candidates": len(candidate_table),
                "Target Holdings": len(selected),
                "Exposure": exposure,
                "Turnover": turnover,
                "Trading Cost %": trading_cost_pct * 100.0,
                "Selected Assets": ", ".join(selected),
                "Snapshot Source": liquidity_snapshot["snapshot_source"].iloc[0] if not liquidity_snapshot.empty else "computed",
                "Capacity Limit USD": capacity,
                "Expected Return": optimization["expected_return"],
                "Expected Volatility": optimization["expected_volatility"],
                "Expected Sharpe": optimization["expected_sharpe"],
            }
        )
        previous_weights = target_weights

    if curve_rows:
        curve = pd.DataFrame(curve_rows).drop_duplicates(subset="Date").set_index("Date").sort_index()
    else:
        curve = pd.DataFrame(columns=["PortValue"])
    weight_history = pd.DataFrame(weight_rows) if weight_rows else pd.DataFrame(columns=["Date", "Asset", "Weight"])
    rebalance_report = pd.DataFrame(rebalance_rows) if rebalance_rows else pd.DataFrame()
    metrics = calculate_performance_metrics(curve, construction_cfg.risk_free_rate)
    if curve.empty:
        benchmark_curve = pd.DataFrame(columns=["PortValue"])
    else:
        benchmark_slice = benchmark.reindex(curve.index).ffill().dropna()
        if benchmark_slice.empty:
            benchmark_curve = pd.DataFrame(columns=["PortValue"])
        else:
            benchmark_values = benchmark_slice / benchmark_slice.iloc[0] * 10_000.0
            benchmark_curve = pd.DataFrame({"PortValue": benchmark_values}, index=benchmark_slice.index)
    stress = run_stress_tests(curve)
    governance = governance_summary(rebalance_report, weight_history, governance_cfg)

    return {
        "curve": curve,
        "benchmark_curve": benchmark_curve,
        "metrics": metrics,
        "weight_history": weight_history,
        "rebalance_report": rebalance_report,
        "stress_tests": stress,
        "governance": governance,
    }


def estimate_capacity_limit(
    weights: pd.Series,
    avg_dollar_volume_m: pd.Series,
    governance_cfg: GovernanceConfig,
) -> float:
    usable_adv = avg_dollar_volume_m.reindex(weights.index).fillna(0.0) * 1_000_000.0
    position_caps: List[float] = []
    for asset, weight in weights.items():
        if weight <= 0:
            continue
        tradable = usable_adv.get(asset, 0.0) * governance_cfg.adv_participation_limit * governance_cfg.liquidation_days
        position_caps.append(tradable / weight if weight > 0 else np.inf)
    if not position_caps:
        return 0.0
    return float(min(position_caps))


def governance_summary(
    rebalance_report: pd.DataFrame,
    weight_history: pd.DataFrame,
    governance_cfg: GovernanceConfig,
) -> pd.DataFrame:
    notes: List[Dict[str, object]] = []
    if rebalance_report.empty:
        return pd.DataFrame()

    avg_turnover = rebalance_report["Turnover"].mean()
    avg_capacity = rebalance_report["Capacity Limit USD"].replace([np.inf, -np.inf], np.nan).dropna().mean()
    notes.append(
        {
            "Check": "Capacity proxy",
            "Value": avg_capacity,
            "Status": "OK" if avg_capacity >= governance_cfg.assumed_aum_usd else "Watch",
        }
    )
    notes.append(
        {
            "Check": "Average turnover",
            "Value": avg_turnover,
            "Status": "OK" if avg_turnover <= 0.80 else "Watch",
        }
    )

    if not weight_history.empty:
        max_weight = weight_history.groupby("Date")["Weight"].max().max()
        notes.append(
            {
                "Check": "Single-name concentration",
                "Value": max_weight,
                "Status": "OK" if max_weight <= 0.30 else "Watch",
            }
        )

    return pd.DataFrame(notes)


def run_stress_tests(curve: pd.DataFrame) -> pd.DataFrame:
    if curve.empty:
        return pd.DataFrame()

    windows = {
        "GFC 2008": ("2007-10-01", "2009-03-31"),
        "COVID Crash": ("2020-02-01", "2020-05-31"),
        "Rate Shock 2022": ("2022-01-01", "2022-12-31"),
    }
    rows: List[Dict[str, object]] = []
    for name, (start, end) in windows.items():
        sample = curve.loc[(curve.index >= start) & (curve.index <= end)]
        if len(sample) < 2:
            continue
        values = sample["PortValue"]
        returns = values.pct_change().dropna()
        drawdown = values / values.cummax() - 1.0
        rows.append(
            {
                "Scenario": name,
                "Return": values.iloc[-1] / values.iloc[0] - 1.0,
                "Max Drawdown": float(drawdown.min()),
                "Annualized Volatility": float(returns.std(ddof=0) * np.sqrt(TRADING_DAYS)),
            }
        )
    return pd.DataFrame(rows)
