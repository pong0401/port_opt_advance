from __future__ import annotations



import argparse

import gzip

import json

import math

import shutil

from datetime import datetime, timezone

from pathlib import Path

from typing import Any



import pandas as pd





PROJECT_ROOT = Path(__file__).resolve().parents[1]

PRECOMPUTED_DIR = PROJECT_ROOT / "data" / "precomputed"

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "public" / "data"

VISIBLE_WEIGHT_FLOOR = 0.01

WEIGHT_SUM_TOLERANCE = 0.01


EXCLUDED_STRATEGY_NAME_PARTS = ("no cost",)


def _is_excluded_strategy_name(name: object) -> bool:
    text = str(name).casefold()
    return any(part in text for part in EXCLUDED_STRATEGY_NAME_PARTS)





LATEST_WEIGHT_FILES = {

    "qqq_global_rotation_defensive_gtaa_403030": "qqq_global_rotation_defensive_gtaa_403030_latest_effective_weights.csv",

    "country_etf_tactical_gold_dd_boost16": "country_etf_tactical_gold_dd_boost16_latest_effective_weights.csv",

    "mean_covariance_gold30_asset_daily": "mean_covariance_gold30_asset_daily_latest_effective_weights.csv",

    "mean_covariance_gold40_asset_daily": "mean_covariance_gold40_asset_daily_latest_effective_weights.csv",

    "pit_reselect_step2_5_daily_exposure": "pit_reselect_step2_5_latest_effective_security_weights_thb.csv",

    "us_th_one_model_stockcap5_assets50": "us_th_one_model_us70_th30_stockcap5_penalty002_assets50_latest_effective_weights_thb.csv",

    "us_th_jp_optimized_minvol_top10_cap15": "us_th_jp_optimized_minvol_top10_cap15_weekly_latest_effective_weights_thb.csv",

    "us_th_best_asset_sweep_603010_asset_daily": "us_th_best_asset_sweep_latest_effective_weights_live_thb.csv",

}




DASHBOARD_STRATEGY_GROUPS = {
    "strategy_a": {
        "label": "Strategy A: Best Param S&P Port Opt Advance",
        "default": "QQQ + global rotation + defensive GTAA 40/30/30",
        "options": [
            {"label": "S&P 500 hold: 100% SPY, no daily exposure", "series": "S&P 500 buy and hold", "latest_weight_id": None},
            {"label": "Monthly multi-asset allocation: SPY 35%, Gold 40%, BTC 10%, BIL 15%", "series": "Monthly allocation SPY/Gold/BTC/BIL 35/40/10/15", "latest_weight_id": None},
            {"label": "Country ETF tactical + Gold DD boost 16", "series": "Country ETF tactical + Gold DD boost 16", "latest_weight_id": "country_etf_tactical_gold_dd_boost16"},

            {"label": "QQQ + global rotation + defensive GTAA 40/30/30", "series": "QQQ + global rotation + defensive GTAA 40/30/30", "latest_weight_id": "qqq_global_rotation_defensive_gtaa_403030"},
        ],
    },
    "strategy_b": {
        "label": "Strategy B: PIT Reselect Strategies",
        "default": "One-model US cap 70% / TH cap 30% stockcap5 penalty0.02 assets50 + daily exposure",
        "options": [
            {"label": "Tactical TH/Gold/BTC 60/30/10 asset-level daily exposure", "series": "Best asset sweep US30/TH30/max6 dynamic cash drag, fee+slippage", "latest_weight_id": "us_th_best_asset_sweep_603010_asset_daily"},
            {"label": "US/TH/JP index signal + JP optimized top10 cap15 + weekly exposure + Gold DD252", "series": "JP optimized min_vol_mom_tilt top10 cap15 weekly exposure with Gold DD252", "latest_weight_id": "us_th_jp_optimized_minvol_top10_cap15"},
            {"label": "One-model US cap 70% / TH cap 30% stockcap5 penalty0.02 assets50 + daily exposure", "series": "One-model US cap 70% / TH cap 30% stockcap5 penalty0.02 assets50 + daily exposure", "latest_weight_id": "us_th_one_model_stockcap5_assets50"},
        ],
    },
}

def _read_json(path: Path) -> dict[str, Any]:

    if not path.exists():

        return {}

    return json.loads(path.read_text(encoding="utf-8"))





def _clean_records(df: pd.DataFrame) -> list[dict[str, Any]]:

    clean = df.copy().astype(object)

    clean = clean.where(pd.notna(clean), None)

    records = clean.to_dict(orient="records")

    for row in records:

        for key, value in list(row.items()):

            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):

                row[key] = None

    return records





def _find_weight_column(df: pd.DataFrame) -> str:

    for column in ("Effective Weight", "Weight", "Target Weight", "Portfolio Exposure"):

        if column in df.columns:

            return column

    for column in df.columns:

        if column.lower().strip().endswith("weight"):

            return column

    raise ValueError(f"Could not find a weight column in columns: {list(df.columns)}")





def _find_date_column(df: pd.DataFrame) -> str | None:

    for column in ("Date", "Latest Cache Trading Date", "Last Exposure Date", "Signal Source Close Date"):

        if column in df.columns:

            return column

    return None





def _is_cash_row(row: pd.Series) -> bool:

    text = " ".join(str(row.get(column, "")) for column in ("Asset", "Ticker", "Sleeve"))

    return "Cash / Reduced Exposure" in text or text.strip().lower() in {"cash", "bil"}





def _export_latest_weight_file(strategy_id: str, source_name: str, latest_dir: Path) -> dict[str, Any]:

    source = PRECOMPUTED_DIR / source_name

    if not source.exists():

        raise FileNotFoundError(f"Missing latest-weight source: {source}")



    df = pd.read_csv(source)

    if df.empty:

        raise ValueError(f"Latest-weight source is empty: {source}")



    weight_column = _find_weight_column(df)

    df[weight_column] = pd.to_numeric(df[weight_column], errors="coerce").fillna(0.0)

    raw_weight_sum = float(df[weight_column].sum())

    if abs(raw_weight_sum - 1.0) > WEIGHT_SUM_TOLERANCE:

        raise ValueError(

            f"{source.name} {weight_column} sum is {raw_weight_sum:.6f}; expected approximately 1.0"

        )



    visible_df = df[df.apply(lambda row: _is_cash_row(row) or float(row[weight_column]) >= VISIBLE_WEIGHT_FLOOR, axis=1)]

    visible_weight_sum = float(visible_df[weight_column].sum())



    date_column = _find_date_column(df)

    market_date = None

    if date_column:

        dates = pd.to_datetime(df[date_column], errors="coerce").dropna()

        if not dates.empty:

            market_date = dates.max().date().isoformat()



    strategy_name = None

    if "Strategy" in df.columns:

        names = [str(value) for value in df["Strategy"].dropna().unique()]

        strategy_name = names[0] if names else None



    latest_dir.mkdir(parents=True, exist_ok=True)

    csv_target = latest_dir / f"{strategy_id}.csv"

    json_target = latest_dir / f"{strategy_id}.json"

    shutil.copy2(source, csv_target)

    json_target.write_text(

        json.dumps(

            {

                "schema_version": 1,

                "strategy_id": strategy_id,

                "strategy": strategy_name or strategy_id,

                "market_date": market_date,

                "source_file": f"data/precomputed/{source.name}",

                "weight_column": weight_column,

                "raw_weight_sum": raw_weight_sum,

                "visible_weight_sum": visible_weight_sum,

                "hidden_below_weight": VISIBLE_WEIGHT_FLOOR,

                "rows": _clean_records(visible_df),

            },

            indent=2,

            ensure_ascii=False,

            allow_nan=False,

        ),

        encoding="utf-8",

    )



    return {

        "strategy_id": strategy_id,

        "strategy": strategy_name or strategy_id,

        "market_date": market_date,

        "source_file": f"data/precomputed/{source.name}",

        "csv": f"latest_weights/{csv_target.name}",

        "json": f"latest_weights/{json_target.name}",

        "weight_column": weight_column,

        "raw_weight_sum": raw_weight_sum,

        "visible_rows": int(len(visible_df)),

        "raw_rows": int(len(df)),

    }





def _write_json(path: Path, payload: Any) -> None:

    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")





def _write_gzip_json(path: Path, payload: Any) -> None:

    with gzip.open(path, "wt", encoding="utf-8") as fh:

        json.dump(payload, fh, ensure_ascii=False, separators=(",", ":"), allow_nan=False)





def _dataframe_time_series_payload(df: pd.DataFrame) -> dict[str, Any]:

    if df.empty:

        raise ValueError("Time-series dataframe is empty")

    frame = df.copy()

    if not isinstance(frame.index, pd.DatetimeIndex):

        frame.index = pd.to_datetime(frame.index, errors="coerce")

    frame = frame[~frame.index.isna()].sort_index()

    if frame.empty:

        raise ValueError("Time-series dataframe has no valid dates")

    frame = frame.astype(object).where(pd.notna(frame), None)

    return {

        "columns": [str(column) for column in frame.columns],

        "dates": [idx.date().isoformat() for idx in frame.index],

        "data": frame.values.tolist(),

    }





def export_static_dashboard_data(output_dir: Path = DEFAULT_OUTPUT_DIR, run_id: str | None = None) -> dict[str, Any]:

    run_id = run_id or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")

    run_dir = output_dir / "runs" / run_id

    latest_dir = run_dir / "latest_weights"

    run_dir.mkdir(parents=True, exist_ok=True)



    metadata = _read_json(PRECOMPUTED_DIR / "streamlit_10y_metadata.json")

    summary = pd.read_csv(PRECOMPUTED_DIR / "streamlit_10y_strategy_summary.csv")

    curves = pd.read_parquet(PRECOMPUTED_DIR / "streamlit_10y_strategy_curves.parquet")

    returns = pd.read_parquet(PRECOMPUTED_DIR / "streamlit_10y_strategy_returns.parquet")

    summary = summary[~summary["Strategy"].apply(_is_excluded_strategy_name)].copy()
    curves = curves.loc[:, [column for column in curves.columns if not _is_excluded_strategy_name(column)]]
    returns = returns.loc[:, [column for column in returns.columns if not _is_excluded_strategy_name(column)]]



    if summary.empty:

        raise ValueError("Strategy summary is empty")

    if curves.empty or returns.empty:

        raise ValueError("Strategy curves/returns are empty")



    summary_path = run_dir / "strategy_summary.json"

    curves_path = run_dir / "strategy_curves.json"

    returns_path = run_dir / "strategy_returns.parquet"

    metadata_path = run_dir / "strategy_metadata.json"



    _write_json(summary_path, _clean_records(summary))

    _write_json(curves_path, _dataframe_time_series_payload(curves))

    returns.to_parquet(returns_path)

    _write_json(metadata_path, metadata)



    latest_weights = [

        _export_latest_weight_file(strategy_id, source_name, latest_dir)

        for strategy_id, source_name in LATEST_WEIGHT_FILES.items()

    ]

    market_dates = [item["market_date"] for item in latest_weights if item["market_date"]]

    market_date = max(market_dates) if market_dates else None



    manifest = {

        "schema_version": 1,

        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),

        "market_date": market_date,

        "run_id": run_id,

        "status": "ok",

        "data_base_url": f"runs/{run_id}/",

        "files": {

            "strategy_summary": summary_path.name,

            "strategy_curves": curves_path.name,

            "strategy_returns": returns_path.name,

            "strategy_metadata": metadata_path.name,

            "latest_weights": {item["strategy_id"]: item["json"] for item in latest_weights},

        },

        "latest_weights": latest_weights,

        "dashboard_strategy_groups": DASHBOARD_STRATEGY_GROUPS,

        "validation": {

            "strategy_summary_rows": int(len(summary)),

            "strategy_curve_columns": int(len(curves.columns)),

            "strategy_curve_start": curves.index.min().date().isoformat(),

            "strategy_curve_end": curves.index.max().date().isoformat(),

            "hidden_below_weight": VISIBLE_WEIGHT_FLOOR,

            "weight_sum_tolerance": WEIGHT_SUM_TOLERANCE,

        },

    }

    _write_json(run_dir / "manifest.json", manifest)

    _write_json(output_dir / "latest_manifest.json", manifest)

    return manifest





def main() -> None:

    parser = argparse.ArgumentParser(description="Export Streamlit precomputed artifacts for a static dashboard.")

    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    parser.add_argument("--run-id", default=None)

    args = parser.parse_args()

    manifest = export_static_dashboard_data(args.output_dir, args.run_id)

    print(json.dumps({"status": "ok", "run_id": manifest["run_id"], "output_dir": str(args.output_dir)}, indent=2))





if __name__ == "__main__":

    main()







