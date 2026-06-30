from __future__ import annotations

import importlib
from collections.abc import Callable

import pandas as pd

import refresh_us_th_tactical_one_model_us70_th30_latest as refresh


Panel = tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str], list[str], pd.DataFrame, pd.Timestamp, str]


def _copy_panel(panel: Panel) -> Panel:
    prices, volumes, benchmark, vol_proxy, us_all, th_all, overlay, as_of, fresh_start = panel
    return (
        prices.copy(),
        volumes.copy(),
        benchmark.copy(),
        vol_proxy.copy(),
        list(us_all),
        list(th_all),
        overlay.copy(),
        pd.Timestamp(as_of),
        str(fresh_start),
    )


def _set_base_config(
    *,
    strategy: str,
    case: str,
    result_prefix: str,
    stock_cap: float,
    us_assets: int,
    th_assets: int,
    concentration_penalty: float,
) -> None:
    refresh.STRATEGY = strategy
    refresh.CASE = case
    refresh.RESULT_PREFIX = result_prefix
    refresh.STOCK_CAP = stock_cap
    refresh.US_ASSETS = us_assets
    refresh.TH_ASSETS = th_assets
    refresh.CONCENTRATION_PENALTY = concentration_penalty


def _run_variant(
    label: str,
    panel: Panel,
    *,
    optimize_one_model: Callable | None,
    write_outputs: Callable,
) -> None:
    refresh._fresh_us_th_panel = lambda: _copy_panel(panel)
    refresh._optimize_one_model = optimize_one_model or _BASE_OPTIMIZE_ONE_MODEL
    refresh._write_outputs = write_outputs
    print(f"Refreshing {label}...")
    refresh.main()


_BASE_FRESH_US_TH_PANEL = refresh._fresh_us_th_panel
_BASE_OPTIMIZE_ONE_MODEL = refresh._optimize_one_model
_BASE_WRITE_OUTPUTS = refresh._write_outputs


def main(panel: Panel | None = None) -> None:
    if panel is None:
        print("Downloading one-model US/TH latest panel once...")
        panel = _BASE_FRESH_US_TH_PANEL()
    else:
        print("Reusing shared one-model US/TH latest panel...")

    _set_base_config(
        strategy="One-model US cap 70% / TH cap 30% + daily exposure",
        case="US cap 70% / TH cap 30%",
        result_prefix="us_th_tactical_one_model_us70_th30",
        stock_cap=0.08,
        us_assets=30,
        th_assets=30,
        concentration_penalty=0.02,
    )
    _run_variant(
        "one-model US70/TH30 base",
        panel,
        optimize_one_model=None,
        write_outputs=_BASE_WRITE_OUTPUTS,
    )

    _set_base_config(
        strategy="One-model US cap 70% / TH cap 30% stockcap5 penalty0.02 assets50 + daily exposure",
        case="US cap 70% / TH cap 30% stockcap5 penalty0.02 assets50",
        result_prefix="us_th_one_model_us70_th30_stockcap5_penalty002_assets50",
        stock_cap=0.05,
        us_assets=50,
        th_assets=50,
        concentration_penalty=0.02,
    )
    _run_variant(
        "one-model US70/TH30 stockcap5 assets50",
        panel,
        optimize_one_model=None,
        write_outputs=_BASE_WRITE_OUTPUTS,
    )

    theme = importlib.import_module("refresh_us_th_one_model_us70_th30_theme_cap25_latest")
    _run_variant(
        "one-model US70/TH30 stockcap5 assets50 AI-tech cap25",
        panel,
        optimize_one_model=theme._optimize_one_model_with_theme_cap,
        write_outputs=theme._write_outputs_with_theme_meta,
    )

    refresh._write_outputs = _BASE_WRITE_OUTPUTS
    segment = importlib.import_module("refresh_us_th_one_model_us70_th30_segment_cap30_latest")
    _run_variant(
        "one-model US70/TH30 stockcap5 assets50 US segment cap30",
        panel,
        optimize_one_model=segment._optimize_one_model_with_segment_cap,
        write_outputs=segment._write_outputs_with_segment_meta,
    )


if __name__ == "__main__":
    main()
