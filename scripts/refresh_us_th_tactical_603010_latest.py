from __future__ import annotations

import pandas as pd

import refresh_us_th_tactical_final_best_latest as refresh


STRATEGY = "Tactical TH/Gold/BTC 60/30/10 asset-level daily exposure"
RESULT_PREFIX = "us_th_tactical_perf_momentum_603010"


def main(panel: tuple | None = None) -> None:
    original_values = {
        "SELECTED_MIX": refresh.SELECTED_MIX,
        "STRATEGY": refresh.STRATEGY,
        "RESULT_PREFIX": refresh.RESULT_PREFIX,
        "OVERLAY_MIX_LABEL": refresh.OVERLAY_MIX_LABEL,
        "GOLD_EXPOSURE_MODE": refresh.GOLD_EXPOSURE_MODE,
        "DAILY_EXPOSURE_DESCRIPTION": refresh.DAILY_EXPOSURE_DESCRIPTION,
        "_fresh_us_th_panel": refresh._fresh_us_th_panel,
    }
    try:
        refresh.SELECTED_MIX = {"Equity": 0.60, "Gold": 0.30, "BTC": 0.10}
        refresh.STRATEGY = STRATEGY
        refresh.RESULT_PREFIX = RESULT_PREFIX
        refresh.OVERLAY_MIX_LABEL = "Equity/Gold/BTC 60/30/10"
        refresh.GOLD_EXPOSURE_MODE = "ma50_below100"
        refresh.DAILY_EXPOSURE_DESCRIPTION = (
            "US SPY MA300 below50%; TH SET MA200 below0%; "
            "Gold MA50 below100%; BTC MA50 below0%"
        )
        if panel is not None:
            refresh._fresh_us_th_panel = lambda: (
                panel[0].copy(), panel[1].copy(), panel[2].copy(), panel[3].copy(),
                list(panel[4]), list(panel[5]), panel[6].copy(), pd.Timestamp(panel[7]), str(panel[8]),
            )
        refresh.main()
    finally:
        for name, value in original_values.items():
            setattr(refresh, name, value)


if __name__ == "__main__":
    main()
