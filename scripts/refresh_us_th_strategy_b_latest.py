from __future__ import annotations

import pandas as pd

import refresh_us_th_tactical_final_best_latest as final_best
import refresh_us_th_tactical_603010_latest as tactical_603010
import refresh_us_th_one_model_variants_latest as one_model_variants


Panel = one_model_variants.Panel


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


def main() -> None:
    print("Downloading shared Strategy B US/TH latest panel once...")
    panel = final_best._fresh_us_th_panel()

    original_final_panel_loader = final_best._fresh_us_th_panel
    try:
        final_best._fresh_us_th_panel = lambda: _copy_panel(panel)
        print("Refreshing Strategy B final best Sharpe latest weights...")
        final_best.main()
    finally:
        final_best._fresh_us_th_panel = original_final_panel_loader

    print("Refreshing Tactical TH/Gold/BTC 60/30/10 latest weights...")
    tactical_603010.main(panel=_copy_panel(panel))

    one_model_variants.main(panel=_copy_panel(panel))


if __name__ == "__main__":
    main()
