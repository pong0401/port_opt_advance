from __future__ import annotations

import refresh_us_th_tactical_one_model_us70_th30_latest as refresh


refresh.STRATEGY = "One-model US cap 70% / TH cap 30% stockcap5 penalty0.02 assets50 + daily exposure"
refresh.CASE = "US cap 70% / TH cap 30% stockcap5 penalty0.02 assets50"
refresh.RESULT_PREFIX = "us_th_one_model_us70_th30_stockcap5_penalty002_assets50"
refresh.STOCK_CAP = 0.05
refresh.US_ASSETS = 50
refresh.TH_ASSETS = 50
refresh.CONCENTRATION_PENALTY = 0.02


if __name__ == "__main__":
    refresh.main()
