# Best US/TH Side Trigger Config

Selected from fee/slippage-adjusted results only.

- Strategy: `Side trigger realloc to active stock side, fee+slippage`
- Objective: `min_vol_mom_tilt`
- Fee + slippage: `17.0` bps
- US trigger: `SPY + ^VIX`
- Thailand trigger: `^SET.BK`
- Reallocate stock sleeve: `True`
- US assets: `30`
- Thailand assets: `30`
- Max stock weight: `6.00%` inside equity sleeve
- Strategic weights: `Equity 60% / Gold 30% / BTC 10%`

## Metrics

- CAGR: `16.2970%`
- Sharpe: `0.9412`
- Sortino: `1.1797`
- Max Drawdown: `-20.3341%`
- Hit Rate: `0.5546`
- Start: `2017-12-29`
- End: `2026-04-29`

## Files

- `result/us_th_best_config_side_trigger_fee_slippage.json`
- `result/us_th_best_config_side_trigger_fee_slippage.csv`
- `result/us_th_side_trigger_latest_asset_weights_thb.csv`
- `result/us_th_side_trigger_reallocation_summary_thb.csv`