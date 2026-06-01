# US/TH Best Config Test

This repo can now re-run the latest US/Thailand blended best-config test from
`../dynamic_port_opt`.

## Source Config

- Config record: `result/us_th_best_config_side_trigger_fee_slippage.json`
- Human notes: `doc/BEST_CONFIG_US_TH_SIDE_TRIGGER.md`
- Notebook source checked: `../dynamic_port_opt/notebook/us_th_blend_poc.ipynb`
- Original runner checked: `../dynamic_port_opt/scripts/run_us_th_joint_model.py`

## Current Best Config

- Objective: `min_vol_mom_tilt`
- Equity sleeve: `60%`
- Gold sleeve: `30%`
- BTC sleeve: `10%`
- US assets: `30`
- Thailand assets: `30`
- Max single stock weight inside equity sleeve: `6%`
- Lookback: `504` trading days
- Clusters: `4`
- Momentum signal: `mom_63`
- Feature flags dropped: `resid_vol`, `drawdown`, `downside_beta`
- Transaction model: `15` bps commission plus `2` bps slippage
- Side triggers: US uses `SPY + ^VIX`, Thailand uses `^SET.BK`
- Idle stock exposure is reallocated to the active stock side.

## Run

```powershell
python scripts\run_us_th_best_config.py
```

The default command runs the best config only. To re-run the broader historical
sweeps, use:

```powershell
python scripts\run_us_th_best_config.py --asset-sweep-only
python scripts\run_us_th_best_config.py --full-sweep
```

## Main Outputs

- `result/us_th_best_config_side_trigger_fee_slippage.json`
- `result/us_th_best_config_side_trigger_fee_slippage.csv`
- `result/us_th_side_trigger_reallocation_summary_thb.csv`
- `result/us_th_side_trigger_latest_asset_weights_thb.csv`
- `result/us_th_best_asset_sweep_fee_realloc_summary_thb.csv`
- `result/us_th_best_asset_sweep_fee_realloc_curves_thb.csv`
