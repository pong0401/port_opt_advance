"""Turn current broker holdings into a buy/sell trade plan against a strategy's
latest target weights, respecting Thai/US lot rules.

Pure logic, no Streamlit, so the rebalance math stays testable:

  - ``map_symbol_to_asset`` align a broker symbol with a strategy asset id.
  - ``compute_trade_plan``  diff holdings vs target weights -> per-asset orders.

Holdings come from a holdings.csv (produced offline by the ``asset_extract_ocr``
tool) or manual entry; screenshot OCR is intentionally kept out of the web app
so it stays light enough for free hosting.

Lot rules (the broker constraints we must honor):
  * Thai stocks (``*.BK``): traded in lots of 100 shares, buy and sell alike.
  * US stocks: buys may be fractional; sells must be whole shares (lot of 1).
  * Gold / BTC / cash: not lot-constrained equities, shown value-only (THB).
"""
from __future__ import annotations

import math
from typing import Mapping, Optional

import pandas as pd

# Broker lot sizes.
TH_LOT = 100

# Groups we keep for the rebalance (funds etc. are out of the strategy universe).
STOCK_GROUPS = {"us_stocks", "thai_stocks"}


def map_symbol_to_asset(symbol: str, group: str) -> str:
    """Align a broker symbol with the strategy's asset id.

    Thai holdings carry no exchange suffix on screen, so append ``.BK`` to match
    strategy weights (e.g. ``AWC`` -> ``AWC.BK``). US symbols are kept upper-cased.
    """
    sym = str(symbol).strip().upper()
    if group == "thai_stocks" and not sym.endswith(".BK"):
        return f"{sym}.BK"
    return sym


def is_thai_asset(asset: str) -> bool:
    return str(asset).upper().endswith(".BK")


def asset_group_label(asset: str) -> str:
    a = str(asset).upper()
    if a in {"CASH", "CASH / REDUCED EXPOSURE"}:
        return "Cash"
    if a in {"GOLD", "GC=F"}:
        return "Gold"
    if a in {"BTC", "BTC-USD"}:
        return "BTC"
    if a.endswith(".BK"):
        return "TH Equity"
    return "US Equity"


def _round_lot_shares(delta_shares: float, thai: bool) -> tuple[float, str]:
    """Apply the broker lot rule to a desired share delta.

    Returns (lot_adjusted_shares, rule_note). Positive = buy, negative = sell.
    """
    if thai:
        adj = round(delta_shares / TH_LOT) * TH_LOT
        return float(adj), "Thai lot 100"
    if delta_shares >= 0:
        # US buy: fractional shares allowed.
        return round(delta_shares, 4), "US buy fractional"
    # US sell: whole shares only (round magnitude down toward zero).
    return float(-math.floor(abs(delta_shares))), "US sell whole shares"


def compute_trade_plan(
    holdings_df: pd.DataFrame,
    target_df: pd.DataFrame,
    prices_thb: Mapping[str, float],
    extra_cash: float = 0.0,
    value_tol_pct: float = 0.5,
) -> pd.DataFrame:
    """Diff current holdings against target weights into per-asset orders.

    Args:
        holdings_df: current holdings with ``asset`` and ``value_thb`` columns.
        target_df: strategy latest weights with ``Asset`` and ``Portfolio %``.
        prices_thb: asset id -> latest price in THB (Thai shares already THB; US
            shares should be pre-converted to THB by the caller). Assets absent
            here (gold/BTC/cash) are treated as value-only, not lot-constrained.
        extra_cash: additional THB to deploy on top of current holdings value.
        value_tol_pct: |delta| below this % of total is treated as HOLD.

    Returns a DataFrame, one row per asset, sorted BUY/SELL first then by |Î” THB|.
    """
    current_thb: dict[str, float] = {}
    current_shares: dict[str, float] = {}
    prices_thb = {str(k).upper(): float(v) for k, v in prices_thb.items() if v}
    for _, row in holdings_df.iterrows():
        asset = str(row["asset"]).upper()
        share_val = row.get("shares", row.get("current_shares", None))
        shares = None
        if share_val is not None and not pd.isna(share_val):
            shares = float(share_val)
            if shares > 0:
                current_shares[asset] = current_shares.get(asset, 0.0) + shares

        val = row.get("value_thb")
        if shares is not None and shares > 0 and asset in prices_thb:
            val = shares * prices_thb[asset]
        if val is None or pd.isna(val):
            continue
        current_thb[asset] = current_thb.get(asset, 0.0) + float(val)

    target_pct: dict[str, float] = {}
    for _, row in target_df.iterrows():
        asset = str(row["Asset"]).strip()
        pct = row.get("Portfolio %")
        if asset and pct is not None and not pd.isna(pct):
            target_pct[asset.upper()] = target_pct.get(asset.upper(), 0.0) + float(pct)
    current_thb = {k.upper(): v for k, v in current_thb.items()}

    total_invested = sum(current_thb.values()) + float(extra_cash)
    tol_thb = total_invested * value_tol_pct / 100.0

    assets = sorted(set(current_thb) | set(target_pct))
    plan: list[dict] = []
    for asset in assets:
        cur = current_thb.get(asset, 0.0)
        tgt = total_invested * target_pct.get(asset, 0.0) / 100.0
        delta = tgt - cur
        price = prices_thb.get(asset)
        thai = is_thai_asset(asset)

        if delta > tol_thb:
            action = "BUY"
        elif delta < -tol_thb:
            action = "SELL"
        else:
            action = "HOLD"

        cur_shares = current_shares.get(asset)
        if cur_shares is None:
            cur_shares = (cur / price) if price else None
        delta_shares: Optional[float] = None
        executed_thb = delta
        if price and action != "HOLD":
            raw_shares = delta / price
            delta_shares, rule = _round_lot_shares(raw_shares, thai)
            if delta_shares < 0 and cur_shares is not None:
                if thai:
                    max_sell = math.floor(cur_shares / TH_LOT) * TH_LOT
                else:
                    max_sell = math.floor(cur_shares)
                if abs(delta_shares) > max_sell:
                    delta_shares = float(-max_sell)
                    rule = f"{rule}; capped by current shares"
            executed_thb = delta_shares * price
            if abs(raw_shares - delta_shares) > 1e-9:
                note = f"{rule} (req {raw_shares:,.2f} sh -> {delta_shares:,.0f} sh)"
            else:
                note = rule
        elif action == "HOLD":
            note = "within tolerance"
        else:
            note = "value-only (not lot-constrained)"

        plan.append({
            "Asset": asset,
            "Group": asset_group_label(asset),
            "Action": action,
            "Price THB": round(price, 4) if price else None,
            "Current Shares": round(cur_shares, 4) if cur_shares is not None else None,
            "Current THB": round(cur, 2),
            "Target %": round(target_pct.get(asset, 0.0), 4),
            "Target THB": round(tgt, 2),
            "Delta THB": round(delta, 2),
            "Delta Shares": delta_shares,
            "Executed THB": round(executed_thb, 2),
            "Note": note,
        })

    plan_df = pd.DataFrame(plan)
    if plan_df.empty:
        return plan_df
    action_order = {"BUY": 0, "SELL": 1, "HOLD": 2}
    plan_df["_o"] = plan_df["Action"].map(action_order)
    plan_df["_m"] = plan_df["Delta THB"].abs()
    plan_df = plan_df.sort_values(["_o", "_m"], ascending=[True, False]).drop(columns=["_o", "_m"])
    return plan_df.reset_index(drop=True)


