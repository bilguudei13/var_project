# =============================================================================
# compute_pnl.py
# Computes IRS P&L, straddle P&L and total portfolio P&L
#
# Inputs:  data/raw/prices.csv, vix.csv, dgs10.csv
#          data/processed/log_returns.csv
# Outputs: data/processed/instrument_pnl.csv
#          data/processed/total_portfolio_pnl.csv
# =============================================================================

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DATA_DIR = REPO_ROOT / "src" / "data"
if str(SRC_DATA_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DATA_DIR))

from config import (WEIGHTS_DICT, V0, IRS_NOTIONAL, IRS_FIXED_RATE,
                    STRADDLE_DAYS, STRADDLE_SHARES,
                    RAW_DIR, PROCESSED_DIR)
from portfolio_pricing import build_straddle_state, price_irs, price_straddle_position

RAW_PATH = REPO_ROOT / RAW_DIR
PROCESSED_PATH = REPO_ROOT / PROCESSED_DIR

def compute_instrument_pnl(prices, vix, dgs10):
    """
    IRS:      DV_irs_t      = V_irs(r_t) - V_irs(r_{t-1})
    Straddle: DV_straddle_t = BS(S_t,K,T_t) - BS(S_{t-1},K,T_{t-1}+1/252)
    Theory: report/theoretical_background.md — Section 1 (Irle Eq. 2, p. 82)
    """
    spy    = prices["SPY"]
    common = spy.index.intersection(vix.index).intersection(dgs10.index)
    spy    = spy.loc[common]
    vix_s  = vix.loc[common].squeeze()
    dgs_s  = dgs10.loc[common].squeeze()

    pnl_irs, pnl_straddle, dates = [], [], []
    straddle_state = build_straddle_state(spy, straddle_days=STRADDLE_DAYS)

    print("Computing instrument P&L...")
    for t in range(1, len(common)):
        v_today, _ = price_irs(IRS_NOTIONAL, IRS_FIXED_RATE, dgs_s.iloc[t]   / 100)
        v_prev,  _ = price_irs(IRS_NOTIONAL, IRS_FIXED_RATE, dgs_s.iloc[t-1] / 100)
        pnl_irs_t  = v_today - v_prev

        prev_state = straddle_state.iloc[t - 1]
        today_state = straddle_state.iloc[t]
        rf_today = max(float(dgs_s.iloc[t]) / 100.0, 0.0)
        rf_prev = max(float(dgs_s.iloc[t - 1]) / 100.0, 0.0)

        value_today = float(
            price_straddle_position(
                float(spy.iloc[t]),
                float(today_state["strike_spy"]),
                float(today_state["tenor_years"]),
                rf_today,
                float(vix_s.iloc[t]) / 100.0,
            )
        ) * STRADDLE_SHARES
        value_prev = float(
            price_straddle_position(
                float(spy.iloc[t - 1]),
                float(prev_state["strike_spy"]),
                float(prev_state["tenor_years"]),
                rf_prev,
                float(vix_s.iloc[t - 1]) / 100.0,
            )
        ) * STRADDLE_SHARES

        pnl_irs.append(pnl_irs_t)
        # Roll days include the economic jump from an expiring straddle into
        # the newly opened ATM straddle; this is not a pure market-move P&L.
        pnl_straddle.append(value_today - value_prev)
        dates.append(common[t])

    return pd.DataFrame({"pnl_irs": pnl_irs, "pnl_straddle": pnl_straddle},
                         index=dates)

def compute_total_pnl(prices, instrument_pnl, weights_dict, portfolio_value):
    """
    Total DV = DV_linear + DV_irs + DV_straddle
    Theory: report/theoretical_background.md — Section 1 (Irle Eq. 2)
    """
    weights_aligned = {
        ("EURUSD" if k == "EURUSD=X" else k): v
        for k, v in weights_dict.items()
    }
    price_frame = prices.rename(columns={"EURUSD=X": "EURUSD"}).sort_index()
    columns = list(price_frame.columns)
    w = pd.Series(weights_aligned)[columns].astype(float)

    initial_prices = price_frame.iloc[0]
    linear_shares = portfolio_value * w / initial_prices
    prices_prev = price_frame.shift(1)
    linear_pnl = ((price_frame - prices_prev) * linear_shares).sum(axis=1)
    linear_pnl = linear_pnl.dropna().rename("pnl_linear")

    common = linear_pnl.index.intersection(instrument_pnl.index)
    total  = pd.DataFrame({
        "pnl_linear"   : linear_pnl.loc[common],
        "pnl_irs"      : instrument_pnl.loc[common, "pnl_irs"],
        "pnl_straddle" : instrument_pnl.loc[common, "pnl_straddle"],
    })
    total["pnl_total"] = total.sum(axis=1)
    return total

if __name__ == "__main__":
    PROCESSED_PATH.mkdir(parents=True, exist_ok=True)

    prices      = pd.read_csv(RAW_PATH / "prices.csv",
                               index_col=0, parse_dates=True)
    vix         = pd.read_csv(RAW_PATH / "vix.csv",
                               index_col=0, parse_dates=True).squeeze()
    dgs10       = pd.read_csv(RAW_PATH / "dgs10.csv",
                               index_col=0, parse_dates=True).squeeze()
    instrument_pnl = compute_instrument_pnl(prices, vix, dgs10)
    total_pnl      = compute_total_pnl(prices, instrument_pnl,
                                        WEIGHTS_DICT, V0)

    instrument_pnl.to_csv(PROCESSED_PATH / "instrument_pnl.csv")
    total_pnl.to_csv(PROCESSED_PATH / "total_portfolio_pnl.csv")

    print("\nTOTAL P&L STATS:")
    print(total_pnl.describe().round(2))
    print("\nSaved -> instrument_pnl.csv, total_portfolio_pnl.csv")
