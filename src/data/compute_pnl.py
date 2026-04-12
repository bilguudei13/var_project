# =============================================================================
# compute_pnl.py
# Computes IRS P&L, straddle P&L and total portfolio P&L
#
# Inputs:  data/raw/prices.csv, vix.csv, dgs10.csv
#          data/processed/log_returns.csv
# Outputs: data/processed/instrument_pnl.csv
#          data/processed/total_portfolio_pnl.csv
# =============================================================================

import os
import sys
import numpy as np
import pandas as pd

sys.path.append("src/data")
from config import (WEIGHTS_DICT, V0, IRS_NOTIONAL, IRS_FIXED_RATE,
                    STRADDLE_DAYS, STRADDLE_SHARES,
                    RAW_DIR, PROCESSED_DIR)
from portfolio_pricing import price_irs, price_straddle

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
    K, days_held = spy.iloc[0], 0

    print("Computing instrument P&L...")
    for t in range(1, len(common)):
        v_today, _ = price_irs(IRS_NOTIONAL, IRS_FIXED_RATE, dgs_s.iloc[t]   / 100)
        v_prev,  _ = price_irs(IRS_NOTIONAL, IRS_FIXED_RATE, dgs_s.iloc[t-1] / 100)
        pnl_irs_t  = v_today - v_prev

        if days_held >= STRADDLE_DAYS:
            K, days_held = spy.iloc[t], 0

        T_today = max((STRADDLE_DAYS - days_held) / 252, 1/252)
        rf_today = max(float(dgs_s.iloc[t]) / 100.0, 0.0)
        rf_prev = max(float(dgs_s.iloc[t - 1]) / 100.0, 0.0)
        p_today, _ = price_straddle(spy.iloc[t],   K, T_today,
                                     rf_today, vix_s.iloc[t]   / 100)
        p_prev,  _ = price_straddle(spy.iloc[t-1], K, T_today + 1/252,
                                     rf_prev, vix_s.iloc[t-1] / 100)

        pnl_irs.append(pnl_irs_t)
        pnl_straddle.append((p_today - p_prev) * STRADDLE_SHARES)
        dates.append(common[t])
        days_held += 1

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

    prices_prev = price_frame.shift(1)
    linear_shares = portfolio_value * w / prices_prev
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
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    prices      = pd.read_csv(os.path.join(RAW_DIR, "prices.csv"),
                               index_col=0, parse_dates=True)
    vix         = pd.read_csv(os.path.join(RAW_DIR, "vix.csv"),
                               index_col=0, parse_dates=True).squeeze()
    dgs10       = pd.read_csv(os.path.join(RAW_DIR, "dgs10.csv"),
                               index_col=0, parse_dates=True).squeeze()
    instrument_pnl = compute_instrument_pnl(prices, vix, dgs10)
    total_pnl      = compute_total_pnl(prices, instrument_pnl,
                                        WEIGHTS_DICT, V0)

    instrument_pnl.to_csv(os.path.join(PROCESSED_DIR, "instrument_pnl.csv"))
    total_pnl.to_csv(os.path.join(PROCESSED_DIR,      "total_portfolio_pnl.csv"))

    print("\nTOTAL P&L STATS:")
    print(total_pnl.describe().round(2))
    print("\nSaved -> instrument_pnl.csv, total_portfolio_pnl.csv")
