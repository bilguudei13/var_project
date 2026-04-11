# =============================================================================
# compute_returns.py
# Computes log returns and linear portfolio returns from raw prices
#
# Inputs:  data/raw/prices.csv
# Outputs: data/processed/log_returns.csv
#          data/processed/portfolio_returns.csv
# =============================================================================

import os
import sys
import numpy as np
import pandas as pd

sys.path.append("src/data")
from config import WEIGHTS_DICT, V0, RAW_DIR, PROCESSED_DIR

def compute_log_returns(prices):
    """
    r_t = log(P_t / P_{t-1})
    Theory: report/theoretical_background.md — Section 2 (Irle p. 35)
    """
    log_returns = np.log(prices / prices.shift(1)).dropna()
    print(f"Log returns: {len(log_returns)} obs x {len(log_returns.columns)} assets")
    return log_returns

def compute_portfolio_returns(log_returns, weights_dict):
    """
    R^{d,P} = sum_i w_i * R_i   (Irle Eq. 10)
    """
    weights_aligned = {
        ("EURUSD" if k == "EURUSD=X" else k): v
        for k, v in weights_dict.items()
    }
    w = pd.Series(weights_aligned)[log_returns.columns]
    portfolio_returns = log_returns.dot(w)
    portfolio_returns.name = "portfolio"
    print(f"Portfolio returns computed | Mean: {portfolio_returns.mean():.6f}")
    return portfolio_returns

if __name__ == "__main__":
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    prices = pd.read_csv(os.path.join(RAW_DIR, "prices.csv"),
                          index_col=0, parse_dates=True)

    log_returns       = compute_log_returns(prices)
    portfolio_returns = compute_portfolio_returns(log_returns, WEIGHTS_DICT)

    log_returns.to_csv(os.path.join(PROCESSED_DIR, "log_returns.csv"))
    portfolio_returns.to_csv(os.path.join(PROCESSED_DIR, "portfolio_returns.csv"))
    print("Saved -> log_returns.csv, portfolio_returns.csv")