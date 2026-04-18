# =============================================================================
# compute_returns.py
# Computes log returns and linear portfolio returns from raw prices
#
# Inputs:  data/raw/prices.csv, vix.csv, dgs10.csv
# Outputs: data/processed/log_returns.csv
#          data/processed/portfolio_returns.csv
#          data/processed/risk_factors.csv
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

def compute_risk_factors(prices, vix, dgs10):
    """
    Compute the 6 main risk factors for the GARCH model:
    1. SPY log-return
    2. 10-year yield change (DGS10)
    3. Gold log-returns (GLD)
    4. EUR/USD log-return (EURUSD=X)
    5. S&P 500 level change (SPY)
    6. Implied volatility change (VIX)
    """
    # Ensure common dates
    common = prices.index.intersection(vix.index).intersection(dgs10.index)
    p = prices.loc[common]
    v = vix.loc[common].squeeze()
    d = dgs10.loc[common].squeeze()

    factors = pd.DataFrame(index=common)
    
    # 1. SPY log-return
    factors['SPY_log_return'] = np.log(p['SPY'] / p['SPY'].shift(1))
    
    # 2. 10-year yield change
    factors['DGS10_change'] = d - d.shift(1)
    
    # 3. Gold log-returns
    factors['GLD_log_return'] = np.log(p['GLD'] / p['GLD'].shift(1))
    
    # 4. EUR/USD log-return
    if 'EURUSD=X' in p.columns:
        factors['EURUSD_log_return'] = np.log(p['EURUSD=X'] / p['EURUSD=X'].shift(1))
    elif 'EURUSD' in p.columns:
        factors['EURUSD_log_return'] = np.log(p['EURUSD'] / p['EURUSD'].shift(1))
        
    # 5. SPY level change
    factors['SPY_level_change'] = p['SPY'] - p['SPY'].shift(1)
    
    # 6. VIX implied volatility change
    factors['VIX_change'] = v - v.shift(1)
    
    factors = factors.dropna()
    print(f"Risk factors: {len(factors)} obs x {len(factors.columns)} factors")
    return factors

if __name__ == "__main__":
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    prices = pd.read_csv(os.path.join(RAW_DIR, "prices.csv"),
                          index_col=0, parse_dates=True)
    vix = pd.read_csv(os.path.join(RAW_DIR, "vix.csv"),
                       index_col=0, parse_dates=True)
    dgs10 = pd.read_csv(os.path.join(RAW_DIR, "dgs10.csv"),
                         index_col=0, parse_dates=True)

    log_returns       = compute_log_returns(prices)
    portfolio_returns = compute_portfolio_returns(log_returns, WEIGHTS_DICT)
    risk_factors      = compute_risk_factors(prices, vix, dgs10)

    log_returns.to_csv(os.path.join(PROCESSED_DIR, "log_returns.csv"))
    portfolio_returns.to_csv(os.path.join(PROCESSED_DIR, "portfolio_returns.csv"))
    risk_factors.to_csv(os.path.join(PROCESSED_DIR, "risk_factors.csv"))
    print("Saved -> log_returns.csv, portfolio_returns.csv, risk_factors.csv")