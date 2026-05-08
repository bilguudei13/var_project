# =============================================================================
# compute_returns.py
# Computes log returns and linear portfolio returns from raw prices,
# and builds the combined 6-factor return matrix needed for MC simulation.
#
# Inputs:  data/raw/prices.csv
#          data/raw/vix.csv
#          data/raw/dgs10.csv
# Outputs: data/processed/log_returns.csv          (4 linear assets)
#          data/processed/portfolio_returns.csv     (scalar portfolio return)
#          data/processed/all_factor_returns.csv    (6 risk factors for MC)
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
    Log returns for the 4 linear assets: r_t = log(P_t / P_{t-1})
    Valid approximation for discrete returns at 1-day horizon (Irle p. 35).
    Theory: report/theoretical_background.md — Section 2
    """
    log_returns = np.log(prices / prices.shift(1)).dropna()
    print(f"Log returns:  {len(log_returns)} obs x {len(log_returns.columns)} assets")
    return log_returns

def compute_vix_returns(vix):
    """
    Log returns of VIX level: r_VIX_t = log(VIX_t / VIX_{t-1})

    VIX is treated as a risk factor whose *proportional* changes drive the
    straddle's vega P&L. Log returns are appropriate here for the same reason
    as for asset prices — they are more stationary than raw level changes.

    Used in MC to simulate the next-day VIX level:
        VIX_{t+1} = VIX_t * exp(r_VIX_sim)  =>  sigma_{t+1} = VIX_{t+1} / 100
    """
    vix_ret = np.log(vix / vix.shift(1)).dropna()
    vix_ret.columns = ["VIX_ret"]
    print(f"VIX returns:  {len(vix_ret)} obs")
    return vix_ret

def compute_dgs10_changes(dgs10):
    """
    Absolute daily changes in the 10Y Treasury yield, expressed in decimal.
        dDGS10_t = (DGS10_t - DGS10_{t-1}) / 100

    We use ABSOLUTE changes (not log returns) because:
      - Log returns of rates are poorly defined when rates approach zero.
      - IRS mark-to-market is linear in rate changes (DV01 approximation),
        so absolute changes map directly to dollar P&L.
      - This matches market convention (yield moves quoted in basis points).

    Used in MC to simulate the next-day yield:
        rate_{t+1} = rate_t + dDGS10_sim    (both in decimal)
    """
    dgs10_chg = dgs10.diff().dropna() / 100   # convert % to decimal
    dgs10_chg.columns = ["DGS10_chg"]
    print(f"DGS10 changes:{len(dgs10_chg)} obs")
    return dgs10_chg

def combine_all_factors(log_returns, vix_ret, dgs10_chg):
    """
    Join all 6 risk factor returns on their common dates.

    Column order: [EURUSD, GLD, IEF, SPY, VIX_ret, DGS10_chg]
      - Columns 0-3 : log returns of linear assets  (drive linear P&L)
      - Column  4   : VIX log return                (drives straddle revaluation)
      - Column  5   : DGS10 absolute change         (drives IRS revaluation)

    The 6x6 covariance matrix estimated from this matrix is the core input
    to the Monte Carlo simulation (Irle p. 87, Section 6.2).
    """
    common = (log_returns.index
              .intersection(vix_ret.index)
              .intersection(dgs10_chg.index))

    all_factors = pd.concat([
        log_returns.loc[common],
        vix_ret.loc[common],
        dgs10_chg.loc[common]
    ], axis=1)

    print(f"\nCombined factor matrix: {all_factors.shape[0]} obs x {all_factors.shape[1]} factors")
    print(f"  Columns : {list(all_factors.columns)}")
    print(f"  Range   : {all_factors.index[0].date()} -> {all_factors.index[-1].date()}")
    print(f"  NaNs    : {all_factors.isnull().sum().to_dict()}")
    return all_factors

def compute_portfolio_returns(log_returns, weights_dict):
    """
    Scalar portfolio return: R^{d,P} = sum_i w_i * R_i   (Irle Eq. 10)
    """
    weights_aligned = {
        ("EURUSD" if k == "EURUSD=X" else k): v
        for k, v in weights_dict.items()
    }
    w = pd.Series(weights_aligned)[log_returns.columns]
    portfolio_returns = log_returns.dot(w)
    portfolio_returns.name = "portfolio"
    print(f"Portfolio returns: mean={portfolio_returns.mean():.6f}")
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

    # --- Load raw data ---
    prices = pd.read_csv(os.path.join(RAW_DIR, "prices.csv"),
                         index_col=0, parse_dates=True)
    vix    = pd.read_csv(os.path.join(RAW_DIR, "vix.csv"),
                         index_col=0, parse_dates=True)
    dgs10  = pd.read_csv(os.path.join(RAW_DIR, "dgs10.csv"),
                         index_col=0, parse_dates=True)

    # --- Compute returns for each factor type ---
    log_returns       = compute_log_returns(prices)
    vix_ret           = compute_vix_returns(vix)
    dgs10_chg         = compute_dgs10_changes(dgs10)
    portfolio_returns = compute_portfolio_returns(log_returns, WEIGHTS_DICT)
    risk_factors      = compute_risk_factors(prices, vix, dgs10)

    # --- Build combined 6-factor matrix ---
    all_factors = combine_all_factors(log_returns, vix_ret, dgs10_chg)

    # --- Save ---
    log_returns.to_csv(os.path.join(PROCESSED_DIR, "log_returns.csv"))
    portfolio_returns.to_csv(os.path.join(PROCESSED_DIR, "portfolio_returns.csv"))
    all_factors.to_csv(os.path.join(PROCESSED_DIR, "all_factor_returns.csv"))

    print("\nSaved:")
    print(f"  -> {os.path.join(PROCESSED_DIR, 'log_returns.csv')}")
    print(f"  -> {os.path.join(PROCESSED_DIR, 'portfolio_returns.csv')}")
    print(f"  -> {os.path.join(PROCESSED_DIR, 'all_factor_returns.csv')}")
