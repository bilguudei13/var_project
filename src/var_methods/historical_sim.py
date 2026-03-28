# =============================================================================
# THEORETICAL BACKGROUND
# Full derivations and formulas: report/background.md
#
# Relevant sections for this script:
#   Section 1  — Risk Factors and Risk Mapping
#   Section 2  — Returns: Log vs Discrete
#   Section 6  — Historical Simulation VaR    (Irle p. 83-85)
#   Section 11 — Backtesting                  (Irle p. 183-184)
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# =============================================
# SETTINGS
# =============================================

WINDOW  = 500
ALPHA   = 0.99
V0      = 1_000_000
WEIGHTS = np.ones(6) / 6

PROCESSED_DIR = os.path.join("data", "processed")
OUTPUT_FIGS   = os.path.join("ouputs", "figures")
OUTPUT_TABLES = os.path.join("ouputs", "tables")

# =============================================
# STEP 1. LOAD DATA
# =============================================

def load_data():
    """
    Load log-returns as approximation of discrete returns R^{d,i}.
    Valid for 1-day horizon (Irle page 35: log ≈ discrete for small dt).
    """
    returns = pd.read_csv(
        os.path.join(PROCESSED_DIR, "log_returns.csv"),
        index_col=0, parse_dates=True
    )
    print(f"Loaded: {returns.shape[0]} days x {returns.shape[1]} assets")
    print(f"Risk factors Y: {list(returns.columns)}")
    return returns

# =============================================
# STEP 2. VaR Computation
# =============================================

def compute_historical_sim_var(returns, weights, window, alpha, V0):
    """
    Rolling 1-day Historical Simulation VaR.

    Theory: report/theoretical_background.md — Section 6
    
    Algorithm (Section 6, Irle p. 83-85):
      1. Collect k historical factor returns: dy_1, ..., dy_k
      2. Compute portfolio P&L for each scenario:
             DV_i = V0 * w' * dy_i          (Section 1, Eq. 5-6)
      3. VaR = -Q_{1-alpha} of {DV_i}       (Section 6, Irle p. 80)

    No distributional assumption required — fat tails and dependencies
    are captured automatically from historical data (Section 6).
    """
    n = len(returns)
    records, dates = [], []

    for t in range(window, n):
        
        # 1. Collect k historical factor returns: dy_1, ..., dy_k
        window_data = returns.iloc[t - window :t]

        # 2. Compute portfolio P&L for each scenario
        # DV_i = V0 * w' * dy_i          (Section 1, Eq. 5-6)
        pnl_scenarios = V0 * window_data.dot(weights)

        # 3. VaR = -Q_{1-alpha} of {DV_i}       (Section 6, Irle p. 80)
        VaR_t = -np.percentile(pnl_scenarios, (1 - alpha) * 100)

        records.append({"VaR_HistSim"}: VaR_t})
        dates.append(returns.index[t])

        return pd.DataFrame(records, index=dates)
    
# =============================================
# STEP 3. BACKTESTING
# =============================================
