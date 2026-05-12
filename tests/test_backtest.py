"""
backtesting/test_backtest.py
============================
Smoke-test and demo for the centralised backtesting module.

Generates synthetic data that intentionally mimics two scenarios:
  1. A "well-calibrated" 99% VaR model (expect ~1% exceptions)
  2. A "fat-tail" model that systematically underestimates tail risk
     (exception rate >> 1%, clustering expected)

Run from the repo root:
    python -m backtesting.test_backtest
"""

import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtesting.backtest import run_backtest, compare_methods

rng = np.random.default_rng(42)
T   = 4522  # approx 2006–2024 trading days
dates = pd.bdate_range("2006-01-02", periods=T)

# ── Scenario 1: Well-calibrated normal VaR ────────────────────────────────
# Simulate N(0, 0.01) returns; VaR set at 99th percentile of same distribution
sigma    = 0.01
returns  = rng.normal(0, sigma, T)
pnl_good = pd.Series(returns * 100_000, index=dates)      # scale to USD P&L
var_good = pd.Series(sigma * 2.326 * 100_000, index=dates)  # 99% normal VaR (fixed)

result_good = run_backtest(
    pnl=pnl_good,
    var=var_good,
    confidence=0.99,
    method_name="GoodModel (N=0.01)",
)
print(result_good)
print()

# ── Scenario 2: Fat-tail returns, normal VaR (underestimates tail risk) ──
# Uses t(df=4) returns so tails are much fatter than normal;
# VaR still set assuming normality → many more exceptions than expected,
# and they cluster around volatility spikes.
from scipy.stats import t as t_dist

fat_returns = t_dist.rvs(df=4, scale=sigma, size=T, random_state=rng)
# Inject a 10-day stress cluster to ensure independence is rejected
fat_returns[1000:1010] = -0.05
pnl_fat  = pd.Series(fat_returns * 100_000, index=dates)
var_fat  = pd.Series(sigma * 2.326 * 100_000, index=dates)   # same normal VaR

result_fat = run_backtest(
    pnl=pnl_fat,
    var=var_fat,
    confidence=0.99,
    method_name="FatTailModel (t-dist returns)",
)
print(result_fat)
print()

# ── Comparison table ──────────────────────────────────────────────────────
print("── Comparison across methods ──")
summary = compare_methods([result_good, result_fat])
print(summary.to_string())

# ── Assert expected outcomes ──────────────────────────────────────────────
assert not result_good.reject_uc,  "Good model should NOT be rejected by Kupiec"
assert result_fat.reject_uc,       "Fat-tail model SHOULD be rejected by Kupiec"
assert result_fat.reject_cc,       "Fat-tail model SHOULD be rejected by CC test"
print("\nAll assertions passed.")