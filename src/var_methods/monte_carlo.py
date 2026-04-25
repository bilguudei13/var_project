# =============================================================================
# monte_carlo.py
# 1-day 99% Monte Carlo VaR with full revaluation of non-linear instruments
#
# Theory references (Irle lecture notes + theoretical_background.md):
#
# [Section 6.2, Irle p. 87-88] Monte Carlo simulation:
#   Simulate M scenarios from an assumed joint distribution of risk factor
#   changes, compute P&L for each scenario, estimate VaR as empirical quantile.
#
# [Section 6.2, Irle p. 87] Multivariate normal assumption:
#   DeltaY = (DY^1, ..., DY^d)^T ~ N(mu, Sigma)
#   Parameters estimated from rolling window of k=500 observations.
#
# [Section 2, Irle p. 14-15, Eq. 1-3] Full revaluation:
#   For non-linear instruments, reprice using the exact pricing function
#   f(t, Y_t) rather than the linear (Delta) approximation.
#   This avoids the Delta-Gamma-Vega-Theta approximation error (Irle p. 82).
#
# Portfolio components (full revaluation for each MC scenario):
#   1. Linear positions: DV_linear = V0 * w' * sim_returns      (Irle Eq. 5-6)
#   2. IRS:              DV_irs    = V_irs(rate + drate_sim) - V_irs(rate)
#   3. Straddle:         DV_strad  = BS(S*exp(r_spy), K, T-dt, r, sig*exp(r_vix))
#                                  - BS(S, K, T, r, sig)
#
# [Section 8, Irle p. 183] Backtesting:
#   Exceptions N ~ B(T, 1-alpha) under correctly specified model.
#
# Simulation settings:
#   M      = 10,000 scenarios per day (stable 99th percentile estimate)
#   Window = 500 trading days (rolling)
#   Alpha  = 99%
# =============================================================================

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import norm

sys.path.append("src/data")
from config import (WEIGHTS_DICT, V0, IRS_NOTIONAL, IRS_FIXED_RATE,
                    STRADDLE_DAYS, STRADDLE_SHARES, RF_RATE,
                    RAW_DIR, PROCESSED_DIR)

# =============================================================================
# VECTORISED PRICING FUNCTIONS
# =============================================================================
# These mirror portfolio_pricing.py but accept numpy arrays for S and sigma
# (straddle) or swap_rate (IRS), enabling full revaluation of all M scenarios
# in a single numpy call rather than M Python-level loop iterations.

def price_straddle_vec(S, K, T, r, sigma):
    """
    Vectorised ATM straddle pricer (Black-Scholes).
    S, sigma : numpy arrays of shape (M,)  — one per simulated scenario
    K, T, r  : scalars                     — same for all scenarios
    Returns  : numpy array of shape (M,)   — straddle value per scenario
    """
    d1   = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2   = d1 - sigma * np.sqrt(T)
    call = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put  = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return call + put

def price_irs_vec(notional, fixed_rate, swap_rates, maturity=10):
    """
    Vectorised IRS mark-to-market pricer.
    swap_rates : numpy array of shape (M,) — one per simulated scenario
    Returns    : numpy array of shape (M,) — IRS value per scenario
    """
    return notional * (swap_rates - fixed_rate) * maturity / (1 + swap_rates)

# =============================================================================
# SETTINGS
# =============================================================================

WINDOW  = 500        # rolling estimation window (trading days)
ALPHA   = 0.99       # confidence level
M       = 10_000     # number of MC scenarios per day
SEED    = 42         # random seed for reproducibility

OUTPUT_FIGS   = os.path.join("outputs", "figures")
OUTPUT_TABLES = os.path.join("outputs", "tables")

# Column indices in all_factor_returns.csv
# Order: [EURUSD, GLD, IEF, SPY, VIX_ret, DGS10_chg]
IDX_LINEAR = [0, 1, 2, 3]   # columns for linear P&L
IDX_SPY    = 3               # SPY log return (for straddle revaluation)
IDX_VIX    = 4               # VIX log return (for straddle revaluation)
IDX_DGS10  = 5               # DGS10 absolute change in decimal (for IRS)

# =============================================================================
# STEP 1 — LOAD DATA
# =============================================================================

def load_data():
    """
    Load all inputs:
      - all_factor_returns : 6-factor return matrix (linear + VIX + DGS10)
      - prices             : raw SPY prices (for straddle spot price S_t)
      - vix                : raw VIX levels (for straddle sigma_t = VIX_t / 100)
      - dgs10              : raw DGS10 yields (for IRS rate_t = DGS10_t / 100)
      - total_pnl          : actual portfolio P&L for backtesting
    """
    factors = pd.read_csv(
        os.path.join(PROCESSED_DIR, "all_factor_returns.csv"),
        index_col=0, parse_dates=True
    )
    prices = pd.read_csv(
        os.path.join(RAW_DIR, "prices.csv"),
        index_col=0, parse_dates=True
    )
    vix = pd.read_csv(
        os.path.join(RAW_DIR, "vix.csv"),
        index_col=0, parse_dates=True
    ).squeeze()
    dgs10 = pd.read_csv(
        os.path.join(RAW_DIR, "dgs10.csv"),
        index_col=0, parse_dates=True
    ).squeeze()
    total_pnl = pd.read_csv(
        os.path.join(PROCESSED_DIR, "total_portfolio_pnl.csv"),
        index_col=0, parse_dates=True
    )["pnl_total"]

    # Align all series to factor return dates (already the tightest intersection)
    common = factors.index
    prices = prices.reindex(common, method="ffill")
    vix    = vix.reindex(common, method="ffill")
    dgs10  = dgs10.reindex(common, method="ffill")

    print(f"Factor matrix   : {factors.shape}  | cols: {list(factors.columns)}")
    print(f"Date range      : {common[0].date()} -> {common[-1].date()}")
    print(f"Total P&L obs   : {len(total_pnl.dropna())}")
    return factors, prices, vix, dgs10, total_pnl

# =============================================================================
# STEP 2 — MONTE CARLO VAR
# =============================================================================

def compute_mc_var(factors, prices, vix, dgs10, weights_dict,
                   window=WINDOW, alpha=ALPHA, M=M, seed=SEED):
    """
    Rolling 1-day 99% Monte Carlo VaR with full revaluation.

    For each day t (from t=window to t=T):

      (a) ESTIMATE distribution parameters from rolling window [t-window, t):
            mu    = sample mean of 6-factor returns    (6-vector)
            Sigma = sample covariance matrix            (6x6)

      (b) SIMULATE M scenarios via Cholesky decomposition (Irle p. 87):
            L     = cholesky(Sigma)       s.t. L @ L.T = Sigma
            Z     ~ N(0, I)              (M x 6 independent standard normals)
            sim   = Z @ L.T + mu         (M x 6 correlated factor returns)

      (c) FULL REVALUATION for each of M scenarios:
            Linear  : DV_i = V0 * w' * sim_i[linear]
            IRS     : DV_i = price_irs(rate + sim_i[DGS10]) - price_irs(rate)
            Straddle: DV_i = BS(S*exp(sim_i[SPY]), K, T-1/252, r,
                                sigma*exp(sim_i[VIX])) - BS(S, K, T, r, sigma)
            Total   : DV_total_i = DV_linear + DV_irs + DV_straddle

      (d) VaR = -Q_{1-alpha} of {DV_total_i}_{i=1..M}     (Irle p. 80)

    Straddle state (K, T) tracks the rolling 30-day ATM reset logic.
    """
    np.random.seed(seed)

    # Portfolio weights aligned to factor column order [EURUSD, GLD, IEF, SPY]
    weights_aligned = {("EURUSD" if k == "EURUSD=X" else k): v
                       for k, v in weights_dict.items()}
    linear_cols = factors.columns[IDX_LINEAR].tolist()
    w = np.array([weights_aligned[c] for c in linear_cols])  # (4,)

    n          = len(factors)
    factor_arr = factors.values     # (n x 6) numpy array for speed

    records, dates = [], []

    # --- Straddle state tracking (mirrors compute_pnl.py logic) ---
    spy_prices = prices["SPY"]
    K         = spy_prices.iloc[0]   # initial ATM strike
    days_held = 0

    print(f"\n{'='*65}")
    print(f"Monte Carlo VaR  —  Irle Section 6.2")
    print(f"M={M:,} scenarios | window={window} | alpha={alpha:.0%}")
    print(f"Full revaluation: linear + IRS + straddle")
    print(f"Computing VaR for {n - window} days...")

    for t in range(window, n):

        # ------------------------------------------------------------------ #
        # (a) ESTIMATE mu and Sigma from rolling window
        # ------------------------------------------------------------------ #
        W     = factor_arr[t - window : t]   # (500 x 6)
        mu    = W.mean(axis=0)               # (6,)
        Sigma = np.cov(W, rowvar=False)      # (6 x 6)

        # Small regularisation to guarantee Sigma is positive definite.
        # Necessary because sample covariance can be numerically singular,
        # which would cause cholesky() to fail.
        Sigma += 1e-8 * np.eye(6)

        # ------------------------------------------------------------------ #
        # (b) SIMULATE M scenarios via Cholesky decomposition
        # ------------------------------------------------------------------ #
        # Decompose: Sigma = L @ L.T
        # Then: sim = Z @ L.T + mu  has covariance Sigma
        # Proof: Cov(L @ z) = L @ Cov(z) @ L.T = L @ I @ L.T = Sigma
        L   = np.linalg.cholesky(Sigma)      # (6 x 6) lower triangular
        Z   = np.random.standard_normal((M, 6))
        sim = Z @ L.T + mu                   # (M x 6) correlated scenarios

        # ------------------------------------------------------------------ #
        # (c) FULL REVALUATION
        # ------------------------------------------------------------------ #

        # --- Current state of non-linear instruments at time t ---
        S_now     = spy_prices.iloc[t]              # SPY spot price
        sigma_now = vix.iloc[t] / 100               # implied vol (decimal)
        rate_now  = dgs10.iloc[t] / 100             # 10Y rate (decimal)

        # Straddle: reset strike K every 30 days (mirrors compute_pnl.py)
        if days_held >= STRADDLE_DAYS:
            K, days_held = S_now, 0
        T_now = max((STRADDLE_DAYS - days_held) / 252, 1 / 252)
        days_held += 1

        # Simulated next-day state for each of M scenarios
        S_sim     = S_now    * np.exp(sim[:, IDX_SPY])     # (M,) SPY prices
        sigma_sim = sigma_now * np.exp(sim[:, IDX_VIX])    # (M,) implied vols
        rate_sim  = rate_now  + sim[:, IDX_DGS10]          # (M,) rates
        T_next    = max(T_now - 1 / 252, 1 / 252)          # time decay by 1 day

        # Current instrument values (scalar, same for all scenarios)
        v_strad_now = price_straddle_vec(S_now, K, T_now, RF_RATE, sigma_now)
        v_irs_now   = price_irs_vec(IRS_NOTIONAL, IRS_FIXED_RATE, rate_now)

        # Vectorised revaluation across all M scenarios — single numpy calls
        pnl_linear   = V0 * sim[:, IDX_LINEAR] @ w                # (M,)

        # IRS: one numpy call over the (M,) rate array
        v_irs_sim    = price_irs_vec(IRS_NOTIONAL, IRS_FIXED_RATE, rate_sim)
        pnl_irs      = v_irs_sim - v_irs_now                       # (M,)

        # Straddle: one numpy call over (M,) S and sigma arrays
        v_strad_sim  = price_straddle_vec(S_sim, K, T_next, RF_RATE, sigma_sim)
        pnl_straddle = (v_strad_sim - v_strad_now) * STRADDLE_SHARES  # (M,)

        # Total P&L across all M scenarios
        pnl_total = pnl_linear + pnl_irs + pnl_straddle           # (M,)

        # ------------------------------------------------------------------ #
        # (d) VaR = negative of the (1-alpha) empirical quantile
        # ------------------------------------------------------------------ #
        VaR_t = -np.percentile(pnl_total, (1 - alpha) * 100)

        records.append({
            "VaR_MC"       : VaR_t,
            "mean_pnl_sim" : pnl_total.mean(),
            "std_pnl_sim"  : pnl_total.std(),
        })
        dates.append(factors.index[t])

    results = pd.DataFrame(records, index=dates)

    print(f"\n{'':20s} {'Value':>12s}")
    print(f"  Mean VaR          ${results['VaR_MC'].mean():>12,.0f}")
    print(f"  Min  VaR          ${results['VaR_MC'].min():>12,.0f}")
    print(f"  Max  VaR          ${results['VaR_MC'].max():>12,.0f}")

    return results

# =============================================================================
# STEP 3 — BACKTESTING
# =============================================================================

def backtest(var_series, total_pnl, alpha=ALPHA):
    """
    Count VaR exceptions against actual total portfolio P&L (Irle p. 183-184).

    We use total_portfolio_pnl (linear + IRS + straddle) as the actual P&L,
    which is consistent with the full revaluation used in the MC VaR.

    Exception: actual loss -DV_t > VaR_t
    Under H0 (correct model): N ~ B(T, 1-alpha)
    """
    pnl_aligned = total_pnl.reindex(var_series.index)
    actual_loss = -pnl_aligned
    exceptions  = actual_loss > var_series["VaR_MC"]

    T        = len(exceptions.dropna())
    N_exc    = int(exceptions.sum())
    exc_rate = N_exc / T
    p        = 1 - alpha

    lower = int(np.floor(T * p - 1.96 * np.sqrt(T * p * (1 - p))))
    upper = int(np.ceil( T * p + 1.96 * np.sqrt(T * p * (1 - p))))

    print(f"\n{'='*65}")
    print(f"BACKTESTING  —  Irle p. 183-184")
    print(f"{'='*65}")
    print(f"T (observations)         : {T}")
    print(f"Expected E[N] = T*0.01   : {T * p:.1f}")
    print(f"95% CI  B(T, 0.01)       : [{lower}, {upper}]")
    print(f"Actual exceptions N      : {N_exc}")
    print(f"Actual rate              : {exc_rate:.2%}")
    print(f"Within 95% CI?           : {'YES ✓' if lower <= N_exc <= upper else 'NO ✗'}")

    if N_exc > upper:
        print(f"\n→ MC VaR UNDERESTIMATES risk (too many exceptions)")
        print(f"  Likely cause: normal marginals understate tail risk.")
        print(f"  Consider Student-t marginals + Gaussian copula (Irle p. 88).")
    elif N_exc < lower:
        print(f"\n→ MC VaR OVERESTIMATES risk (too few exceptions)")
    else:
        print(f"\n→ MC VaR adequately calibrated")

    return exceptions, pnl_aligned

# =============================================================================
# STEP 4 — PLOT
# =============================================================================

def plot_var(results, pnl, exceptions):
    var_series  = results["VaR_MC"]
    actual_loss = -pnl

    crises = [
        ("2008-09-15", "2009-03-09", "GFC 2008"),
        ("2020-02-19", "2020-03-23", "COVID 2020"),
        ("2022-01-01", "2022-10-01", "Rate Hikes"),
    ]

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    # Panel 1: VaR vs actual loss
    axes[0].fill_between(var_series.index, 0, var_series,
                         alpha=0.12, color="#9C27B0")
    axes[0].plot(var_series.index, var_series,
                 color="#9C27B0", linewidth=1.2, label="MC VaR 99%")
    axes[0].plot(actual_loss.index, actual_loss.reindex(var_series.index),
                 color="#90A4AE", linewidth=0.6, alpha=0.7, label="Actual loss (−ΔV)")
    breaches = exceptions[exceptions].index
    axes[0].scatter(breaches, actual_loss.loc[breaches],
                    color="#F44336", s=15, zorder=5,
                    label=f"Exceptions (N={len(breaches)})")
    axes[0].axhline(0, color="black", linewidth=0.5, linestyle="--")
    axes[0].set_title("Monte Carlo VaR (full revaluation) vs Actual Portfolio Loss | Irle Section 6.2",
                      fontsize=12, fontweight="bold")
    axes[0].set_ylabel("USD")
    axes[0].legend(fontsize=9)

    # Panel 2: Rolling simulated P&L std dev
    axes[1].plot(results.index, results["std_pnl_sim"],
                 color="#FF9800", linewidth=1.0)
    axes[1].set_title("Rolling σ of Simulated P&L — Captures volatility clustering via rolling Σ",
                      fontsize=12, fontweight="bold")
    axes[1].set_ylabel("σ(P&L sim) (USD)")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    for ax in axes:
        for s, e, lbl in crises:
            ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), alpha=0.08, color="red")

    plt.tight_layout()
    os.makedirs(OUTPUT_FIGS, exist_ok=True)
    path = os.path.join(OUTPUT_FIGS, "07_mc_var.png")
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"\nPlot saved -> {path}")

# =============================================================================
# STEP 5 — SAVE
# =============================================================================

def save_results(results, exceptions, pnl):
    os.makedirs(OUTPUT_TABLES, exist_ok=True)

    results.to_csv(os.path.join(PROCESSED_DIR, "var_mc.csv"))
    print(f"VaR saved  -> {os.path.join(PROCESSED_DIR, 'var_mc.csv')}")

    pd.DataFrame({
        "VaR_MC"      : results["VaR_MC"],
        "actual_loss" : (-pnl).reindex(results.index),
        "exception"   : exceptions,
    }).to_csv(os.path.join(OUTPUT_TABLES, "backtest_mc.csv"))
    print(f"Backtest   -> {os.path.join(OUTPUT_TABLES, 'backtest_mc.csv')}")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    os.makedirs(OUTPUT_FIGS,   exist_ok=True)
    os.makedirs(OUTPUT_TABLES, exist_ok=True)

    factors, prices, vix, dgs10, total_pnl = load_data()

    results                  = compute_mc_var(factors, prices, vix, dgs10, WEIGHTS_DICT)
    exceptions, pnl_aligned  = backtest(results, total_pnl)

    plot_var(results, pnl_aligned, exceptions)
    save_results(results, exceptions, pnl_aligned)

    print(f"\n{'='*65}\nMonte Carlo VaR complete!\n{'='*65}")
