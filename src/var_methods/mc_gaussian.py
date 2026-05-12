# =============================================================================
# mc_gaussian.py
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
#   1. Linear positions: DV_linear = sum_j shares_j * P_{t-1,j} * (exp(sim_j) - 1)
#                        where shares_j = V0 * w_j / P_{0,j} are the inception
#                        share counts (fixed, never rebalanced) — same basis
#                        as compute_pnl.py so the simulated and realised P&L
#                        share the same dollar scale as the portfolio drifts. (Irle Eq. 5-6)
#   2. IRS:              DV_irs    = V_irs(rate + drate_sim) - V_irs(rate)
#   3. Straddle:         DV_strad  = BS(S*exp(r_spy), K, T-dt, r, sig*exp(r_vix))
#                                  - BS(S, K, T, r, sig)
#
# [Section 8, Irle p. 183] Backtesting:
#   Exceptions N ~ B(T, 1-alpha) under correctly specified model.
#
# Simulation settings:
#   M      = 10,000 scenarios per day (stable 99th percentile estimate)
#   Window = 750 trading days (rolling)
#   Alpha  = 99%
# =============================================================================

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from scipy.stats import norm

_ROOT = Path(__file__).resolve().parent.parent.parent   # var_methods/ → src/ → var_project/
sys.path.insert(0, str(_ROOT / "src" / "data"))
sys.path.insert(0, str(_ROOT))
from config import (WEIGHTS_DICT, V0, IRS_NOTIONAL, IRS_FIXED_RATE,
                    STRADDLE_DAYS, STRADDLE_SHARES, RF_RATE,
                    RAW_DIR, PROCESSED_DIR)
from portfolio_pricing import build_straddle_state
from portfolio_pricing import price_irs as _price_irs_prod
from backtesting.backtest import run_backtest

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
    Vectorised IRS mark-to-market pricer using full discrete annuity valuation.
    Delegates to portfolio_pricing.price_irs for consistency with the realized
    P&L pipeline and mc_garch_t_copula.py.
    swap_rates : numpy array of shape (M,) — one per simulated scenario
    Returns    : numpy array of shape (M,) — IRS value per scenario
    """
    value, _ = _price_irs_prod(notional, fixed_rate, swap_rates, maturity=maturity)
    return value

# =============================================================================
# SETTINGS
# =============================================================================

WINDOW  = 750        # rolling estimation window (trading days)
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
            Linear  : DV_i = sum_j shares_j * P_{t-1,j} * (exp(sim_i[j]) - 1)
                      shares_j = V0 * w_j / P_{0,j} fixed at inception, so the
                      linear leg scales with the drifted current portfolio
                      value and matches compute_pnl.py's realised basis.
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

    # Static inception shares — matching compute_pnl.py convention exactly.
    # shares_j = V0 * w_j / initial_price_j, fixed at inception, never rebalanced.
    prices_linear = prices.rename(columns={"EURUSD=X": "EURUSD"})[linear_cols]
    prices_linear = prices_linear.reindex(factors.index, method="ffill")
    linear_shares = V0 * w / prices_linear.iloc[0].values   # (4,) fixed at inception

    n          = len(factors)
    factor_arr = factors.values     # (n x 6) numpy array for speed

    records, dates = [], []

    spy_prices     = prices["SPY"]
    straddle_state = build_straddle_state(spy_prices, straddle_days=STRADDLE_DAYS)

    print(f"\n{'='*65}")
    print(f"Monte Carlo VaR  —  Irle Section 6.2")
    print(f"M={M:,} scenarios | window={window} | alpha={alpha:.0%}")
    print(f"Full revaluation: linear + IRS + straddle")
    print(f"Computing VaR for {n - window} days...")

    for t in range(window, n):

        # ------------------------------------------------------------------ #
        # (a) ESTIMATE mu and Sigma from rolling window
        # ------------------------------------------------------------------ #
        W     = factor_arr[t - window : t]   # (750 x 6) — the rolling estimation window
        # mu is the vector of sample mean daily returns for each risk factor.
        # Economically it represents the drift: the expected return per day
        # for EURUSD, GLD, IEF, SPY (log returns), VIX (log change), and
        # DGS10 (absolute change in decimal).  For simulation purposes mu is
        # typically very small relative to sigma, but omitting it would
        # introduce a small systematic bias in the simulated scenarios.
        mu    = W.mean(axis=0)               # (6,)
        # Sigma is the 6×6 sample covariance matrix of daily factor returns.
        # Economically it encodes both the individual volatilities of each
        # factor (diagonal entries) and the pairwise co-movements between
        # factors (off-diagonal entries, e.g. the negative SPY-VIX correlation
        # that drives the "vol spike on equity crash" dynamics).
        # We use a 750-day (approximately 3-year) rolling window rather than
        # the full history: short enough to respond to regime changes (a crisis
        # raises Sigma within weeks), yet long enough that the 6×6 covariance
        # matrix is estimated from at least 750 > 6 observations and does not
        # degenerate.  This is the recency-vs-stability tradeoff: a shorter
        # window adapts faster but produces a noisier, potentially rank-deficient
        # Sigma; a longer window is more stable but may average over outdated
        # volatility regimes.
        Sigma = np.cov(W, rowvar=False)      # (6 x 6)

        # Numerical regularisation: add a tiny multiple of the identity matrix
        # to the diagonal of Sigma before computing the Cholesky decomposition.
        # The sample covariance of 750 observations across 6 factors can be
        # nearly singular (or exactly singular if any factor is a near-linear
        # combination of others), which would cause np.linalg.cholesky to raise
        # a LinAlgError.  Adding 1e-8 * I shifts all eigenvalues up by 1e-8,
        # guaranteeing positive definiteness while having negligible economic
        # impact (typical variances are on the order of 1e-4 to 1e-3).
        Sigma += 1e-8 * np.eye(6)

        # ------------------------------------------------------------------ #
        # (b) SIMULATE M scenarios via Cholesky decomposition
        # ------------------------------------------------------------------ #
        # The Cholesky decomposition finds a lower-triangular matrix L such
        # that L @ L.T = Sigma.  This is the multivariate analogue of taking
        # a square root: just as X = mu + sigma * z transforms a standard
        # normal z into N(mu, sigma^2), the transform sim = Z @ L.T + mu
        # converts M independent standard normals into M draws from N(mu, Sigma).
        #
        # The covariance proof: let z ~ N(0, I) and define x = L @ z.
        # Then Cov(x) = L @ Cov(z) @ L.T = L @ I @ L.T = L @ L.T = Sigma.
        # Adding mu shifts the mean without changing the covariance structure.
        # Because Z has shape (M, 6) (rows are independent draws), multiplying
        # by L.T on the right applies the correlation structure to each row,
        # yielding M rows each of which is a single correlated 6-factor scenario.
        L   = np.linalg.cholesky(Sigma)      # (6 x 6) lower triangular
        Z   = np.random.standard_normal((M, 6))
        # Each row of sim is one simulated next-day vector of factor changes:
        # [EURUSD_ret, GLD_ret, IEF_ret, SPY_ret, VIX_ret, DGS10_chg].
        # The correlation between columns is exactly Sigma as estimated above,
        # so the model captures, e.g., that large negative SPY returns tend to
        # coincide with large positive VIX returns (flight-to-safety dynamics).
        sim = Z @ L.T + mu                   # (M x 6) correlated scenarios

        # ------------------------------------------------------------------ #
        # (c) FULL REVALUATION
        # ------------------------------------------------------------------ #

        # --- Current state of non-linear instruments at t-1 (no look-ahead) ---
        S_now     = spy_prices.iloc[t - 1]          # SPY spot price known at start of day t
        sigma_now = vix.iloc[t - 1] / 100           # implied vol (decimal)
        rate_now  = dgs10.iloc[t - 1] / 100         # 10Y rate (decimal)

        K     = float(straddle_state.iloc[t - 1]["strike_spy"])
        T_now = float(straddle_state.iloc[t - 1]["tenor_years"])

        # Simulated next-day state for each of M scenarios.
        # SPY: the factor return for SPY is modelled as a log return
        # r_SPY = log(S_t / S_{t-1}), so the simulated next-day price is
        # S_sim = S_now * exp(r_SPY).  We use exp() rather than (1 + r_SPY)
        # because the simulation unit is the log return: compounding is exact
        # under log returns, whereas (1 + r_SPY) is only a first-order
        # approximation valid for small r.  Using exp() ensures S_sim > 0
        # even for very large negative shocks.
        S_sim     = S_now    * np.exp(sim[:, IDX_SPY])     # (M,) SPY prices
        # VIX: VIX_ret in the factor matrix is also a log return of the VIX
        # level: r_VIX = log(VIX_t / VIX_{t-1}).  Therefore the simulated
        # implied volatility is sigma_sim = sigma_now * exp(r_VIX), which
        # keeps sigma strictly positive and correctly applies log-return
        # compounding to the volatility level used in the straddle pricer.
        sigma_sim = sigma_now * np.exp(sim[:, IDX_VIX])    # (M,) implied vols
        # DGS10: unlike the equity and vol factors, DGS10_chg is an absolute
        # change in the yield level (in decimal), not a log return.  For example
        # a value of 0.0025 means the 10-year rate moved up by 25 basis points.
        # We therefore add the simulated change directly to the current rate,
        # rather than multiplying: rate_sim = rate_now + sim[:, IDX_DGS10].
        # This is consistent with how DGS10_chg is constructed in the data
        # pipeline and ensures that rate changes are in the same units (decimal
        # rate) as the IRS pricer expects.
        rate_sim  = rate_now  + sim[:, IDX_DGS10]          # (M,) rates
        # Time decay: the straddle expires in T_now years from today.  After
        # one trading day elapses (= 1/252 years), the time to expiry shrinks
        # to T_now - 1/252.  This theta effect is captured by repricing the
        # straddle at T_next rather than T_now — an important source of P&L
        # for short-dated options even in the absence of any underlying move.
        # The floor at 1/252 prevents T_next from reaching zero or going negative.
        T_next    = max(T_now - 1 / 252, 1 / 252)          # time decay by 1 day

        # Current instrument values (scalar, same for all scenarios).
        # Use the actual DGS10 rate (rate_now) as the Black-Scholes discount
        # factor instead of the constant RF_RATE=5%, matching the realized P&L
        # pipeline and mc_garch_t_copula.py.
        v_strad_now = price_straddle_vec(S_now, K, T_now, rate_now, sigma_now)
        v_irs_now   = price_irs_vec(IRS_NOTIONAL, IRS_FIXED_RATE, rate_now)

        # Vectorised revaluation across all M scenarios — single numpy calls
        prices_now   = prices_linear.iloc[t - 1].values
        dollar_pos   = linear_shares * prices_now
        pnl_linear   = (dollar_pos * (np.exp(sim[:, IDX_LINEAR]) - 1.0)).sum(axis=1)  # (M,)

        # IRS: one numpy call over the (M,) rate array
        v_irs_sim    = price_irs_vec(IRS_NOTIONAL, IRS_FIXED_RATE, rate_sim)
        pnl_irs      = v_irs_sim - v_irs_now                       # (M,)

        # Straddle: one numpy call over (M,) S and sigma arrays.
        # Use rate_sim for full revaluation: the simulated next-day discount
        # rate changes with the yield scenario, matching mc_garch_t_copula.py.
        v_strad_sim  = price_straddle_vec(S_sim, K, T_next, rate_sim, sigma_sim)
        pnl_straddle = (v_strad_sim - v_strad_now) * STRADDLE_SHARES  # (M,)

        # Total P&L across all M scenarios
        pnl_total = pnl_linear + pnl_irs + pnl_straddle           # (M,)

        # ------------------------------------------------------------------ #
        # (d) VaR = negative of the (1-alpha) empirical quantile
        # ------------------------------------------------------------------ #
        # np.percentile(pnl_total, 1) returns the 1st percentile of the
        # simulated P&L distribution — the dollar loss exceeded in only 1% of
        # the M scenarios.  Because losses are negative P&L values, this
        # percentile is itself a negative number (e.g., -$45,000 means the
        # portfolio lost more than $45k in 1% of scenarios).  The sign flip
        # (VaR = -Q_{1%}) converts this to a positive dollar figure that
        # represents the potential loss: "with 99% confidence, the 1-day
        # loss will not exceed VaR_t."  This sign convention follows the
        # industry standard where VaR is reported as a positive loss amount.
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
# STEP 3 — BACKTESTING  (central module — Kupiec + Christoffersen)
# =============================================================================
# run_backtest imported from backtesting.backtest at the top of this file.

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
    import sys as _sys
    if hasattr(_sys.stdout, 'reconfigure'):
        _sys.stdout.reconfigure(encoding='utf-8')
    os.makedirs(OUTPUT_FIGS,   exist_ok=True)
    os.makedirs(OUTPUT_TABLES, exist_ok=True)

    factors, prices, vix, dgs10, total_pnl = load_data()

    results    = compute_mc_var(factors, prices, vix, dgs10, WEIGHTS_DICT)
    var_series = results["VaR_MC"]
    pnl_bt     = total_pnl.reindex(var_series.index)

    bt = run_backtest(pnl=pnl_bt, var=var_series, confidence=ALPHA, method_name="MC-Gaussian")
    print(bt)

    exceptions = pnl_bt < -var_series
    plot_var(results, pnl_bt, exceptions)
    save_results(results, exceptions, pnl_bt)

    print(f"\n{'='*65}\nMonte Carlo VaR complete!\n{'='*65}")
