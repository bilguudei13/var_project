# =============================================================================
# evt.py
# 1-day 99% EVT VaR — Peaks-Over-Threshold (POT) / Generalised Pareto Distribution
#
# Theory references (Irle lecture notes):
#
# [Section 9 — Extreme Value Theory]
#
# [Pickands-Balkema-de Haan Theorem (Irle p. 213)]:
#   For large enough threshold u, the conditional excess distribution
#   F_u(x) = P(L - u ≤ x | L > u) converges to the GPD G_{ξ,σ}(x).
#
# [GPD definition (Irle p. 212, Definition 7)]:
#   G_{ξ,σ}(x) = 1 - (1 + ξ·x/σ)^{-1/ξ}   ξ ≠ 0
#              = 1 - exp(-x/σ)                ξ = 0  (Gumbel / exponential)
#   Parameters: shape ξ (tail index), scale σ > 0
#   ξ > 0 → heavy tail (Pareto), ξ = 0 → light tail, ξ < 0 → bounded tail
#
# [POT VaR formula (Irle p. 223-225)]:
#   VaR_α = u + (σ/ξ) · [(T_w/N_u · (1-α))^{-ξ} - 1]   ξ ≠ 0
#   VaR_α = u - σ · log(N_u / (T_w · (1-α)))              ξ ≈ 0  (Gumbel)
#   Where: T_w = rolling window size, N_u = exceedances above u
#
# [McNeil & Frey (2000), "Estimation of tail-related risk measures for
#  heteroscedastic financial time series", J. Empirical Finance 7, 271-300]
#
# Rolling window : 500 trading days
# Confidence     : α = 99%
# Threshold      : 90th percentile of losses in window  (~50 exceedances)
# =============================================================================

import os
import sys
import warnings
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # handle Unicode on Windows
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive: save to disk, no window
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import genpareto

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from backtesting.backtest import run_backtest

# =============================================================================
# SETTINGS
# =============================================================================

WINDOW          = 500
ALPHA           = 0.99
THRESHOLD_Q     = 0.90        # 90th percentile → ~50 exceedances per 500-day window
MIN_EXCEEDANCES = 10          # minimum to attempt GPD MLE; else empirical fallback
V0              = 1_000_000

PROCESSED_DIR = os.path.join("data", "processed")
OUTPUT_FIGS   = os.path.join("outputs", "figures")
OUTPUT_TABLES = os.path.join("outputs", "tables")

# =============================================================================
# STEP 1 — LOAD DATA
# =============================================================================

def load_data():
    """
    Load total portfolio P&L from total_portfolio_pnl.csv.

    EVT operates directly on dollar losses (L_t = -pnl_total_t).
    Using pnl_total ensures all instruments (linear, IRS, straddle) are included.
    """
    pnl = pd.read_csv(
        os.path.join(PROCESSED_DIR, "total_portfolio_pnl.csv"),
        index_col=0, parse_dates=True
    )["pnl_total"]
    pnl.name = "pnl"

    print(f"Loaded: {len(pnl)} days of portfolio P&L")
    print(f"Date range : {pnl.index[0].date()} -> {pnl.index[-1].date()}")
    print(f"P&L mean   : ${pnl.mean():>10,.0f}")
    print(f"P&L std    : ${pnl.std():>10,.0f}")
    print(f"P&L min    : ${pnl.min():>10,.0f}")
    print(f"P&L max    : ${pnl.max():>10,.0f}")
    return pnl

# =============================================================================
# STEP 2 — THRESHOLD DIAGNOSTICS (full-sample, static)
# =============================================================================

def plot_threshold_diagnostics(losses):
    """
    Full-sample threshold diagnostic plots to justify the 90th-percentile choice.

    (a) Mean Excess Plot (MEP):
        e(u) = E[L - u | L > u]  plotted against u.
        Under GPD, e(u) is linear in u (Irle p. 215).
        A linear region starting around the chosen threshold validates GPD.

    (b) Hill Estimator Plot:
        ξ_Hill(k) = (1/k) Σ_{i=1}^{k} [log X_{(n-i+1)} - log X_{(n-k)}]
        where X_{(1)} ≤ ... ≤ X_{(n)} are order statistics of losses.
        (Irle Section 9; Hill 1975)
        Stability of ξ_Hill in k confirms heavy-tail behaviour.

    These are exploratory / diagnostic figures, not used in the rolling VaR.
    """
    loss_vals = losses.values
    n = len(loss_vals)

    # (a) Mean Excess Plot on a grid of thresholds
    u_quantiles = np.linspace(0.50, 0.98, 200)
    u_grid      = np.quantile(loss_vals, u_quantiles)
    mean_excess = []
    for u in u_grid:
        exc = loss_vals[loss_vals > u] - u
        mean_excess.append(exc.mean() if len(exc) >= 5 else np.nan)
    mean_excess = np.array(mean_excess)

    # (b) Hill estimator for k = 1..200
    x_desc = np.sort(loss_vals)[::-1]           # descending order statistics
    max_k  = min(300, n - 1)
    k_vals = np.arange(1, max_k + 1)
    hill   = np.array([
        np.mean(np.log(x_desc[:k]) - np.log(x_desc[k]))
        for k in k_vals
    ])

    # Chosen threshold
    u_chosen = np.quantile(loss_vals, THRESHOLD_Q)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel (a): MEP
    mask = ~np.isnan(mean_excess)
    axes[0].plot(u_grid[mask], mean_excess[mask],
                 color="#2196F3", linewidth=1.4, label="Mean excess e(u)")
    axes[0].axvline(u_chosen, color="#F44336", linestyle="--", linewidth=1.6,
                    label=f"Chosen u = ${u_chosen:,.0f}  ({THRESHOLD_Q:.0%} quantile)")
    axes[0].set_xlabel("Threshold u  (USD)", fontsize=10)
    axes[0].set_ylabel("E[L − u | L > u]  (USD)", fontsize=10)
    axes[0].set_title("Mean Excess Plot (MEP)\n"
                      "Linear region above u validates GPD (Irle p. 215)",
                      fontsize=11, fontweight="bold")
    axes[0].legend(fontsize=9)

    # Panel (b): Hill estimator
    # k ≈ N_u when threshold = 90th pct of full sample
    k_chosen = int(np.round(n * (1 - THRESHOLD_Q)))
    axes[1].plot(k_vals, hill, color="#4CAF50", linewidth=1.2,
                 label="ξ_Hill(k)")
    axes[1].axvline(k_chosen, color="#F44336", linestyle="--", linewidth=1.6,
                    label=f"k at 90th pct ≈ {k_chosen}")
    axes[1].axhline(0, color="black", linewidth=0.7, linestyle=":")
    axes[1].set_xlabel("k  (top-k order statistics)", fontsize=10)
    axes[1].set_ylabel("Hill estimator ξ_Hill(k)", fontsize=10)
    axes[1].set_title("Hill Estimator Plot\n"
                      "ξ > 0 → heavy tail (Pareto-type) — Irle Section 9",
                      fontsize=11, fontweight="bold")
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(OUTPUT_FIGS, "evt_01_threshold_diagnostics.png")
    plt.savefig(path, dpi=150)
    plt.close("all")
    print(f"Threshold diagnostics saved -> {path}")

# =============================================================================
# STEP 3 — POT VaR HELPER + ROLLING COMPUTATION
# =============================================================================

def _pot_var(losses_w, threshold_q, alpha, T_w):
    """
    Compute single-period POT VaR via GPD fit.

    Algorithm (Irle p. 223-225):
      1. u = quantile(losses_w, threshold_q)
      2. exceedances = losses_w[losses_w > u] - u
      3. Fit GPD with floc=0:  scipy.stats.genpareto.fit(exceedances, floc=0)
         → shape ξ (= c in scipy), loc = 0, scale σ
      4. ratio = (T_w / N_u) * (1 - α)
         ξ ≠ 0: VaR = u + (σ/ξ) * (ratio^{-ξ} - 1)
         ξ ≈ 0: VaR = u - σ * log(ratio)            [Gumbel limit]

    Edge cases:
      - N_u < MIN_EXCEEDANCES  → empirical quantile fallback, xi = NaN
      - scipy GPD fit raises   → empirical quantile fallback, xi = NaN
      - result < u             → clamp to u (VaR cannot be below threshold)

    Returns
    -------
    var : float   VaR in same units as losses_w (USD)
    xi  : float   GPD shape parameter (NaN if fallback)
    """
    u = np.quantile(losses_w, threshold_q)
    exceedances = losses_w[losses_w > u] - u
    N_u = len(exceedances)

    if N_u < MIN_EXCEEDANCES:
        return np.quantile(losses_w, alpha), np.nan

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # genpareto.fit returns (shape=ξ, loc, scale=σ); floc=0 fixes location
            c, loc, sigma = genpareto.fit(exceedances, floc=0)
        xi = c

        ratio = (T_w / N_u) * (1.0 - alpha)

        if abs(xi) < 1e-4:                        # Gumbel limit: exponential exceedances
            var = u - sigma * np.log(ratio)
        else:
            var = u + (sigma / xi) * (ratio ** (-xi) - 1.0)

        var = max(var, u)                         # sanity: VaR ≥ threshold
        return var, xi

    except Exception:
        return np.quantile(losses_w, alpha), np.nan


def compute_evt_var(pnl, window=WINDOW, threshold_q=THRESHOLD_Q, alpha=ALPHA):
    """
    Rolling 500-day POT EVT VaR (Irle Section 9).

    For each day t in [window, T):
      losses_w = -pnl[t-window:t]   (dollar losses, positive = actual loss)
      Apply _pot_var() → VaR_t, ξ_t

    Losses are in dollar terms (from pnl_total), so no V0 scaling needed.

    Returns
    -------
    pd.DataFrame  columns: VaR_EVT, xi, threshold_u
    """
    n      = len(pnl)
    losses = (-pnl).values
    records, dates = [], []

    print(f"\n{'='*60}")
    print(f"EVT (POT) VaR  --  Irle Section 9")
    print(f"k={window} days | alpha={alpha:.0%} | threshold={threshold_q:.0%} quantile")
    print(f"Computing VaR for {n - window} days ...")

    n_fallbacks = 0
    for t in range(window, n):
        losses_w = losses[t - window : t]
        var_t, xi_t = _pot_var(losses_w, threshold_q, alpha, window)
        if np.isnan(xi_t):
            n_fallbacks += 1
        records.append({
            "VaR_EVT"     : var_t,
            "xi"          : xi_t,
            "threshold_u" : np.quantile(losses_w, threshold_q),
        })
        dates.append(pnl.index[t])

    results = pd.DataFrame(records, index=dates)
    xi_ok   = results["xi"].dropna()

    print(f"\nRolling EVT summary:")
    print(f"  Mean VaR     : ${results['VaR_EVT'].mean():>12,.0f}")
    print(f"  Min  VaR     : ${results['VaR_EVT'].min():>12,.0f}")
    print(f"  Max  VaR     : ${results['VaR_EVT'].max():>12,.0f}")
    print(f"  Mean xi      :  {xi_ok.mean():.4f}  (>0 -> heavy tail / Pareto)")
    print(f"  Fallback days: {n_fallbacks} / {n - window}")

    return results

# =============================================================================
# STEP 4 — BACKTEST
# =============================================================================

def backtest_evt(pnl, results):
    """
    Kupiec + Christoffersen backtests via shared framework.
    (Irle Chapter 8, p. 183-185 — same as all other VaR methods.)
    """
    bt = run_backtest(
        pnl=pnl,
        var=results["VaR_EVT"],
        confidence=ALPHA,
        method_name="EVT",
    )
    print(bt)
    return bt

# =============================================================================
# STEP 5 — PLOTS
# =============================================================================

def plot_evt_results(pnl, results, bt):
    """
    Three-panel figure:
      Panel 1: EVT VaR vs actual loss with exception markers
      Panel 2: Rolling threshold u over time
      Panel 3: Rolling ξ (GPD shape) over time
                ξ > 0 throughout confirms heavy-tail Pareto behaviour (Irle p. 213).
    Crisis periods shaded in red.
    """
    crises = [
        ("2008-09-15", "2009-03-09", "GFC 2008"),
        ("2020-02-19", "2020-03-23", "COVID 2020"),
        ("2022-01-01", "2022-10-01", "Rate Hikes 2022"),
    ]

    var_s       = results["VaR_EVT"]
    actual_loss = -pnl.reindex(results.index)
    exc_idx     = bt.exceptions_index

    fig, axes = plt.subplots(3, 1, figsize=(14, 13), sharex=True)

    # Panel 1: VaR vs actual loss
    axes[0].fill_between(var_s.index, 0, var_s,
                         alpha=0.12, color="#E65100")
    axes[0].plot(var_s.index, var_s, color="#E65100", linewidth=1.2,
                 label=f"EVT 99% VaR (POT/GPD, Irle p.223)")
    axes[0].plot(actual_loss.index, actual_loss, color="#90A4AE",
                 linewidth=0.6, alpha=0.7, label="Actual loss (−ΔV)")
    if exc_idx is not None and len(exc_idx) > 0:
        axes[0].scatter(exc_idx, actual_loss.loc[exc_idx],
                        color="#F44336", s=20, zorder=5,
                        label=f"Exceptions  N={bt.N}  ({bt.exception_rate:.2%})")
    axes[0].axhline(0, color="black", linewidth=0.5, linestyle="--")
    axes[0].set_title("EVT (POT) VaR 99% vs Actual Portfolio Loss  |  Irle Section 9",
                      fontsize=12, fontweight="bold")
    axes[0].set_ylabel("USD")
    axes[0].legend(fontsize=9)

    # Panel 2: Rolling threshold u
    axes[1].plot(results.index, results["threshold_u"],
                 color="#7B1FA2", linewidth=0.9,
                 label=f"Threshold u  (90th pct of rolling {WINDOW}-day losses)")
    axes[1].set_title("Rolling Threshold u — 90th Percentile of Losses in Window",
                      fontsize=12, fontweight="bold")
    axes[1].set_ylabel("u  (USD)")
    axes[1].legend(fontsize=9)

    # Panel 3: Rolling ξ (GPD shape parameter)
    xi = results["xi"]
    axes[2].plot(results.index, xi, color="#00796B", linewidth=0.9,
                 label="ξ  (GPD shape parameter)")
    axes[2].axhline(0, color="black", linewidth=0.8, linestyle="--",
                    label="ξ = 0  (Gumbel boundary)")
    axes[2].fill_between(results.index, 0, xi.clip(lower=0),
                         alpha=0.12, color="#00796B",
                         label="Heavy-tail region  ξ > 0")
    axes[2].set_title("Rolling ξ — GPD Shape Parameter  |  ξ > 0 → Pareto Heavy Tail (Irle p. 213)",
                      fontsize=12, fontweight="bold")
    axes[2].set_ylabel("ξ  (shape)")
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[2].legend(fontsize=9)

    for ax in axes:
        for s, e, lbl in crises:
            ax.axvspan(s, e, alpha=0.07, color="red")

    plt.tight_layout()
    path = os.path.join(OUTPUT_FIGS, "evt_02_var_and_xi.png")
    plt.savefig(path, dpi=150)
    plt.close("all")
    print(f"EVT results plot saved -> {path}")

# =============================================================================
# STEP 6 — SAVE RESULTS
# =============================================================================

def save_results(pnl, results, bt):
    """
    Save VaR time series and backtest detail table.
    Convention matches delta_normal.py.
    """
    results.to_csv(os.path.join(PROCESSED_DIR, "var_evt.csv"))
    print(f"VaR saved     -> {os.path.join(PROCESSED_DIR, 'var_evt.csv')}")

    common = pnl.index.intersection(results.index)
    loss_aligned = -pnl.reindex(common)
    pd.DataFrame({
        "VaR"         : results.loc[common, "VaR_EVT"].values,
        "actual_loss" : loss_aligned.values,
        "exception"   : (loss_aligned > results.loc[common, "VaR_EVT"]).astype(int).values,
        "xi"          : results.loc[common, "xi"].values,
        "threshold_u" : results.loc[common, "threshold_u"].values,
    }, index=common).to_csv(
        os.path.join(OUTPUT_TABLES, "backtest_evt.csv")
    )
    print(f"Backtest      -> {os.path.join(OUTPUT_TABLES, 'backtest_evt.csv')}")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    os.makedirs(OUTPUT_FIGS,   exist_ok=True)
    os.makedirs(OUTPUT_TABLES, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"EVT (POT) VaR  |  alpha={ALPHA:.0%}  |  window={WINDOW} days")
    print("Theory: Irle Section 9 / Pickands-Balkema-de Haan Theorem")
    print(f"{'='*60}")

    # Step 1: Load data
    pnl    = load_data()
    losses = -pnl

    # Step 2: Threshold diagnostics (full sample)
    plot_threshold_diagnostics(losses)

    # Step 3: Rolling EVT VaR
    results = compute_evt_var(pnl)

    # Step 4: Backtest
    bt = backtest_evt(pnl, results)

    # Step 5: Plots
    plot_evt_results(pnl, results, bt)

    # Step 6: Save
    save_results(pnl, results, bt)

    print(f"\n{'='*60}")
    print(f"EVT (POT) VaR complete!")
    print(f"  Exceptions  : {bt.N}  (expected {bt.expected_N:.1f}  at 99% VaR)")
    print(f"  Kupiec H0   : {'NOT rejected' if not bt.reject_uc else 'REJECTED'}  "
          f"(p={bt.pvalue_uc:.4f})")
    print(f"{'='*60}")
