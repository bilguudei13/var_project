# =============================================================================
# garch_evt.py
# 1-day 99% GARCH(1,1) + EVT Conditional VaR
#
# Theory references:
#
# [McNeil & Frey (2000), "Estimation of tail-related risk measures for
#  heteroscedastic financial time series", J. Empirical Finance 7, 271-300]
#  → canonical two-step conditional EVT approach
#
# TWO-STEP PROCEDURE:
#
# STEP A — GARCH(1,1) volatility filter (Irle Section 8.3, p. 172-186):
#   Model: X_t = σ_t · Z_t,  Z_t iid, E[Z_t]=0, Var[Z_t]=1
#   Variance: σ²_t = ω + α_1·X²_{t-1} + β·σ²_{t-1}   (Irle Eq. 17)
#   One-step forecast: σ̂²_{t+1} = ω + α_1·X²_t + β·σ̂²_t  (Irle Eq. 18)
#   Innovations: Student-t(ν) — captures fat tails in daily returns (Irle p. 179-180)
#   Stationarity: α_1 + β < 1  (Irle p. 176)
#
# STEP B — EVT on standardised residuals (Irle Section 9, p. 212-225):
#   Ẑ_t = X_t / σ̂_t   (approximately iid after GARCH filtering)
#   Fit GPD to lower tail of {Ẑ_t} using POT/GPD:
#     G_{ξ,σ}(x) = 1 - (1 + ξ·x/σ)^{-1/ξ}   (Irle p. 212, Definition 7)
#   POT quantile on residuals (dimensionless):
#     q_EVT(α) = u_z + (σ_z/ξ) · [(T_w/N_u · (1-α))^{-ξ} - 1]
#
# STEP C — Conditional VaR (McNeil & Frey 2000, Eq. 4):
#   VaR_{α,t+1} = V0 · σ̂_{t+1} · q_EVT(α)
#
# Note on arch scaling:
#   arch_model() is designed for %-returns and is ill-conditioned on raw
#   decimal returns (≈0.001). We pass r_p * 100 and divide cond_vol by 100:
#     cond_vol_decimal = res.conditional_volatility / 100
#   The final VaR is: V0 × cond_vol_decimal_{t+1} × q_EVT  [USD]
#
# Rolling window : 500 trading days
# GARCH fit      : full sample (parameters stable over long horizons)
# Confidence     : α = 99%
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
from scipy.stats import genpareto, t as student_t, probplot
from arch import arch_model

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from backtesting.backtest import run_backtest

# =============================================================================
# SETTINGS
# =============================================================================

WINDOW          = 500
ALPHA           = 0.99
THRESHOLD_Q     = 0.90        # 90th percentile of residuals → ~50 exceedances/window
MIN_EXCEEDANCES = 10
V0              = 1_000_000
WEIGHTS         = np.ones(4) / 4   # equal weights: EURUSD, GLD, IEF, SPY (4 assets)

PROCESSED_DIR = os.path.join("data", "processed")
OUTPUT_FIGS   = os.path.join("outputs", "figures")
OUTPUT_TABLES = os.path.join("outputs", "tables")

# =============================================================================
# STEP 1 — LOAD DATA
# =============================================================================

def load_data():
    """
    Load total portfolio P&L and derive total portfolio return for GARCH.

    GARCH is fit on total portfolio returns (pnl_total / V0) so that the
    conditional volatility captures all risk sources: linear positions,
    IRS (DV01 sensitivity), and ATM straddle (gamma/vega). Fitting on
    linear-only returns would underestimate VaR by ~40% when backtested
    against pnl_total (which includes non-linear P&L components).

    Total portfolio return: r_p_t = pnl_total_t / V0

    Returns
    -------
    r_p : pd.Series   total portfolio return (decimal, all instruments)
    pnl : pd.Series   dollar P&L -- pnl_total (for backtesting)
    """
    pnl = pd.read_csv(
        os.path.join(PROCESSED_DIR, "total_portfolio_pnl.csv"),
        index_col=0, parse_dates=True
    )["pnl_total"]

    # Derive total portfolio return from P&L: r_t = pnl_t / V0
    # This includes linear + IRS + straddle components.
    r_p = pnl / V0
    r_p.name = "portfolio_return_total"

    print(f"Loaded: {len(pnl)} days of total portfolio P&L")
    print(f"Date range : {pnl.index[0].date()} -> {pnl.index[-1].date()}")
    print(f"r_p (total): mean={r_p.mean():.6f}  std={r_p.std():.6f}")
    return r_p, pnl

# =============================================================================
# STEP 2 — GARCH(1,1) FULL-SAMPLE FIT
# =============================================================================

def fit_garch_full(r_p):
    """
    Fit GARCH(1,1) with Student-t innovations on full portfolio return series.
    (Irle Section 8.3, p. 172-186; McNeil & Frey 2000, Section 2)

    Implementation:
      - arch_model scales input ×100 internally for numerical stability
      - cond_vol returned in %-units → divide by 100 to get decimal σ̂_t
      - dist='t' fits ν (degrees of freedom) jointly with GARCH params
      - Stationarity check: α_1 + β < 1 (Irle p. 176)

    Standardised residuals:
      Ẑ_t = (X_t - μ̂) / σ̂_t   (should be approximately iid, Irle p. 175)

    Returns
    -------
    res         : arch ModelResult
    cond_vol    : pd.Series  σ̂_t in return (decimal) units
    z_residuals : pd.Series  standardised residuals Ẑ_t
    """
    print(f"\n{'='*60}")
    print(f"GARCH(1,1) fit  --  Irle Section 8.3")
    print(f"dist=Student-t | full sample | {len(r_p)} observations")

    garch = arch_model(r_p * 100, vol="Garch", p=1, q=1, dist="t", mean="Constant")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = garch.fit(disp="off")

    # Convert cond_vol from %-units back to decimal return units
    cond_vol = res.conditional_volatility / 100.0
    cond_vol.index = r_p.index

    # Standardised residuals: Ẑ_t = (r_p - μ̂) / σ̂_t
    r_p_aligned   = r_p.loc[cond_vol.index]
    mu_hat        = r_p_aligned.mean()
    z_residuals   = (r_p_aligned - mu_hat) / cond_vol
    z_residuals.name = "z_residuals"

    # Print parameter estimates
    params = res.params
    # arch parameter names differ by version; access robustly
    param_names = list(params.index)
    print(f"\nGARCH(1,1) parameters (Irle Eq. 17):")
    for nm, val in zip(param_names, params.values):
        print(f"  {nm:20s} = {val:.6f}")

    # Identify alpha and beta for stationarity check
    alpha1 = next((v for n, v in zip(param_names, params.values)
                   if "alpha" in n.lower() and "[1]" in n), None)
    beta1  = next((v for n, v in zip(param_names, params.values)
                   if "beta"  in n.lower() and "[1]" in n), None)
    if alpha1 is not None and beta1 is not None:
        ab = alpha1 + beta1
        flag = "OK  stationarity holds" if ab < 1 else "!!  UNIT ROOT WARNING"
        print(f"\n  alpha_1 + beta = {ab:.6f}  ->  {flag}  (Irle p. 176)")

    print(f"\nStandardised residuals:")
    print(f"  mean = {z_residuals.mean():.4f}  (~=0 expected)")
    print(f"  std  = {z_residuals.std():.4f}   (~=1 expected)")

    return res, cond_vol, z_residuals

# =============================================================================
# STEP 3 — GARCH DIAGNOSTICS (plots)
# =============================================================================

def plot_garch_diagnostics(r_p, cond_vol, z_residuals):
    """
    Two diagnostic panels:

    Panel 1: GARCH conditional volatility (annualised ×√252) over time.
             Demonstrates volatility clustering captured by GARCH (Irle p. 172-174).
             Crisis spikes (GFC, COVID) confirm model responsiveness.

    Panel 2: QQ-plot of standardised residuals vs Student-t distribution.
             Heavy tails in residuals justify EVT on Ẑ_t (Irle p. 179-180).
             If residuals were perfectly Student-t, GARCH+EVT would reduce to
             pure GARCH. The QQ-plot typically shows heavier tails than t-dist,
             confirming EVT adds value.
    """
    crises = [
        ("2008-09-15", "2009-03-09", "GFC 2008"),
        ("2020-02-19", "2020-03-23", "COVID 2020"),
        ("2022-01-01", "2022-10-01", "Rate Hikes 2022"),
    ]

    ann_vol = cond_vol * np.sqrt(252) * 100    # annualised vol in %
    z       = z_residuals.dropna().values

    # Fit Student-t to standardised residuals for QQ reference
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dof, loc_fit, scale_fit = student_t.fit(z, floc=0)
    z_norm = (z - loc_fit) / scale_fit

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: GARCH conditional vol
    axes[0].plot(ann_vol.index, ann_vol, color="#1976D2",
                 linewidth=0.8, label="σ̂_t annualised (%)")
    axes[0].set_title("GARCH(1,1) Conditional Volatility — Annualised\n"
                      "Volatility clustering captured  |  Irle p. 174",
                      fontsize=11, fontweight="bold")
    axes[0].set_ylabel("σ̂_t  (%/year)")
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for s, e, lbl in crises:
        axes[0].axvspan(s, e, alpha=0.10, color="red")
    # Label first crisis manually for legend
    axes[0].axvspan("2008-09-15", "2009-03-09", alpha=0.10, color="red",
                    label="Crisis periods")
    axes[0].legend(fontsize=8)

    # Panel 2: QQ-plot vs Student-t
    (osm, osr), (slope, intercept, r_val) = probplot(
        z_norm, dist=student_t, sparams=(dof,)
    )
    axes[1].scatter(osm, osr, color="#43A047", s=5, alpha=0.4,
                    label=f"Residuals  (n={len(z)})")
    line_x = np.array([min(osm), max(osm)])
    axes[1].plot(line_x, slope * line_x + intercept,
                 color="#C62828", linewidth=1.6,
                 label=f"Student-t(ν={dof:.1f}) reference line")
    axes[1].set_xlabel("Theoretical quantiles  [Student-t]")
    axes[1].set_ylabel("Sample quantiles  [Ẑ_t]")
    axes[1].set_title("QQ-Plot: Standardised Residuals vs Student-t\n"
                      "Tail deviations justify EVT on Ẑ_t  |  Irle p. 179-180",
                      fontsize=11, fontweight="bold")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(OUTPUT_FIGS, "garch_evt_01_diagnostics.png")
    plt.savefig(path, dpi=150)
    plt.close("all")
    print(f"GARCH diagnostics saved -> {path}")

# =============================================================================
# STEP 4 — POT ON RESIDUALS (helper) + ROLLING VaR
# =============================================================================

def _pot_var_residuals(z_w, threshold_q, alpha, T_w):
    """
    Apply POT/GPD to the lower tail of a window of standardised residuals.

    Lower-tail losses in residual space: z_losses = -z_w
    Threshold: u_z = quantile(z_losses, threshold_q)
    Fit GPD on exceedances above u_z (dimensionless).

    Returns
    -------
    q_evt  : float   EVT quantile of residuals (dimensionless)  [= VaR / (V0·σ̂)]
    xi     : float   GPD shape parameter (NaN if fallback used)
    sigma  : float   GPD scale parameter (NaN if fallback used)

    Reference: Irle p. 223-225 applied to Ẑ_t; McNeil & Frey (2000) Eq. 4
    """
    z_losses    = -z_w
    u_z         = np.quantile(z_losses, threshold_q)
    exceedances = z_losses[z_losses > u_z] - u_z
    N_u         = len(exceedances)

    if N_u < MIN_EXCEEDANCES:
        return np.quantile(z_losses, alpha), np.nan, np.nan

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            c, loc, sigma = genpareto.fit(exceedances, floc=0)
        xi    = c
        ratio = (T_w / N_u) * (1.0 - alpha)

        if abs(xi) < 1e-4:
            q_evt = u_z - sigma * np.log(ratio)
        else:
            q_evt = u_z + (sigma / xi) * (ratio ** (-xi) - 1.0)

        q_evt = max(q_evt, u_z)
        return q_evt, xi, sigma

    except Exception:
        return np.quantile(z_losses, alpha), np.nan, np.nan


def compute_garch_evt_var(pnl, cond_vol, z_residuals,
                          window=WINDOW, threshold_q=THRESHOLD_Q, alpha=ALPHA, V0=V0):
    """
    Rolling 500-day GARCH(1,1)+EVT conditional VaR (McNeil & Frey 2000).

    For each day t in [window, T):
      (A) σ̂_{t+1} = cond_vol[t]        one-step-ahead GARCH vol forecast
                                          (Irle Eq. 18: σ̂²_{t+1} = ω + α_1·X²_t + β·σ̂²_t
                                           → cond_vol[t] is already the t+1 forecast)
      (B) z_w = z_residuals[t-window:t]  window of standardised residuals
          q_EVT = POT quantile of lower tail of z_w  (dimensionless)
      (C) VaR_t = V0 × σ̂_{t+1} × q_EVT  (USD)

    Note: GARCH is fit once on the full sample. Full rolling re-estimation
    every 50 days (as in Irle p. 182) would improve accuracy but requires
    ~90 additional GARCH fits; documented here as a recommended extension.

    Returns
    -------
    pd.DataFrame  columns: VaR_GARCH_EVT, sigma_hat, q_EVT, xi
    """
    # Align cond_vol and z_residuals on shared dates
    common    = cond_vol.index.intersection(z_residuals.index)
    vol_vals  = cond_vol.loc[common].values
    z_vals    = z_residuals.loc[common].values
    dates_all = common
    n         = len(common)

    print(f"\n{'='*60}")
    print(f"GARCH(1,1)+EVT VaR  --  McNeil & Frey (2000)")
    print(f"k={window} | alpha={alpha:.0%} | threshold={threshold_q:.0%} on residuals")
    print(f"Computing conditional VaR for {n - window} days ...")

    records, dates = [], []
    n_fallbacks = 0

    for t in range(window, n):
        sigma_hat_t1 = vol_vals[t]               # σ̂_{t+1} in decimal return units
        z_w          = z_vals[t - window : t]    # residuals window

        q_evt, xi_t, sig_z = _pot_var_residuals(z_w, threshold_q, alpha, window)
        if np.isnan(xi_t):
            n_fallbacks += 1

        var_t = V0 * sigma_hat_t1 * q_evt        # VaR in USD (Eq. C above)

        records.append({
            "VaR_GARCH_EVT" : var_t,
            "sigma_hat"     : sigma_hat_t1,
            "q_EVT"         : q_evt,
            "xi"            : xi_t,
        })
        dates.append(dates_all[t])

    results = pd.DataFrame(records, index=dates)
    xi_ok   = results["xi"].dropna()

    print(f"\nRolling GARCH+EVT summary:")
    print(f"  Mean VaR     : ${results['VaR_GARCH_EVT'].mean():>12,.0f}")
    print(f"  Min  VaR     : ${results['VaR_GARCH_EVT'].min():>12,.0f}")
    print(f"  Max  VaR     : ${results['VaR_GARCH_EVT'].max():>12,.0f}")
    print(f"  Mean sigma_hat: {results['sigma_hat'].mean():.6f}  (decimal return units)")
    print(f"  Mean q_EVT   : {results['q_EVT'].mean():.4f}  (dimensionless residual quantile)")
    print(f"  Mean xi      :  {xi_ok.mean():.4f}  (>0 -> heavy tail)")
    print(f"  Fallback days: {n_fallbacks} / {n - window}")

    return results

# =============================================================================
# STEP 5 — BACKTEST
# =============================================================================

def backtest_garch_evt(pnl, results):
    """
    Kupiec + Christoffersen backtests via shared framework.
    (Irle Chapter 8, p. 183-185)
    """
    bt = run_backtest(
        pnl=pnl,
        var=results["VaR_GARCH_EVT"],
        confidence=ALPHA,
        method_name="GARCH+EVT",
    )
    print(bt)
    return bt

# =============================================================================
# STEP 6 — PLOTS
# =============================================================================

def plot_garch_evt_results(pnl, results, bt):
    """
    Two-panel figure:
      Panel 1: GARCH+EVT conditional VaR vs actual loss with exceptions
      Panel 2: Rolling σ̂_{t+1} (GARCH one-step vol) showing VaR = V0·σ̂·q_EVT

    The GARCH component makes VaR spike sharply during crises — unlike static
    EVT — illustrating the key advantage of the conditional approach.
    (McNeil & Frey 2000: GARCH+EVT dominates unconditional EVT in backtests)
    """
    crises = [
        ("2008-09-15", "2009-03-09", "GFC 2008"),
        ("2020-02-19", "2020-03-23", "COVID 2020"),
        ("2022-01-01", "2022-10-01", "Rate Hikes 2022"),
    ]

    var_s       = results["VaR_GARCH_EVT"]
    actual_loss = -pnl.reindex(results.index)
    exc_idx     = bt.exceptions_index

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Panel 1: VaR vs actual loss
    axes[0].fill_between(var_s.index, 0, var_s,
                         alpha=0.12, color="#0288D1")
    axes[0].plot(var_s.index, var_s, color="#0288D1", linewidth=1.2,
                 label="GARCH+EVT 99% VaR  (McNeil & Frey 2000)")
    axes[0].plot(actual_loss.index, actual_loss, color="#90A4AE",
                 linewidth=0.6, alpha=0.7, label="Actual loss (−ΔV)")
    if exc_idx is not None and len(exc_idx) > 0:
        axes[0].scatter(exc_idx, actual_loss.loc[exc_idx],
                        color="#F44336", s=20, zorder=5,
                        label=f"Exceptions  N={bt.N}  ({bt.exception_rate:.2%})")
    axes[0].axhline(0, color="black", linewidth=0.5, linestyle="--")
    axes[0].set_title("GARCH(1,1)+EVT Conditional VaR 99% vs Actual Portfolio Loss\n"
                      "VaR_t = V0 × σ̂_{t+1} × q_EVT(α)   |   McNeil & Frey (2000) Eq. 4",
                      fontsize=12, fontweight="bold")
    axes[0].set_ylabel("USD")
    axes[0].legend(fontsize=9)

    # Panel 2: GARCH one-step vol σ̂_{t+1}
    vol_pct = results["sigma_hat"] * np.sqrt(252) * 100   # annualised %
    axes[1].plot(results.index, vol_pct, color="#7B1FA2",
                 linewidth=0.8, label="σ̂_{t+1} annualised (%)")
    axes[1].set_title("GARCH One-Step-Ahead Conditional Vol σ̂_{t+1}  |  "
                      "Irle Eq. 18: σ̂²_{t+1} = ω + α_1·X²_t + β·σ̂²_t",
                      fontsize=12, fontweight="bold")
    axes[1].set_ylabel("σ̂_{t+1}  (%/year)")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[1].legend(fontsize=9)

    for ax in axes:
        for s, e, lbl in crises:
            ax.axvspan(s, e, alpha=0.07, color="red")

    plt.tight_layout()
    path = os.path.join(OUTPUT_FIGS, "garch_evt_02_var_results.png")
    plt.savefig(path, dpi=150)
    plt.close("all")
    print(f"GARCH+EVT results plot saved -> {path}")

# =============================================================================
# STEP 7 — SAVE RESULTS
# =============================================================================

def save_results(pnl, results, bt):
    """
    Save VaR time series and backtest detail table.
    Convention matches delta_normal.py and evt.py.
    """
    results.to_csv(os.path.join(PROCESSED_DIR, "var_garch_evt.csv"))
    print(f"VaR saved     -> {os.path.join(PROCESSED_DIR, 'var_garch_evt.csv')}")

    common = pnl.index.intersection(results.index)
    loss_aligned = -pnl.reindex(common)
    pd.DataFrame({
        "VaR"         : results.loc[common, "VaR_GARCH_EVT"].values,
        "actual_loss" : loss_aligned.values,
        "exception"   : (loss_aligned > results.loc[common, "VaR_GARCH_EVT"]).astype(int).values,
        "sigma_hat"   : results.loc[common, "sigma_hat"].values,
        "q_EVT"       : results.loc[common, "q_EVT"].values,
        "xi"          : results.loc[common, "xi"].values,
    }, index=common).to_csv(
        os.path.join(OUTPUT_TABLES, "backtest_garch_evt.csv")
    )
    print(f"Backtest      -> {os.path.join(OUTPUT_TABLES, 'backtest_garch_evt.csv')}")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    os.makedirs(OUTPUT_FIGS,   exist_ok=True)
    os.makedirs(OUTPUT_TABLES, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"GARCH(1,1)+EVT VaR  |  alpha={ALPHA:.0%}  |  window={WINDOW} days")
    print(f"Theory: McNeil & Frey (2000) / Irle Section 8.3 + Section 9")
    print(f"{'='*60}")

    # Step 1: Load data
    r_p, pnl = load_data()

    # Step 2: Fit GARCH(1,1) on full sample
    garch_res, cond_vol, z_residuals = fit_garch_full(r_p)

    # Step 3: GARCH diagnostics
    plot_garch_diagnostics(r_p, cond_vol, z_residuals)

    # Step 4: Rolling GARCH+EVT VaR
    results = compute_garch_evt_var(pnl, cond_vol, z_residuals)

    # Step 5: Backtest
    bt = backtest_garch_evt(pnl, results)

    # Step 6: VaR results plot
    plot_garch_evt_results(pnl, results, bt)

    # Step 7: Save
    save_results(pnl, results, bt)

    print(f"\n{'='*60}")
    print(f"GARCH(1,1)+EVT VaR complete!")
    print(f"  Exceptions  : {bt.N}  (expected {bt.expected_N:.1f}  at 99% VaR)")
    print(f"  Kupiec H0   : {'NOT rejected' if not bt.reject_uc else 'REJECTED'}  "
          f"(p={bt.pvalue_uc:.4f})")
    print(f"{'='*60}")
