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
#   F_u(x) = P(L - u <= x | L > u) converges to the GPD G_{xi,sigma}(x).
#
# [GPD definition (Irle p. 212, Definition 7)]:
#   G_{xi,sigma}(x) = 1 - (1 + xi*x/sigma)^{-1/xi}   xi != 0
#                   = 1 - exp(-x/sigma)                 xi = 0  (exponential GPD limit; Gumbel maximum-domain case)
#   Parameters: shape xi (tail index), scale sigma > 0
#   xi > 0 -> heavy tail (Pareto), xi = 0 -> light tail, xi < 0 -> bounded tail
#
# [POT VaR formula (Irle p. 223-225)]:
#   VaR_alpha = u + (sigma/xi) * [(T_w/N_u * (1-alpha))^{-xi} - 1]   xi != 0
#   VaR_alpha = u - sigma * log((T_w/N_u) * (1-alpha))                xi -> 0  (exponential GPD limit)
#   Where: T_w = rolling window size, N_u = exceedances above u
#   Note: ratio = (T_w/N_u) * (1-alpha) is small (typically ~0.1) when alpha
#   is deep in the tail, so log(ratio) < 0 and the Gumbel VaR sits above u.
#
# [McNeil & Frey (2000), "Estimation of tail-related risk measures for
#  heteroscedastic financial time series", J. Empirical Finance 7, 271-300]
#
# =============================================================================
# MODEL SCOPE AND LIMITATIONS
# =============================================================================
#
# This is an UNCONDITIONAL rolling EVT/POT VaR model. POT/GPD is fitted directly
# to dollar losses (-pnl_total) without volatility filtering. Two consequences
# follow:
#
#   1. The i.i.d. assumption (Pickands-Balkema-de Haan, Irle p. 213) is applied
#      to raw daily losses, which financial losses are not — they exhibit
#      well-documented volatility clustering. The POT approximation is still used
#      as an asymptotic tail approximation, but the formal i.i.d. justification is
#      weakened for raw financial losses. The rolling 500-day window will adapt
#      slowly to regime changes.
#
#   2. Exception clustering is expected. The Christoffersen independence test
#      (LR_IND) typically rejects for unconditional EVT models because
#      exceptions concentrate during high-volatility periods. This is not a
#      coding error; it is an expected statistical limitation of the unconditional
#      EVT specification. The appendix (assumption A.4) documents this.
#
# For CONDITIONAL EVT — POT applied to GARCH-standardised residuals, making the
# i.i.d. assumption more plausible after volatility filtering (McNeil & Frey 2000)
# — see garch_evt.py. In the current portfolio run, that model does not reject both
# Kupiec and Christoffersen; rerun results should be checked if data or assumptions
# change. evt.py is retained as a benchmark for the unconditional case to
# quantify the value-add of GARCH pre-filtering.
#
#Backtest expectation:
#Unconditional EVT may produce a reasonable VaR level, but in the current
# portfolio run both Kupiec and Christoffersen reject: the model has too many
#  exceptions and those exceptions are clustered in stress periods. This is
# consistent with the limitation of applying EVT directly to raw losses without
#  volatility filtering.
# =============================================================================
#
# Rolling window : 500 trading days
# Confidence     : alpha = 99%
# Threshold      : 90th percentile of losses in window  (~50 exceedances)
#
# =============================================================================
# METHODOLOGY AND IMPLEMENTATION NOTES
# =============================================================================
#
# Threshold selection:
#   threshold_q = 0.90 yields approximately 50 exceedances per 500-day window —
#   a pragmatic bias-variance compromise: enough exceedances for estimation while
#   keeping the threshold in the tail.
#   Robustness across {0.85, 0.90, 0.925, 0.95} is reported in
#   outputs/tables/evt_threshold_sensitivity.csv.
#
# GPD fit (R-equivalent note):
#   scipy.stats.genpareto.fit(exceedances, floc=0) fixes the GPD location at
#   zero and fits shape (xi) and scale (sigma) to excess losses above threshold_u.
#   This is conceptually analogous to fitting excess losses with ismev::gpd.fit or
#   evd::fpot in R, with the threshold/location fixed at zero for the excess series.
#   np.quantile(...) corresponds to R's quantile(..., type=7).
#   pandas Series/DataFrame indexed by dates are analogous to zoo/xts objects in R.
#
# xi clamping:
#   xi is clamped to [-0.5, 1.0] as a defensive VaR guard, not a constrained MLE.
#   At xi=1.0 the GPD mean is still not finite, but VaR (a quantile) remains
#   well-defined for any alpha < 1.  xi_raw stores the unconstrained MLE value;
#   xi_clamped flags when the guard triggers.  ks_pvalue is set to NaN when xi
#   is clamped because (xi, sigma) no longer represents a jointly fitted
#   distribution.
#
# VaR columns (diagnostic transparency):
#   VaR_EVT         — regulatory/policy output (non-negative; empirical fallback
#                     applied if GPD fit failed or var_raw < threshold_u).
#   VaR_EVT_raw     — In "none" paths: POT quantile with clamped xi, before the
#                     non-negativity floor.
#                     In "q_below_threshold": the degenerate GPD quantile that fell
#                     below threshold_u, stored intentionally for diagnostics (var_final
#                     uses the empirical fallback instead).
#                     In "gpd_fit_failed" (non-finite GPD quantile): NaN (the non-finite
#                     value is not stored; var_final uses the empirical fallback).
#                     In other fallback paths ("few_exceedances", "alpha_below_threshold",
#                     "gpd_fit_failed" invalid sigma / exception): empirical quantile
#                     (same as var_final).
#   VaR_EVT_mle_raw — POT quantile with unconstrained xi_raw. Equals VaR_EVT_raw
#                     when xi_clamped is False; NaN in most fallback paths.
#
# fallback_type column:
#   "none"                  — GPD MLE succeeded; VaR_EVT = VaR_EVT_raw.
#   "few_exceedances"       — N_u < MIN_EXCEEDANCES; empirical quantile used.
#   "gpd_fit_failed"        — scipy raised an exception, returned invalid sigma,
#                             or var_raw was not finite.
#   "q_below_threshold"     — POT formula returned q < threshold_u despite alpha > f_u_hat;
#                             degenerate fit (xi clamp, near-zero sigma, or numerical issue).
#   "alpha_below_threshold" — alpha <= f_u_hat; target quantile within empirical range.
#
# Expected Shortfall / ES columns:
#   ES_EVT         — 99% Expected Shortfall / Tail VaR policy output.
#                    For successful GPD fits with xi < 1:
#                      ES_alpha = (VaR_alpha + sigma_gpd - xi * threshold_u) / (1 - xi)
#                    Gumbel limit xi ~= 0:
#                      ES_alpha = VaR_alpha + sigma_gpd
#                    In fallback paths, empirical ES is used.
#                    If xi >= 1, the GPD mean / ES is undefined, so empirical ES is used
#                    and flagged via ES_fallback_type.
#   ES_EVT_raw     — GPD ES computed from clamped xi before ES guardrails.
#   ES_EVT_mle_raw — GPD ES computed from unconstrained xi_raw where xi_raw < 1.
#                    NaN when xi_raw >= 1 or unavailable.
#
#   VaR answers "what loss threshold is exceeded 1% of the time?"
#   ES answers "conditional on exceeding VaR, how large is the average loss?"
#   ES is therefore a tail-severity diagnostic and complements the exception-count
#   backtests.
#
# Crisis-period backtesting:
#   The full-sample Kupiec / Christoffersen tests are supplemented by subperiod
#   diagnostics for GFC 2008, COVID 2020, Rate Hikes 2022, and non-crisis periods.
#   These are descriptive because crisis windows are short and formal test power
#   may be low.
#
# KS goodness-of-fit caveat:
#   The KS p-value (ks_pvalue) is diagnostic only. Because xi and sigma are estimated
#   from the same exceedances being tested, the usual KS p-value is anti-conservative
#   and should not be read as a formal acceptance test. It is useful as a relative
#   quality indicator across windows.
#
# Parameter stability diagnostics (C6-equivalent, not a formal Nyblom test):
#   run_evt_parameter_stability_diagnostics() tracks the rolling path of xi,
#   sigma_gpd, threshold_u, N_u, f_u_hat, VaR, fallback rates, clamp rates,
#   and KS poor-fit counts across all forecast windows.
#
# C5 (Engle-Ng sign-bias) — not applicable:
#   evt.py is an unconditional POT model on dollar losses. There is no conditional
#   variance equation and no standardised residual process, so the sign-bias
#   regression z_t^2 ~ c0 + c1*I(z_{t-1}<0) + ... is meaningless here.
#   C5 is implemented in garch_evt.py only.
# =============================================================================

import argparse
import os
import sys
import warnings
if hasattr(sys.stdout, "reconfigure"):   # guard for non-reconfigurable stdout
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive: save to disk, no window
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import genpareto, kstest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from backtesting.backtest import run_backtest

# =============================================================================
# SETTINGS
# =============================================================================

WINDOW          = 500
ALPHA           = 0.99
THRESHOLD_Q     = 0.90        # 90th percentile -> ~50 exceedances per 500-day window
MIN_EXCEEDANCES = 10          # minimum to attempt GPD MLE; else empirical fallback
V0              = 1_000_000

# =============================================================================
# SENSITIVITY ANALYSIS RATIONALE  (diagnostic only; does not alter production)
# =============================================================================
#
# THRESHOLD_Q is the empirical quantile used to set the GPD threshold u within
# each rolling window.  With WINDOW=500 and THRESHOLD_Q=0.90, the model uses
# approximately the worst 10% of losses, yielding around 50 exceedances for the
# GPD tail fit — a pragmatic bias-variance balance (Irle p. 215; McNeil &
# Frey 2000).
#
# Approximate exceedances by (THRESHOLD_Q, WINDOW) combination
# (computed as WINDOW * (1 - THRESHOLD_Q)):
#   THRESHOLD_Q \ WINDOW  |  250  |  500  |  750
#   ----------------------+-------+-------+------
#            0.85         |  ~37  |  ~75  | ~112
#            0.90         |  ~25  |  ~50  |  ~75   <- production
#            0.925        |  ~19  |  ~37  |  ~56
#            0.95         |  ~12  |  ~25  |  ~37
#
# Lower thresholds include more observations but risk contaminating the tail
# sample with non-extreme losses, biasing the GPD shape estimate downward.
# Higher thresholds concentrate the fit on the extreme tail but increase GPD
# estimation variance and may trigger fallback paths when N_u < MIN_EXCEEDANCES.
#
# Shorter rolling windows (250 days) adapt faster to regime shifts such as the
# GFC 2008 or COVID 2020 drawdowns, but produce noisier GPD parameter estimates.
# Longer windows (750 days) yield more stable estimates but are slower to reflect
# post-crisis normalisation.  WINDOW=500 is the production choice.
#
# The sensitivity sweep (--sensitivity flag) tests all twelve combinations of
# {0.85, 0.90, 0.925, 0.95} x {250, 500, 750} and saves diagnostics to
# outputs/tables/evt_threshold_window_sensitivity.csv.
# The window grid {250, 500, 750} is consistent with the group-wide sensitivity
# convention.  All sensitivity results are diagnostic only; they do not change
# any production EVT output.
# =============================================================================

PROCESSED_DIR = os.path.join("data", "processed")
OUTPUT_FIGS   = os.path.join("outputs", "figures")
OUTPUT_TABLES = os.path.join("outputs", "tables")

# =============================================================================
# UTILITIES
# =============================================================================

def _assert_results_aligned_to_pnl(pnl, results, context):
    """
    Date alignment guard shared by all downstream consumers of the rolling VaR series.

    Raises ValueError if any date in results.index is missing from pnl, or if
    NaN losses appear after reindex. Returns the aligned loss series (loss = -pnl)
    so callers do not need to recompute it.

    pandas Series indexed by dates are analogous to xts objects in R.
    """
    if not results.index.isin(pnl.index).all():
        missing = results.index.difference(pnl.index)
        raise ValueError(
            f"{context}: pnl missing {len(missing)} dates from results.index "
            f"(first 5: {missing[:5].tolist()})"
        )
    losses = -pnl.reindex(results.index)
    if losses.isna().any():
        raise ValueError(
            f"{context}: {losses.isna().sum()} NaN losses after pnl/results alignment"
        )
    return losses


def _empirical_var(losses_w, alpha):
    """
    Empirical VaR fallback at confidence alpha.

    Filters to finite values before computing the quantile. Returns np.nan if
    no finite values are present; otherwise returns a non-negative float so
    fallback paths obey the same non-negativity policy as the GPD VaR path.
    """
    losses_w = np.asarray(losses_w, dtype=float)
    finite = losses_w[np.isfinite(losses_w)]
    if len(finite) == 0:
        return np.nan
    return max(float(np.quantile(finite, alpha)), 0.0)


def _empirical_es(losses_w, alpha):
    """
    Empirical Expected Shortfall / Tail VaR at confidence alpha.

    ES_alpha = E[L | L >= VaR_alpha] estimated from the rolling window.
    Used as a fallback when GPD fitting fails, when the POT quantile is
    degenerate, or when xi >= 1 makes the GPD mean / ES undefined.

    Returns a finite non-negative float.
    """
    q = np.quantile(losses_w, alpha)
    tail = losses_w[losses_w >= q]
    if len(tail) == 0:
        return max(float(q), 0.0)
    return max(float(tail.mean()), float(q), 0.0)


def _validate_evt_results(results):
    """
    Post-computation invariant checks on the rolling EVT VaR DataFrame.

    Raises ValueError on violations. These are implementation-correctness checks,
    not statistical diagnostics — they should always pass under correct execution.
    A failure here indicates a bug, not a model limitation.

    Checks:
      1. VaR_EVT non-negative everywhere.
      2. VaR_EVT below 10x the maximum rolling threshold (plausibility bound).
      3. Rows tagged fallback_type='none' have VaR_EVT_raw >= threshold_u (POT contract).
      4. Post-clamp xi in [-0.5, 1.0] wherever non-NaN.
      5. sigma_gpd > 0 wherever non-NaN.
      6. f_u_hat in (0, 1) wherever non-NaN.
      7. ES_EVT finite everywhere.
      8. ES_EVT non-negative everywhere.
      9. ES_EVT >= VaR_EVT everywhere, allowing tiny numerical tolerance.
      10. Rows with fallback_type='none' and ES_fallback_type='none' should have
          ES_EVT_raw >= VaR_EVT_raw.
      11. ES_fallback_type values must be one of the recognised strings.
    """
    # 1. VaR_EVT non-negative everywhere
    if (results["VaR_EVT"] < 0).any():
        raise ValueError(
            f"VaR_EVT non-negativity violated: "
            f"{int((results['VaR_EVT'] < 0).sum())} negative values"
        )

    # 2. VaR_EVT not absurdly large: bound at 10x the largest rolling threshold
    bound = 10.0 * results["threshold_u"].max()
    if (results["VaR_EVT"] > bound).any():
        n_bad = int((results["VaR_EVT"] > bound).sum())
        raise ValueError(
            f"VaR_EVT exceeds plausibility bound (10x max threshold = "
            f"${bound:,.0f}) on {n_bad} days"
        )

    # 3. POT contract: rows tagged "none" must have VaR_EVT_raw >= threshold_u
    clean = results[results["fallback_type"] == "none"]
    if not clean.empty and (clean["VaR_EVT_raw"] < clean["threshold_u"]).any():
        raise ValueError(
            "Contract violation: rows tagged fallback_type='none' have "
            "VaR_EVT_raw < threshold_u — error guard not applied correctly"
        )

    # 4. xi clamp invariant: post-clamp xi in [-0.5, 1.0] wherever non-NaN
    xi_ok = results["xi"].dropna()
    if not xi_ok.empty:
        if (xi_ok < -0.5 - 1e-10).any() or (xi_ok > 1.0 + 1e-10).any():
            raise ValueError(
                f"xi clamp violation: xi range = [{xi_ok.min():.4f}, "
                f"{xi_ok.max():.4f}], expected within [-0.5, 1.0]"
            )

    # 5. sigma_gpd > 0 where non-NaN
    sigma_ok = results["sigma_gpd"].dropna()
    if not sigma_ok.empty and (sigma_ok <= 0).any():
        n_bad = int((sigma_ok <= 0).sum())
        raise ValueError(
            f"sigma_gpd non-positivity violated: {n_bad} rows have sigma_gpd <= 0"
        )

    # 6. f_u_hat in (0, 1) where non-NaN
    fu = results["f_u_hat"].dropna()
    if not fu.empty and ((fu <= 0) | (fu >= 1)).any():
        n_bad = int(((fu <= 0) | (fu >= 1)).sum())
        raise ValueError(f"f_u_hat out of (0, 1): {n_bad} rows")

    # 7. ES_EVT finite everywhere
    if not np.isfinite(results["ES_EVT"].values).all():
        n_bad = int((~np.isfinite(results["ES_EVT"].values)).sum())
        raise ValueError(f"ES_EVT non-finite: {n_bad} rows have Inf or NaN ES_EVT")

    # 8. ES_EVT non-negative everywhere
    if (results["ES_EVT"] < 0).any():
        n_bad = int((results["ES_EVT"] < 0).sum())
        raise ValueError(f"ES_EVT non-negativity violated: {n_bad} negative values")

    # 9. ES_EVT >= VaR_EVT everywhere (allow tiny numerical tolerance)
    tol = 1e-6
    if (results["ES_EVT"] < results["VaR_EVT"] - tol).any():
        n_bad = int((results["ES_EVT"] < results["VaR_EVT"] - tol).sum())
        raise ValueError(
            f"ES_EVT < VaR_EVT violated on {n_bad} rows — ES must be >= VaR"
        )

    # 10. Rows tagged both 'none' must have ES_EVT_raw >= VaR_EVT_raw
    both_clean = results[
        (results["fallback_type"] == "none") &
        (results["ES_fallback_type"] == "none")
    ]
    if not both_clean.empty:
        es_raw_vals  = both_clean["ES_EVT_raw"].dropna()
        var_raw_vals = both_clean.loc[es_raw_vals.index, "VaR_EVT_raw"]
        if (es_raw_vals < var_raw_vals - tol).any():
            raise ValueError(
                "Contract violation: rows with fallback_type='none' and "
                "ES_fallback_type='none' have ES_EVT_raw < VaR_EVT_raw"
            )

    # 11. ES_fallback_type must be one of the recognised strings
    VALID_ES_FALLBACK_TYPES = {
        "none",
        "empirical_fallback",
        "xi_ge_1_gpd_es_undefined",
        "es_invalid_empirical_fallback",
        "alpha_below_threshold",
        "few_exceedances",
        "gpd_fit_failed",
        "q_below_threshold",
    }
    unexpected_es = set(results["ES_fallback_type"].unique()) - VALID_ES_FALLBACK_TYPES
    if unexpected_es:
        raise ValueError(f"Unexpected ES_fallback_type values: {unexpected_es}")

    return True

# =============================================================================
# STEP 1 — LOAD DATA
# =============================================================================

def load_data():
    """
    Load total portfolio P&L from total_portfolio_pnl.csv.

    EVT is fitted directly to dollar losses L_t = -pnl_total_t. Using pnl_total
    ensures all instruments (linear, IRS, straddle) are included. The model is
    unconditional: no volatility filtering is applied before fitting GPD.

    pandas Series indexed by dates are analogous to zoo/xts in R.
    """
    pnl = pd.read_csv(
        os.path.join(PROCESSED_DIR, "total_portfolio_pnl.csv"),
        index_col=0, parse_dates=True
    )["pnl_total"]
    pnl.name = "pnl"

    # Data quality checks
    if not pnl.index.is_monotonic_increasing:
        raise ValueError("pnl index must be sorted ascending")
    if not pnl.index.is_unique:
        raise ValueError("pnl index contains duplicate dates")
    if pnl.isna().any():
        raise ValueError(f"pnl contains {pnl.isna().sum()} NaN values")
    # Catch Inf/-Inf that the NaN check misses
    if not np.isfinite(pnl.values).all():
        n_bad = int((~np.isfinite(pnl.values)).sum())
        raise ValueError(f"pnl contains {n_bad} non-finite values (Inf or NaN)")

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
        A linear region starting around the chosen threshold supports the GPD tail approximation.

    (b) Hill Estimator Plot:
        xi_Hill(k) = (1/k) * sum_{i=1}^{k} [log X_{(n-i+1)} - log X_{(n-k)}]
        where X_{(1)} <= ... <= X_{(n)} are order statistics of STRICTLY POSITIVE
        losses (log requires x > 0; profits/zero observations are excluded).
        (Irle Section 9; Hill 1975)
        Stability of xi_Hill in k suggests heavy-tail behaviour.

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

    # Chosen threshold (computed before Hill block so k_chosen can reference it)
    u_chosen = np.quantile(loss_vals, THRESHOLD_Q)

    # (b) Hill estimator — filter to STRICTLY POSITIVE losses before log()
    # Guard for sparse data; k_chosen counts positive losses strictly above u_chosen
    pos_losses = loss_vals[loss_vals > 0]
    if len(pos_losses) < 20:
        warnings.warn(
            f"Only {len(pos_losses)} positive losses; Hill estimator unstable. "
            f"Skipping Hill plot."
        )
        hill   = np.array([])
        k_vals = np.array([])
    else:
        x_desc = np.sort(pos_losses)[::-1]
        max_k  = min(300, len(x_desc) - 1)
        k_vals = np.arange(1, max_k + 1)
        hill   = np.array([
            np.mean(np.log(x_desc[:k]) - np.log(x_desc[k]))
            for k in k_vals
        ])
    k_chosen = int((pos_losses > u_chosen).sum())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel (a): MEP
    mask = ~np.isnan(mean_excess)
    axes[0].plot(u_grid[mask], mean_excess[mask],
                 color="#2196F3", linewidth=1.4, label="Mean excess e(u)")
    axes[0].axvline(u_chosen, color="#F44336", linestyle="--", linewidth=1.6,
                    label=f"Chosen u = ${u_chosen:,.0f}  ({THRESHOLD_Q:.0%} quantile)")
    axes[0].set_xlabel("Threshold u  (USD)", fontsize=10)
    axes[0].set_ylabel("E[L - u | L > u]  (USD)", fontsize=10)
    axes[0].set_title("Mean Excess Plot (MEP)\n"
                      "Linear region above u supports GPD threshold choice (Irle p. 215)",
                      fontsize=11, fontweight="bold")
    axes[0].legend(fontsize=9)

    # Panel (b): Hill estimator (strictly positive losses only)
    if len(hill) > 0:   # skip plot when data is too sparse
        axes[1].plot(k_vals, hill, color="#4CAF50", linewidth=1.2,
                     label="xi_Hill(k)  [positive losses only]")
        axes[1].axvline(k_chosen, color="#F44336", linestyle="--", linewidth=1.6,
                        label=f"k at 90th pct ~= {k_chosen}")
        axes[1].axhline(0, color="black", linewidth=0.7, linestyle=":")
    else:
        axes[1].text(0.5, 0.5, "Hill estimator skipped\n(< 20 positive losses)",
                     ha="center", va="center", transform=axes[1].transAxes, fontsize=11)
    axes[1].set_xlabel("k  (top-k order statistics of positive losses)", fontsize=10)
    axes[1].set_ylabel("Hill estimator xi_Hill(k)", fontsize=10)
    axes[1].set_title("Hill Estimator Plot  [strictly positive losses]\n"
                      "xi > 0 -> heavy tail (Pareto-type) -- Irle Section 9",
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
    Compute single-period POT VaR via GPD fit (Irle p. 223-225).

    Algorithm:
      1. Structural precondition: alpha > threshold_q required (raises ValueError
         otherwise). The data-dependent check alpha > f_u_hat occurs in step 3;
         that check governs whether POT extrapolation is actually needed.
      2. threshold_u = np.quantile(losses_w, threshold_q)
         In R: quantile(losses_w, threshold_q)
      3. exceedances = losses_w[losses_w > threshold_u] - threshold_u
         f_u_hat = 1 - N_u / T_w   (empirical CDF at threshold_u)
         Early exit if alpha <= f_u_hat (target quantile within empirical range).
      4. Fit GPD with floc=0:
            scipy.stats.genpareto.fit(exceedances, floc=0)
         returns (shape=xi, loc=0, scale=sigma).
         In R: analogous to ismev::gpd.fit(exceedances, threshold=0).
      5. xi_raw = unconstrained MLE. Clamp xi to [-0.5, 1.0] as a defensive
         VaR guard (not a constrained MLE).
      6. ratio = (T_w / N_u) * (1 - alpha)
            xi != 0: VaR = threshold_u + (sigma/xi) * (ratio^{-xi} - 1)
            xi -> 0: VaR = threshold_u - sigma * log(ratio)   (exponential GPD limit)
      7. Error guard: if var_raw < threshold_u despite alpha > f_u_hat, the GPD
         fit is degenerate (xi clamp, near-zero sigma, or numerical issue).
         Fall back to empirical quantile, tag "q_below_threshold".
      8. KS goodness-of-fit on exceedances vs fitted GPD. ks_pvalue is set to NaN
         when xi is clamped — (xi, sigma) is not a jointly fitted distribution
         when xi was overwritten by the guardrail.

    Returns
    -------
    dict with keys:
      var_final     : float  regulatory output (non-negative; empirical fallback
                             if fit failed or degenerate). Used by backtest and
                             capital reporting.
      var_raw       : float  In "none" paths: POT quantile with clamped xi, before
                             the non-negativity floor. In "q_below_threshold": the
                             raw GPD quantile that fell below u (var_final uses the
                             empirical fallback instead). In other fallback paths:
                             the empirical quantile (same as var_final).
      var_mle_raw   : float  POT quantile with the unconstrained xi_raw (pre-clamp).
                             Equals var_raw when xi_clamped is False; differs only
                             when the guardrail was active — used to quantify the
                             clamp impact. NaN in most non-"none" fallback paths.
      xi            : float  GPD shape (post-clamp; NaN in fallback paths)
      xi_raw        : float  GPD shape before clamping (NaN in fallback paths)
      xi_clamped    : bool   True when guardrail was triggered
      sigma_gpd     : float  GPD scale parameter. Non-NaN whenever the GPD fit
                             succeeded (including "q_below_threshold" and the
                             "gpd_fit_failed" path for non-finite var_raw, where
                             sigma was valid but VaR could not be computed).
                             NaN when no valid GPD scale is available (fit never
                             reached sigma, or sigma itself was invalid/non-finite).
      N_u           : int    exceedances above threshold_u in this window
      f_u_hat       : float  empirical CDF at threshold_u = 1 - N_u/T_w
      threshold_u   : float  threshold_u value used
      ks_pvalue     : float  KS p-value for GPD fit (NaN in fallback paths or when
                             xi_clamped is True)
      fallback_type : str    "none" | "few_exceedances" | "gpd_fit_failed" |
                             "q_below_threshold" | "alpha_below_threshold"
      es_final      : float  99% ES policy output (non-negative; empirical fallback
                             if GPD ES is undefined or invalid). Always finite.
      es_raw        : float  GPD ES computed from clamped xi before ES guardrails.
                             NaN in fallback paths (no meaningful GPD fit available).
      es_mle_raw    : float  GPD ES from unconstrained xi_raw; NaN when xi_raw >= 1
                             or when xi_raw is unavailable / produces non-finite result.
      es_fallback_type : str reason ES fell back to empirical; "none" when GPD ES used.
                             When xi >= 1 the GPD ES is theoretically undefined
                             (GPD mean does not exist), so empirical ES is used and
                             tagged "xi_ge_1_gpd_es_undefined". VaR is unaffected.
    """
    # POT formula requires alpha > threshold_q
    if alpha <= threshold_q:
        raise ValueError(
            f"POT VaR requires alpha > threshold_q; got alpha={alpha}, "
            f"threshold_q={threshold_q}"
        )

    u   = np.quantile(losses_w, threshold_q)
    exceedances = losses_w[losses_w > u] - u
    N_u = len(exceedances)
    f_u_hat = 1.0 - N_u / T_w   # empirical CDF at threshold u

    # If alpha <= f_u_hat the target quantile is within the empirical range;
    # POT extrapolation is unnecessary and unreliable.
    if alpha <= f_u_hat:
        fb = _empirical_var(losses_w, alpha)
        empirical_es = _empirical_es(losses_w, alpha)
        return {
            "var_final": fb, "var_raw": fb, "var_mle_raw": np.nan,
            "xi": np.nan, "xi_raw": np.nan, "xi_clamped": False,
            "sigma_gpd": np.nan,
            "N_u": N_u, "f_u_hat": f_u_hat, "threshold_u": u,
            "ks_pvalue": np.nan,
            "fallback_type": "alpha_below_threshold",
            "es_final": empirical_es, "es_raw": np.nan, "es_mle_raw": np.nan,
            "es_fallback_type": "alpha_below_threshold",
        }

    if N_u < MIN_EXCEEDANCES:
        fb = _empirical_var(losses_w, alpha)
        empirical_es = _empirical_es(losses_w, alpha)
        return {
            "var_final": fb, "var_raw": fb, "var_mle_raw": np.nan,
            "xi": np.nan, "xi_raw": np.nan, "xi_clamped": False,
            "sigma_gpd": np.nan,
            "N_u": N_u, "f_u_hat": f_u_hat, "threshold_u": u,
            "ks_pvalue": np.nan,
            "fallback_type": "few_exceedances",
            "es_final": empirical_es, "es_raw": np.nan, "es_mle_raw": np.nan,
            "es_fallback_type": "few_exceedances",
        }

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # genpareto.fit returns (shape=xi, loc, scale=sigma); floc=0 fixes location
            c, loc, sigma = genpareto.fit(exceedances, floc=0)

        # genpareto.fit can in pathological cases return non-positive or non-finite
        # sigma without raising. Treat as fit failure.
        if not np.isfinite(sigma) or sigma <= 0:
            warnings.warn(
                f"GPD fit returned invalid sigma={sigma!r} (must be finite and > 0). "
                f"Falling back to empirical quantile."
            )
            fb = _empirical_var(losses_w, alpha)
            empirical_es = _empirical_es(losses_w, alpha)
            return {
                "var_final": fb, "var_raw": fb, "var_mle_raw": np.nan,
                "xi": np.nan, "xi_raw": np.nan, "xi_clamped": False,
                "sigma_gpd": np.nan,
                "N_u": N_u, "f_u_hat": f_u_hat, "threshold_u": u,
                "ks_pvalue": np.nan,
                "fallback_type": "gpd_fit_failed",
                "es_final": empirical_es, "es_raw": np.nan, "es_mle_raw": np.nan,
                "es_fallback_type": "gpd_fit_failed",
            }

        xi_raw    = c           # preserve unconstrained MLE value
        sigma_gpd = sigma
        xi_clamped = False

        # Clamp xi to [-0.5, 1.0] as a defensive VaR guardrail, not a constrained MLE.
        # At xi=1.0 the GPD mean is still not finite, but VaR (a quantile) remains
        # well-defined for any alpha < 1.
        if xi_raw > 1.0:
            warnings.warn(
                f"GPD shape xi={xi_raw:.4f} exceeds upper guardrail 1.0; clamped to 1.0. "
                f"This is a defensive VaR guard, not a constrained MLE. "
                f"Note: at xi=1.0 the GPD mean is still not finite, but VaR (a quantile) "
                f"remains well-defined for any alpha < 1."
            )
            xi = 1.0
            xi_clamped = True
        elif xi_raw < -0.5:
            warnings.warn(
                f"GPD shape xi={xi_raw:.4f} below lower guardrail -0.5; clamped to -0.5. "
                f"This is a defensive VaR guard, not a constrained MLE. "
                f"Likely indicates threshold-too-high / sparse-tail artefact."
            )
            xi = -0.5
            xi_clamped = True
        else:
            xi = xi_raw

        ratio = (T_w / N_u) * (1.0 - alpha)

        if abs(xi) < 1e-4:                  # xi -> 0 limit: exponential excess distribution
            var_raw = u - sigma * np.log(ratio)
        else:
            var_raw = u + (sigma / xi) * (ratio ** (-xi) - 1.0)

        # Extreme xi or numerical issues in ratio**(-xi) can produce Inf or NaN.
        if not np.isfinite(var_raw):
            warnings.warn(
                f"GPD POT quantile var_raw={var_raw!r} is not finite "
                f"(xi={xi:.4f}, sigma={sigma:.4e}, ratio={ratio:.4e}). "
                f"Falling back to empirical quantile."
            )
            fb = _empirical_var(losses_w, alpha)
            empirical_es = _empirical_es(losses_w, alpha)
            return {
                "var_final": fb, "var_raw": np.nan, "var_mle_raw": np.nan,
                "xi": np.nan, "xi_raw": xi_raw, "xi_clamped": xi_clamped,
                "sigma_gpd": sigma_gpd,
                "N_u": N_u, "f_u_hat": f_u_hat, "threshold_u": u,
                "ks_pvalue": np.nan,
                "fallback_type": "gpd_fit_failed",
                "es_final": empirical_es, "es_raw": np.nan, "es_mle_raw": np.nan,
                "es_fallback_type": "gpd_fit_failed",
            }

        # Unconstrained MLE quantile using xi_raw (pre-clamp).
        # Equals var_raw when xi_clamped is False; only differs when clamp was active.
        if xi_clamped:
            try:
                with np.errstate(over="ignore", invalid="ignore"):
                    if abs(xi_raw) < 1e-4:
                        var_mle_raw = u - sigma * np.log(ratio)
                    else:
                        var_mle_raw = u + (sigma / xi_raw) * (ratio ** (-xi_raw) - 1.0)
                if not np.isfinite(var_mle_raw):
                    var_mle_raw = np.nan
            except Exception:
                var_mle_raw = np.nan
        else:
            var_mle_raw = var_raw

        # Error guard: var_raw < u despite alpha > f_u_hat indicates a degenerate
        # GPD fit (xi clamp, near-zero sigma, or numerical issue), not a legitimate
        # bounded-tail outcome. Fall back to empirical quantile.
        if var_raw < u:
            warnings.warn(
                f"GPD VaR {var_raw:.2f} below threshold u={u:.2f} "
                f"(xi={xi:.4f}, sigma={sigma:.4e}). This indicates a fit problem; "
                f"using empirical quantile fallback."
            )
            fallback = _empirical_var(losses_w, alpha)
            empirical_es = _empirical_es(losses_w, alpha)
            return {
                "var_final": fallback, "var_raw": var_raw, "var_mle_raw": var_mle_raw,
                "xi": np.nan, "xi_raw": xi_raw, "xi_clamped": xi_clamped,
                "sigma_gpd": sigma_gpd,
                "N_u": N_u, "f_u_hat": f_u_hat, "threshold_u": u,
                "ks_pvalue": np.nan,
                "fallback_type": "q_below_threshold",
                "es_final": empirical_es, "es_raw": np.nan, "es_mle_raw": np.nan,
                "es_fallback_type": "q_below_threshold",
            }

        # Defensive non-negativity floor (numerical guard only)
        var_final = max(var_raw, 0.0)

        # Compute GPD ES from clamped xi (es_raw) and unconstrained xi_raw (es_mle_raw).
        # xi >= 1.0 means GPD mean / ES is theoretically undefined (infinite); VaR stays.
        if abs(xi) < 1e-4:                   # Gumbel limit: exponential excess distribution
            es_raw = var_raw + sigma
        elif xi < 1.0:
            es_raw = (var_raw + sigma - xi * u) / (1.0 - xi)
        else:                                # xi == 1.0 (clamped): ES undefined under GPD
            es_raw = np.nan

        if not np.isfinite(var_mle_raw):
            es_mle_raw = np.nan
        elif abs(xi_raw) < 1e-4:
            es_mle_raw = var_mle_raw + sigma
        elif xi_raw < 1.0:
            es_mle_raw = (var_mle_raw + sigma - xi_raw * u) / (1.0 - xi_raw)
        else:
            es_mle_raw = np.nan

        # ES guardrails — independent of VaR guardrails.
        empirical_es = _empirical_es(losses_w, alpha)
        if xi >= 1.0:
            # GPD ES is undefined when xi >= 1; use empirical ES floored at var_final
            # so the policy invariant ES_EVT >= VaR_EVT always holds.
            es_final = max(empirical_es, var_final)
            es_fallback_type = "xi_ge_1_gpd_es_undefined"
        elif not np.isfinite(es_raw) or es_raw < var_final:
            # es_raw is non-finite or pathologically below VaR; use empirical ES.
            es_final = max(empirical_es, var_final)
            es_fallback_type = "es_invalid_empirical_fallback"
        else:
            es_final = max(es_raw, var_final, 0.0)
            es_fallback_type = "none"

        # KS test is invalid when xi was clamped — sigma is still MLE but xi is not,
        # so (xi, sigma) is not a jointly fitted distribution.
        if xi_clamped:
            ks_pvalue = np.nan
        else:
            _, ks_pvalue = kstest(exceedances, "genpareto", args=(xi, 0, sigma))

        return {
            "var_final": var_final, "var_raw": var_raw, "var_mle_raw": var_mle_raw,
            "xi": xi, "xi_raw": xi_raw, "xi_clamped": xi_clamped,
            "sigma_gpd": sigma_gpd,
            "N_u": N_u, "f_u_hat": f_u_hat, "threshold_u": u,
            "ks_pvalue": ks_pvalue,
            "fallback_type": "none",
            "es_final": es_final, "es_raw": es_raw, "es_mle_raw": es_mle_raw,
            "es_fallback_type": es_fallback_type,
        }

    except Exception as e:
        # Log failure so it is auditable, then fall back to empirical quantile.
        warnings.warn(
            f"GPD fit failed: {type(e).__name__}: {e}. Using empirical quantile fallback."
        )
        fb = _empirical_var(losses_w, alpha)
        empirical_es = _empirical_es(losses_w, alpha)
        return {
            "var_final": fb, "var_raw": fb, "var_mle_raw": np.nan,
            "xi": np.nan, "xi_raw": np.nan, "xi_clamped": False,
            "sigma_gpd": np.nan,
            "N_u": N_u, "f_u_hat": f_u_hat, "threshold_u": u,
            "ks_pvalue": np.nan,
            "fallback_type": "gpd_fit_failed",
            "es_final": empirical_es, "es_raw": np.nan, "es_mle_raw": np.nan,
            "es_fallback_type": "gpd_fit_failed",
        }


def compute_evt_var(pnl, window=WINDOW, threshold_q=THRESHOLD_Q, alpha=ALPHA,
                    min_start=None):
    """
    Rolling 500-day POT EVT VaR (Irle Section 9).

    For each day t in [start, T):
      losses_w = -pnl[t-window:t]   (dollar losses; positive = actual loss)
      Apply _pot_var() -> dict with var_final, xi, ks_pvalue, var_raw, etc.

    Losses are in dollar terms (from pnl_total), so no V0 scaling is needed.

    Parameters
    ----------
    min_start : int or None
        If provided, the loop starts at max(window, min_start) instead of window.
        Used by the sensitivity sweep to align all (threshold_q, window) combinations
        to the same backtest period.  Production calls omit this argument (None),
        which preserves the original loop start of t=window=500.

    Returns
    -------
    pd.DataFrame  columns: VaR_EVT, VaR_EVT_raw, VaR_EVT_mle_raw, xi, xi_raw,
                           xi_clamped, sigma_gpd, N_u, f_u_hat, threshold_u,
                           ks_pvalue, fallback_type,
                           ES_EVT, ES_EVT_raw, ES_EVT_mle_raw, ES_fallback_type
      VaR_EVT         — regulatory/policy output (non-negative; empirical fallback
                        if fit failed or degenerate)
      VaR_EVT_raw     — In "none" paths: POT quantile with clamped xi, before the
                        non-negativity floor. In "q_below_threshold": degenerate GPD
                        quantile that fell below threshold_u (stored intentionally).
                        In "gpd_fit_failed" (non-finite GPD quantile): NaN.
                        In other fallback paths: empirical quantile (same as VaR_EVT).
      VaR_EVT_mle_raw — POT quantile with unconstrained xi_raw; NaN in most fallback paths.
      ES_EVT          — 99% ES policy output (non-negative; empirical fallback if GPD
                        ES undefined or invalid). Always finite and >= VaR_EVT.
      ES_EVT_raw      — GPD ES from clamped xi before ES guardrails; NaN in fallback paths.
      ES_EVT_mle_raw  — GPD ES from unconstrained xi_raw; NaN when xi_raw >= 1 or unavailable.
      ES_fallback_type — reason ES fell back; "none" when GPD ES is used directly.
    """
    # Resolve loop start: production uses t=window; sensitivity can defer to common_start.
    start = max(window, min_start) if min_start is not None else window

    # Sample-length guard
    if len(pnl) <= start:
        raise ValueError(
            f"Insufficient data: len(pnl)={len(pnl)} <= start={start} "
            f"(window={window}, min_start={min_start}). "
            f"Need at least start+1 observations to produce any VaR forecast."
        )
    if len(pnl) < start + 50:
        warnings.warn(
            f"Short backtest period: only {len(pnl) - start} forecast days "
            f"(< 50). Backtest power will be very low; results indicative only."
        )

    n      = len(pnl)
    losses = (-pnl).values
    records, dates = [], []

    print(f"\n{'='*60}")
    print(f"EVT (POT) VaR  --  Irle Section 9")
    print(f"k={window} days | alpha={alpha:.0%} | threshold={threshold_q:.0%} quantile")
    print(f"Computing VaR for {n - start} days ...")

    for t in range(start, n):
        losses_w = losses[t - window : t]
        out = _pot_var(losses_w, threshold_q, alpha, window)
        records.append({
            "VaR_EVT"          : out["var_final"],
            "VaR_EVT_raw"      : out["var_raw"],
            "VaR_EVT_mle_raw"  : out["var_mle_raw"],
            "xi"               : out["xi"],
            "xi_raw"           : out["xi_raw"],
            "xi_clamped"       : out["xi_clamped"],
            "sigma_gpd"        : out["sigma_gpd"],
            "N_u"              : out["N_u"],
            "f_u_hat"          : out["f_u_hat"],
            "threshold_u"      : out["threshold_u"],
            "ks_pvalue"        : out["ks_pvalue"],
            "fallback_type"    : out["fallback_type"],
            "ES_EVT"           : out["es_final"],
            "ES_EVT_raw"       : out["es_raw"],
            "ES_EVT_mle_raw"   : out["es_mle_raw"],
            "ES_fallback_type" : out["es_fallback_type"],
        })
        dates.append(pnl.index[t])

    results   = pd.DataFrame(records, index=dates)

    # Guard against unexpected fallback_type values from future edits
    VALID_FALLBACK_TYPES = {
        "none", "few_exceedances", "gpd_fit_failed",
        "q_below_threshold", "alpha_below_threshold",
    }
    unexpected = set(results["fallback_type"].unique()) - VALID_FALLBACK_TYPES
    if unexpected:
        raise ValueError(f"Unexpected fallback_type values: {unexpected}")

    xi_ok      = results["xi"].dropna()
    n_poor_fit = (results["ks_pvalue"] < 0.05).sum()
    n_clamped  = int(results["xi_clamped"].sum())

    print(f"\nRolling EVT summary:")
    print(f"  Mean VaR         : ${results['VaR_EVT'].mean():>12,.0f}")
    print(f"  Min  VaR         : ${results['VaR_EVT'].min():>12,.0f}")
    print(f"  Max  VaR         : ${results['VaR_EVT'].max():>12,.0f}")
    print(f"  Mean ES          : ${results['ES_EVT'].mean():>12,.0f}")
    print(f"  Min  ES          : ${results['ES_EVT'].min():>12,.0f}")
    print(f"  Max  ES          : ${results['ES_EVT'].max():>12,.0f}")
    print(f"  Mean xi          :  {xi_ok.mean():.4f}  (positive xi suggests Pareto-type tail)")
    print(f"  Low KS p-value   : {n_poor_fit} / {n - start}  windows (p<0.05; anti-conservative, indicator only)")

    # Fallback breakdown (VaR)
    fb_counts = results["fallback_type"].value_counts()
    print(f"  Fallback breakdown:")
    for fb_name in ("none", "few_exceedances", "gpd_fit_failed",
                    "q_below_threshold", "alpha_below_threshold"):
        print(f"    {fb_name:>22}: {fb_counts.get(fb_name, 0)}")

    # ES fallback breakdown
    es_fb_counts = results["ES_fallback_type"].value_counts()
    print(f"  ES fallback breakdown:")
    for es_fb_name in ("none", "empirical_fallback", "xi_ge_1_gpd_es_undefined",
                       "es_invalid_empirical_fallback", "alpha_below_threshold",
                       "few_exceedances", "gpd_fit_failed", "q_below_threshold"):
        cnt = es_fb_counts.get(es_fb_name, 0)
        if cnt > 0:
            print(f"    {es_fb_name:>34}: {cnt}")

    print(f"  xi clamp triggered : {n_clamped} / {n - start}")

    # Validate result invariants before returning
    _validate_evt_results(results)
    print(f"  Result sanity checks: PASSED")

    return results

# =============================================================================
# STEP 4 — BACKTEST
# =============================================================================

def backtest_evt(pnl, results, alpha=ALPHA):
    """
    Kupiec + Christoffersen backtests via shared framework.
    (Irle Chapter 8, p. 183-185 — same as all other VaR methods in this project.)

    Alignment is checked before delegating to run_backtest() to ensure no silent
    index mismatches. alpha is an explicit parameter to avoid reliance on globals.
    """
    _assert_results_aligned_to_pnl(pnl, results, "backtest_evt")
    bt = run_backtest(
        pnl=pnl,
        var=results["VaR_EVT"],
        confidence=alpha,
        method_name="EVT",
    )
    print(bt)
    return bt

# =============================================================================
# STEP 4b — CRISIS-PERIOD BACKTESTING
# =============================================================================

def run_evt_crisis_backtests(pnl, results, alpha=ALPHA):
    """
    Crisis-period EVT diagnostics.

    The full-sample Kupiec / Christoffersen tests are supplemented by descriptive
    subperiod diagnostics.  Crisis windows are short, so p-values are reported
    where available but should be interpreted cautiously.

    Periods:
      Full sample
      GFC 2008       : 2008-09-15 to 2009-03-09
      COVID 2020     : 2020-02-19 to 2020-03-23
      Rate Hikes 2022: 2022-01-01 to 2022-10-01
      Non-crisis     : all dates outside the above crisis windows

    Saves:
      outputs/tables/evt_crisis_backtest.csv
    """
    losses = _assert_results_aligned_to_pnl(pnl, results, "run_evt_crisis_backtests")

    crisis_windows = [
        ("GFC 2008",        "2008-09-15", "2009-03-09"),
        ("COVID 2020",      "2020-02-19", "2020-03-23"),
        ("Rate Hikes 2022", "2022-01-01", "2022-10-01"),
    ]

    # Build a boolean mask for all crisis dates in the results index
    crisis_mask = pd.Series(False, index=results.index)
    for _, s, e in crisis_windows:
        crisis_mask |= (results.index >= s) & (results.index <= e)

    # Define subsets: full sample + each crisis + non-crisis
    subsets = [("Full sample", results.index)]
    for label, s, e in crisis_windows:
        subsets.append((label, results.index[(results.index >= s) & (results.index <= e)]))
    # Use .values to ensure boolean indexing is applied element-wise and not
    # by pandas label alignment, which could silently drop rows when the mask
    # index differs from results.index in edge cases.
    subsets.append(("Non-crisis", results.index[(~crisis_mask).values]))

    rows = []
    for label, idx in subsets:
        if len(idx) == 0:
            rows.append({
                "period": label, "start": np.nan, "end": np.nan,
                "n_obs": 0,
                "expected_exceptions": np.nan,
                "exceptions": np.nan, "exception_rate": np.nan,
                "mean_VaR": np.nan, "mean_ES": np.nan,
                "max_VaR": np.nan, "max_ES": np.nan,
                "max_actual_loss": np.nan, "avg_actual_loss": np.nan,
                "avg_exception_loss": np.nan, "max_exception_loss": np.nan,
                "kupiec_p": np.nan, "christoffersen_p": np.nan,
                "note": "no data in period",
            })
            continue

        sub_results = results.loc[idx]
        sub_losses  = losses.loc[idx]

        n_obs = len(idx)
        expected_exceptions = n_obs * (1.0 - alpha)
        exception_flag = sub_losses > sub_results["VaR_EVT"]
        exceptions     = int(exception_flag.sum())
        exception_rate = exceptions / n_obs if n_obs > 0 else np.nan

        exc_losses = sub_losses[exception_flag]
        avg_exception_loss = float(exc_losses.mean()) if len(exc_losses) > 0 else np.nan
        max_exception_loss = float(exc_losses.max()) if len(exc_losses) > 0 else np.nan

        # Formal Kupiec / Christoffersen only when there is enough data
        kupiec_p          = np.nan
        christoffersen_p  = np.nan
        note              = ""
        if n_obs >= 50 and sub_results["VaR_EVT"].notna().any():
            try:
                sub_pnl = pnl.loc[idx]
                bt_sub  = run_backtest(
                    pnl=sub_pnl,
                    var=sub_results["VaR_EVT"],
                    confidence=alpha,
                    method_name=f"EVT ({label})",
                )
                kupiec_p         = bt_sub.pvalue_uc
                christoffersen_p = getattr(bt_sub, "pvalue_cc", np.nan)
            except Exception as ex:
                note = f"backtest error: {type(ex).__name__}: {ex}"
            # Non-crisis is non-contiguous; Christoffersen independence test
            # assumes a consecutive time series so its p-value is descriptive only.
            if label == "Non-crisis":
                sep = "; " if note else ""
                note = note + sep + "non-contiguous; Christoffersen p descriptive only"
        else:
            note = "short period; descriptive only"
            if label == "Non-crisis":
                note = note + "; non-contiguous; Christoffersen p descriptive only"

        rows.append({
            "period"             : label,
            "start"              : str(idx.min().date()),
            "end"                : str(idx.max().date()),
            "n_obs"              : n_obs,
            "expected_exceptions": round(expected_exceptions, 2),
            "exceptions"         : exceptions,
            "exception_rate"     : round(exception_rate, 4) if exception_rate is not None else np.nan,
            "mean_VaR"           : round(float(sub_results["VaR_EVT"].mean()), 2),
            "mean_ES"            : round(float(sub_results["ES_EVT"].mean()), 2),
            "max_VaR"            : round(float(sub_results["VaR_EVT"].max()), 2),
            "max_ES"             : round(float(sub_results["ES_EVT"].max()), 2),
            "max_actual_loss"    : round(float(sub_losses.max()), 2),
            "avg_actual_loss"    : round(float(sub_losses.mean()), 2),
            "avg_exception_loss" : round(avg_exception_loss, 2) if not np.isnan(avg_exception_loss) else np.nan,
            "max_exception_loss" : round(max_exception_loss, 2) if not np.isnan(max_exception_loss) else np.nan,
            "kupiec_p"           : round(kupiec_p, 4) if not np.isnan(kupiec_p) else np.nan,
            "christoffersen_p"   : round(christoffersen_p, 4) if not np.isnan(christoffersen_p) else np.nan,
            "note"               : note,
        })

    crisis_df = pd.DataFrame(rows)
    path = os.path.join(OUTPUT_TABLES, "evt_crisis_backtest.csv")
    crisis_df.to_csv(path, index=False)
    print(f"  Crisis backtest saved -> {path}")

    # Compact summary table
    print(f"\n  Crisis-period backtest summary:")
    hdr = f"  {'Period':<20} {'n_obs':>6} {'Exc':>5} {'Exp':>6} {'Rate':>7} {'meanVaR':>12} {'meanES':>12}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for _, row in crisis_df.iterrows():
        if row["n_obs"] == 0:
            continue
        print(
            f"  {row['period']:<20} {row['n_obs']:>6.0f} "
            f"{row['exceptions']:>5.0f} {row['expected_exceptions']:>6.1f} "
            f"{row['exception_rate']:>7.2%} "
            f"${row['mean_VaR']:>11,.0f} ${row['mean_ES']:>11,.0f}"
        )

    return crisis_df


# =============================================================================
# STEP 5 — PLOTS
# =============================================================================

def plot_evt_results(pnl, results, bt):
    """
    Three-panel figure:
      Panel 1: EVT VaR vs actual loss with exception markers
      Panel 2: Rolling threshold u over time
      Panel 3: Rolling xi (GPD shape) over time
                positive xi values indicate Pareto-type heavy-tail behaviour in those windows (Irle p. 213).
    Crisis periods shaded in red.
    """
    crises = [
        ("2008-09-15", "2009-03-09", "GFC 2008"),
        ("2020-02-19", "2020-03-23", "COVID 2020"),
        ("2022-01-01", "2022-10-01", "Rate Hikes 2022"),
    ]

    var_s = results["VaR_EVT"]

    actual_loss = _assert_results_aligned_to_pnl(pnl, results, "plot_evt_results")

    exc_idx = bt.exceptions_index

    fig, axes = plt.subplots(3, 1, figsize=(14, 13), sharex=True)

    # Panel 1: VaR vs ES vs actual loss
    axes[0].fill_between(var_s.index, 0, var_s,
                         alpha=0.12, color="#E65100")
    axes[0].plot(var_s.index, var_s, color="#E65100", linewidth=1.2,
                 label="EVT 99% VaR (POT/GPD, Irle p.223)")
    axes[0].plot(results.index, results["ES_EVT"], color="#BF360C",
                 linewidth=0.8, alpha=0.6, linestyle="--",
                 label="EVT 99% ES / Tail VaR")
    axes[0].plot(actual_loss.index, actual_loss, color="#90A4AE",
                 linewidth=0.6, alpha=0.7, label="Actual loss (-dV)")
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
    axes[1].set_title("Rolling Threshold u -- 90th Percentile of Losses in Window",
                      fontsize=12, fontweight="bold")
    axes[1].set_ylabel("u  (USD)")
    axes[1].legend(fontsize=9)

    # Panel 3: Rolling xi (GPD shape parameter)
    xi = results["xi"]
    axes[2].plot(results.index, xi, color="#00796B", linewidth=0.9,
                 label="xi  (GPD shape parameter, clamped to [-0.5, 1.0])")
    axes[2].axhline(0, color="black", linewidth=0.8, linestyle="--",
                    label="xi = 0  (Gumbel boundary)")
    axes[2].fill_between(results.index, 0, xi.clip(lower=0),
                         alpha=0.12, color="#00796B",
                         label="Heavy-tail region  xi > 0")
    axes[2].set_title("Rolling xi -- GPD Shape Parameter  |  xi > 0 -> Pareto Heavy Tail (Irle p. 213)",
                      fontsize=12, fontweight="bold")
    axes[2].set_ylabel("xi  (shape)")
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

    Columns saved to backtest_evt.csv:
      VaR, VaR_raw, VaR_mle_raw, actual_loss, exception, xi, xi_raw,
      xi_clamped, sigma_gpd, N_u, f_u_hat, threshold_u, ks_pvalue, fallback_type,
      ES, ES_raw, ES_mle_raw, ES_fallback_type.

    VaR_raw     — POT quantile with clamped xi (diagnostic column).
    VaR_mle_raw — POT quantile with unconstrained xi_raw (equals VaR_raw when
                  xi_clamped is False).
    ES          — 99% ES policy output (always finite and >= VaR).
    ES_raw      — GPD ES from clamped xi before ES guardrails.
    ES_mle_raw  — GPD ES from unconstrained xi_raw; NaN when xi_raw >= 1 or unavailable.
    ES_fallback_type — reason ES fell back; "none" when GPD ES is used directly.
    """
    results.to_csv(os.path.join(PROCESSED_DIR, "var_evt.csv"))
    print(f"VaR saved     -> {os.path.join(PROCESSED_DIR, 'var_evt.csv')}")

    loss_aligned = _assert_results_aligned_to_pnl(pnl, results, "save_results")
    common = results.index   # all VaR dates verified present in pnl

    pd.DataFrame({
        "VaR"              : results.loc[common, "VaR_EVT"].values,
        "VaR_raw"          : results.loc[common, "VaR_EVT_raw"].values,
        "VaR_mle_raw"      : results.loc[common, "VaR_EVT_mle_raw"].values,
        "actual_loss"      : loss_aligned.values,
        "exception"        : (loss_aligned > results.loc[common, "VaR_EVT"]).astype(int).values,
        "xi"               : results.loc[common, "xi"].values,
        "xi_raw"           : results.loc[common, "xi_raw"].values,
        "xi_clamped"       : results.loc[common, "xi_clamped"].values,
        "sigma_gpd"        : results.loc[common, "sigma_gpd"].values,
        "N_u"              : results.loc[common, "N_u"].values,
        "f_u_hat"          : results.loc[common, "f_u_hat"].values,
        "threshold_u"      : results.loc[common, "threshold_u"].values,
        "ks_pvalue"        : results.loc[common, "ks_pvalue"].values,
        "fallback_type"    : results.loc[common, "fallback_type"].values,
        "ES"               : results.loc[common, "ES_EVT"].values,
        "ES_raw"           : results.loc[common, "ES_EVT_raw"].values,
        "ES_mle_raw"       : results.loc[common, "ES_EVT_mle_raw"].values,
        "ES_fallback_type" : results.loc[common, "ES_fallback_type"].values,
    }, index=common).to_csv(
        os.path.join(OUTPUT_TABLES, "backtest_evt.csv")
    )
    print(f"Backtest      -> {os.path.join(OUTPUT_TABLES, 'backtest_evt.csv')}")

# =============================================================================
# STEP 7 — THRESHOLD SENSITIVITY
# =============================================================================

def threshold_sensitivity(pnl,
                           thresholds=(0.85, 0.90, 0.925, 0.95),
                           window=WINDOW, alpha=ALPHA):
    """
    Re-run rolling POT VaR at alternative threshold_q values.

    Tests {0.85, 0.90, 0.925, 0.95} to assess sensitivity of VaR and backtest
    statistics to the threshold choice (assumption A.2 robustness check).
    Saves to outputs/tables/evt_threshold_sensitivity.csv.
    """
    print(f"\n{'='*60}")
    print(f"Threshold sensitivity sweep")
    print(f"  Thresholds tested: {thresholds}")

    rows = []
    for tq in thresholds:
        res_tq = compute_evt_var(pnl, window=window, threshold_q=tq, alpha=alpha)
        _assert_results_aligned_to_pnl(pnl, res_tq, f"threshold_sensitivity(tq={tq:.3f})")
        bt_tq  = run_backtest(
            pnl=pnl, var=res_tq["VaR_EVT"], confidence=alpha,
            method_name=f"EVT (tq={tq:.3f})",
        )
        xi_ok = res_tq["xi"].dropna()
        rows.append({
            "threshold_q"           : tq,
            "mean_VaR"              : res_tq["VaR_EVT"].mean(),
            "mean_ES"               : res_tq["ES_EVT"].mean(),
            "max_ES"                : res_tq["ES_EVT"].max(),
            "exceptions"            : bt_tq.N,
            "exception_rate"        : bt_tq.exception_rate,
            "kupiec_p"              : bt_tq.pvalue_uc,
            "christoffersen_p"      : getattr(bt_tq, "pvalue_cc", np.nan),
            "mean_xi"               : xi_ok.mean() if len(xi_ok) > 0 else np.nan,
            "n_q_below_threshold"   : int((res_tq["fallback_type"] == "q_below_threshold").sum()),
            "n_xi_clamped"          : int(res_tq["xi_clamped"].sum()),
            "n_few_exceedances"     : int((res_tq["fallback_type"] == "few_exceedances").sum()),
            "n_poor_ks"             : int((res_tq["ks_pvalue"] < 0.05).sum()),
            "n_ES_fallback"         : int((res_tq["ES_fallback_type"] != "none").sum()),
            "n_ES_xi_ge_1"          : int((res_tq["ES_fallback_type"] == "xi_ge_1_gpd_es_undefined").sum()),
        })

    sens_df = pd.DataFrame(rows)
    path = os.path.join(OUTPUT_TABLES, "evt_threshold_sensitivity.csv")
    sens_df.to_csv(path, index=False)

    print(f"\n  Threshold sensitivity summary:")
    print(f"  {'tq':>8}  {'mean_VaR':>12}  {'mean_ES':>12}  {'N_exc':>6}  "
          f"{'kupiec_p':>10}  {'mean_xi':>8}  {'n_clamp':>8}  {'n_ES_fb':>8}")
    for _, row in sens_df.iterrows():
        print(f"  {row['threshold_q']:>8.3f}  ${row['mean_VaR']:>11,.0f}  "
              f"${row['mean_ES']:>11,.0f}  {row['exceptions']:>6.0f}  "
              f"{row['kupiec_p']:>10.4f}  {row['mean_xi']:>8.4f}  "
              f"{row['n_xi_clamped']:>8.0f}  {row['n_ES_fallback']:>8.0f}")
    print(f"  Saved -> {path}")
    return sens_df


def evt_threshold_window_sensitivity(
    pnl,
    thresholds=(0.85, 0.90, 0.925, 0.95),
    windows=(250, 500, 750),
    alpha=ALPHA,
):
    """
    Joint sensitivity sweep over threshold_q x window combinations (diagnostic only).

    Runs the rolling POT EVT VaR for every (threshold_q, window) pair and collects
    backtest statistics for each combination.  This is a robustness check that the
    production model choice (THRESHOLD_Q=0.90, WINDOW=500) is stable to nearby
    hyperparameter values.  It does NOT change the production EVT model or outputs.

    All combinations are evaluated on a COMMON backtest period starting at index
    max(windows) = 750.  This ensures every row in the sensitivity table covers the
    same sample, making window comparisons apples-to-apples.  The production EVT
    model is unaffected: it still starts at index WINDOW=500.

    Note: the sensitivity grid row (threshold_q=0.90, window=500) uses the common
    start index 750, so it covers fewer observations than the standalone production
    backtest (which starts at 500).  Its Kupiec / Christoffersen statistics therefore
    differ from the headline production backtest by design.

    Threshold interpretation (approximate exceedances = WINDOW * (1 - threshold_q)):
      threshold_q=0.85 -> ~75 exceedances at WINDOW=500; more data, lower tail purity
      threshold_q=0.90 -> ~50 exceedances at WINDOW=500; production default
      threshold_q=0.925-> ~37 exceedances at WINDOW=500; cleaner tail, higher variance
      threshold_q=0.95 -> ~25 exceedances at WINDOW=500; very clean tail, noisy GPD fit

    Window interpretation (threshold_q=0.90 baseline):
      WINDOW=250: ~25 exceedances; fast regime adaptation, high estimation variance
      WINDOW=500: ~50 exceedances; production default, balanced stability-reactivity
      WINDOW=750: ~75 exceedances; more stable, slower to reflect regime shifts

    The window grid {250, 500, 750} is consistent with the group-wide sensitivity
    convention (Irle Section 9; McNeil & Frey 2000).

    Saves: outputs/tables/evt_threshold_window_sensitivity.csv
    Activated only via the --sensitivity command-line flag.
    """
    # All combinations share this start index so n_backtest_obs is identical across rows.
    common_start = max(windows)

    print(f"\n{'='*60}")
    print(f"EVT threshold x window sensitivity sweep  (diagnostic only)")
    print(f"  Thresholds : {thresholds}")
    print(f"  Windows    : {windows}")
    print(f"  Combinations: {len(thresholds) * len(windows)}")
    print(f"  Common sensitivity backtest start index: {common_start}")
    print(
        "  Note: production EVT still starts at WINDOW=500; the common start is "
        "used only for same-sample sensitivity comparison."
    )

    # Null row template — consistent CSV schema whether a combination succeeds or fails.
    _null = {
        "threshold_q": np.nan, "window": np.nan,
        "backtest_start_date": np.nan, "backtest_end_date": np.nan,
        "n_backtest_obs": np.nan,
        "mean_VaR": np.nan, "max_VaR": np.nan,
        "mean_ES": np.nan, "max_ES": np.nan,
        "exceptions": np.nan, "exception_rate": np.nan,
        "expected_exceptions": np.nan,
        "kupiec_p": np.nan, "christoffersen_p": np.nan,
        "mean_xi": np.nan,
        "avg_exceedances": np.nan,
        "n_xi_clamped": np.nan,
        "n_ES_fallback": np.nan,
        "n_ES_xi_ge_1": np.nan,
        "fallback_days": np.nan,
        "n_few_exceedances": np.nan,
        "n_gpd_fit_failed": np.nan,
        "n_q_below_threshold": np.nan,
        "n_poor_ks": np.nan,
        "note": "",
    }

    rows = []
    for window in windows:
        for tq in thresholds:
            print(f"  Running: threshold_q={tq:.3f}, window={window} ...", end=" ", flush=True)
            try:
                res = compute_evt_var(
                    pnl, window=window, threshold_q=tq, alpha=alpha,
                    min_start=common_start,
                )
            except ValueError as exc:
                print(f"SKIPPED ({exc})")
                row = dict(_null)
                row.update({"threshold_q": tq, "window": window, "note": str(exc)})
                rows.append(row)
                continue

            _assert_results_aligned_to_pnl(
                pnl, res, f"evt_threshold_window_sensitivity(tq={tq}, w={window})"
            )
            bt = run_backtest(
                pnl=pnl, var=res["VaR_EVT"], confidence=alpha,
                method_name=f"EVT (tq={tq:.3f}, w={window})",
            )

            xi_ok = res["xi"].dropna()
            n_backtest_obs = len(res)

            rows.append({
                "threshold_q"        : tq,
                "window"             : window,
                "backtest_start_date": str(res.index[0].date()),
                "backtest_end_date"  : str(res.index[-1].date()),
                "n_backtest_obs"     : n_backtest_obs,
                "mean_VaR"           : round(float(res["VaR_EVT"].mean()), 2),
                "max_VaR"            : round(float(res["VaR_EVT"].max()), 2),
                "mean_ES"            : round(float(res["ES_EVT"].mean()), 2),
                "max_ES"             : round(float(res["ES_EVT"].max()), 2),
                "exceptions"         : bt.N,
                "exception_rate"     : bt.exception_rate,
                "expected_exceptions": getattr(bt, "expected_N", n_backtest_obs * (1.0 - alpha)),
                "kupiec_p"           : bt.pvalue_uc,
                "christoffersen_p"   : getattr(bt, "pvalue_cc", np.nan),
                "mean_xi"            : float(xi_ok.mean()) if len(xi_ok) > 0 else np.nan,
                "avg_exceedances"    : float(res["N_u"].mean()),
                "n_xi_clamped"       : int(res["xi_clamped"].sum()),
                "n_ES_fallback"      : int((res["ES_fallback_type"] != "none").sum()),
                "n_ES_xi_ge_1"       : int((res["ES_fallback_type"] == "xi_ge_1_gpd_es_undefined").sum()),
                "fallback_days"      : int((res["fallback_type"] != "none").sum()),
                "n_few_exceedances"  : int((res["fallback_type"] == "few_exceedances").sum()),
                "n_gpd_fit_failed"   : int((res["fallback_type"] == "gpd_fit_failed").sum()),
                "n_q_below_threshold": int((res["fallback_type"] == "q_below_threshold").sum()),
                "n_poor_ks"          : int((res["ks_pvalue"] < 0.05).sum()),
                "note"               : "",
            })
            print(f"exceptions={bt.N}, kupiec_p={bt.pvalue_uc:.4f}")

    sens2_df = pd.DataFrame(rows)
    path = os.path.join(OUTPUT_TABLES, "evt_threshold_window_sensitivity.csv")
    sens2_df.to_csv(path, index=False)

    print(f"\n  Threshold x window sensitivity summary:")
    hdr = (f"  {'tq':>8}  {'window':>7}  {'n_obs':>6}  {'N_exc':>6}  "
           f"{'Exp':>6}  {'kupiec_p':>10}  {'mean_xi':>8}  {'avg_Nu':>8}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for _, row in sens2_df.iterrows():
        if pd.isna(row["n_backtest_obs"]):
            print(f"  {row['threshold_q']:>8.3f}  {int(row['window']):>7d}  SKIPPED")
            continue
        print(
            f"  {row['threshold_q']:>8.3f}  {int(row['window']):>7d}  "
            f"{row['n_backtest_obs']:>6.0f}  {row['exceptions']:>6.0f}  "
            f"{row['expected_exceptions']:>6.1f}  {row['kupiec_p']:>10.4f}  "
            f"{row['mean_xi']:>8.4f}  {row['avg_exceedances']:>8.1f}"
        )
    print(f"  Rows: {len(sens2_df)}  (expected {len(thresholds) * len(windows)})")
    print(f"  Saved -> {path}")
    return sens2_df

# =============================================================================
# STEP 8 — EVT PARAMETER STABILITY DIAGNOSTICS  (C6-equivalent)
# =============================================================================

def run_evt_parameter_stability_diagnostics(results):
    """
    Practical parameter stability summary for the unconditional EVT model.

    This is a C6-equivalent diagnostic, not a formal Nyblom test. It tracks the
    rolling xi, sigma_gpd, threshold_u, N_u, f_u_hat, VaR, fallback rates,
    clamp rates, and KS poor-fit counts across all forecast windows.

    C5 (Engle-Ng sign-bias) is not applicable here. evt.py is an unconditional
    rolling POT model on dollar losses. There is no conditional variance equation
    and no standardised GARCH residual process, so the sign-bias regression
    (z_t^2 ~ c0 + c1*I(z_{t-1}<0) + ...) is meaningless. C5 is implemented in
    garch_evt.py only.

    Saves:
      outputs/tables/evt_parameter_path.csv
      outputs/tables/evt_parameter_stability.csv
    """
    n_total = len(results)
    clean   = results[results["fallback_type"] == "none"].copy()

    if clean.empty:
        warnings.warn("run_evt_parameter_stability_diagnostics: no clean windows (all fallbacks).")
        return pd.DataFrame()

    path_csv = os.path.join(OUTPUT_TABLES, "evt_parameter_path.csv")
    path_cols = [
        "VaR_EVT", "VaR_EVT_raw", "VaR_EVT_mle_raw",
        "xi", "xi_raw", "xi_clamped", "sigma_gpd",
        "N_u", "f_u_hat", "threshold_u", "ks_pvalue", "fallback_type",
    ]
    for ec in ["ES_EVT", "ES_EVT_raw", "ES_EVT_mle_raw", "ES_fallback_type"]:
        if ec in results.columns:
            path_cols.append(ec)
    results[path_cols].to_csv(path_csv)
    print(f"  EVT parameter path saved -> {path_csv}")

    cols_of_interest = [c for c in
                        ["VaR_EVT", "VaR_EVT_raw", "VaR_EVT_mle_raw",
                         "ES_EVT", "ES_EVT_raw",
                         "xi", "xi_raw", "sigma_gpd", "threshold_u", "N_u", "f_u_hat"]
                        if c in clean.columns]

    summary_rows = []
    for col in cols_of_interest:
        s = clean[col].dropna()
        if len(s) < 2:
            continue
        step_changes = s.diff().abs().dropna()
        summary_rows.append({
            "parameter"          : col,
            "mean"               : s.mean(),
            "std"                : s.std(),
            "min"                : s.min(),
            "max"                : s.max(),
            "max_abs_step_change": step_changes.max() if len(step_changes) > 0 else np.nan,
        })

    summary_df = pd.DataFrame(summary_rows)

    # Global audit fields
    n_fallbacks  = int((results["fallback_type"] != "none").sum())
    n_xi_clamped = int(results["xi_clamped"].sum()) if "xi_clamped" in results.columns else 0
    n_poor_ks    = int((results["ks_pvalue"] < 0.05).sum())

    meta_row = pd.DataFrame([{
        "parameter": "_meta_n_clean_windows",
        "mean": len(clean), "std": np.nan, "min": np.nan,
        "max": np.nan, "max_abs_step_change": np.nan,
    }, {
        "parameter": "_meta_n_total_windows",
        "mean": n_total, "std": np.nan, "min": np.nan,
        "max": np.nan, "max_abs_step_change": np.nan,
    }, {
        "parameter": "_meta_n_fallbacks",
        "mean": n_fallbacks, "std": np.nan, "min": np.nan,
        "max": np.nan, "max_abs_step_change": np.nan,
    }, {
        "parameter": "_meta_n_xi_clamped",
        "mean": n_xi_clamped, "std": np.nan, "min": np.nan,
        "max": np.nan, "max_abs_step_change": np.nan,
    }, {
        "parameter": "_meta_n_poor_ks",
        "mean": n_poor_ks, "std": np.nan, "min": np.nan,
        "max": np.nan, "max_abs_step_change": np.nan,
    }])
    summary_df = pd.concat([summary_df, meta_row], ignore_index=True)

    stab_csv = os.path.join(OUTPUT_TABLES, "evt_parameter_stability.csv")
    summary_df.to_csv(stab_csv, index=False)
    print(f"  EVT parameter stability saved -> {stab_csv}")

    # Print headline stats
    xi_clean = clean["xi"].dropna() if "xi" in clean.columns else pd.Series(dtype=float)
    if len(xi_clean) >= 2:
        xi_step = xi_clean.diff().abs().dropna()
        print(f"  EVT xi range              : [{xi_clean.min():.4f}, {xi_clean.max():.4f}]")
        print(f"  EVT max |delta xi|        : {xi_step.max():.4f}")
    sg = clean["sigma_gpd"].dropna() if "sigma_gpd" in clean.columns else pd.Series(dtype=float)
    if len(sg) >= 1:
        print(f"  EVT sigma_gpd range       : [{sg.min():.4f}, {sg.max():.4f}]")
    tu = clean["threshold_u"].dropna() if "threshold_u" in clean.columns else pd.Series(dtype=float)
    if len(tu) >= 1:
        print(f"  EVT threshold_u range     : [${tu.min():,.0f}, ${tu.max():,.0f}]")
    print(f"  EVT fallback count        : {n_fallbacks} / {n_total}")
    print(f"  EVT xi clamp count        : {n_xi_clamped}")

    return summary_df


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    _parser = argparse.ArgumentParser(description="EVT (POT) 99% VaR  —  Irle Section 9")
    _parser.add_argument(
        "--sensitivity", action="store_true",
        help=(
            "Run the joint threshold x window sensitivity sweep "
            "(diagnostic only; does not change production EVT outputs). "
            "Saves outputs/tables/evt_threshold_window_sensitivity.csv."
        ),
    )
    _args = _parser.parse_args()

    os.makedirs(OUTPUT_FIGS,   exist_ok=True)
    os.makedirs(OUTPUT_TABLES, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

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

    # C6-equivalent: EVT parameter stability diagnostics
    print(f"\n{'='*60}")
    print(f"EVT parameter stability diagnostics")
    evt_stability_df = run_evt_parameter_stability_diagnostics(results)

    # Step 4: Backtest
    bt = backtest_evt(pnl, results)

    # Step 4b: Crisis-period diagnostics
    crisis_df = run_evt_crisis_backtests(pnl, results)

    # Step 5: Plots
    plot_evt_results(pnl, results, bt)

    # Step 6: Save
    save_results(pnl, results, bt)

    # Threshold and threshold x window sensitivity sweeps (diagnostic; --sensitivity only)
    if _args.sensitivity:
        sens_df = threshold_sensitivity(pnl)
        evt_threshold_window_sensitivity(pnl)

    # Error-guard activation count
    n_error_guard = int((results["fallback_type"] == "q_below_threshold").sum())
    print(f"  Error-guard activations (q_below_threshold): {n_error_guard} / {len(results)} "
          f"(expected 0 in production data)")

    print(f"\n{'='*60}")
    print(f"EVT (POT) VaR complete!")
    print(f"  Exceptions       : {bt.N}  (expected {bt.expected_N:.1f}  at 99% VaR)")
    print(f"  Kupiec H0        : {'NOT rejected' if not bt.reject_uc else 'REJECTED'}  "
          f"(p={bt.pvalue_uc:.4f})")
    cc_p = getattr(bt, "pvalue_cc", float("nan"))
    print(f"  Christoffersen p : {cc_p:.4f}")
    print(f"  Outputs:")
    print(f"    data/processed/var_evt.csv")
    print(f"    outputs/tables/backtest_evt.csv")
    print(f"    outputs/tables/evt_crisis_backtest.csv")
    print(f"    outputs/tables/evt_parameter_path.csv")
    print(f"    outputs/tables/evt_parameter_stability.csv")
    print(f"    outputs/figures/evt_01_threshold_diagnostics.png")
    print(f"    outputs/figures/evt_02_var_and_xi.png")
    if _args.sensitivity:
        print(f"    outputs/tables/evt_threshold_sensitivity.csv  [sensitivity]")
        print(f"    outputs/tables/evt_threshold_window_sensitivity.csv  [sensitivity]")
    print(f"{'='*60}")
