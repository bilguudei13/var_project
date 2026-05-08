# =============================================================================
# garch_evt.py  —  1-day 99% GARCH(1,1) + EVT Conditional VaR
#
# TWO-STEP PROCEDURE (McNeil & Frey 2000):
#
# STEP A — GARCH(1,1) volatility filter (Irle Section 8.3, p. 172-186):
#   r_t = mu_t + sigma_t * z_t,   z_t iid standardised Student-t(nu)  (mean=0, unit-variance normalised)
#   sigma_t^2 = omega + alpha_1*(r_{t-1} - mu)^2 + beta*sigma_{t-1}^2
#                                                          (Irle Eq. 17)
#   Stationarity condition: alpha_1 + beta < 1  (Irle p. 176)
#
#   Expanding-window re-estimation: GARCH is re-fitted every 50 days on r_p[0:t].
#   Between refits, sigma_t is propagated via the GARCH(1,1) recursion.
#   This eliminates look-ahead bias present in a single full-sample fit.
#
#   The lecture case study (Irle p. 173+) uses rolling 500-observation GARCH.
#   This implementation uses expanding-window GARCH as the production choice
#   because parameter stability is materially higher with the longer training
#   sample (typical alpha+beta drift < 0.02 across refits vs. ~0.05 for
#   rolling). The rolling-window challenger is implemented in fit_garch_rolling()
#   and accessible via `python garch_evt.py --challenger`.
#
#   In R, this is closest to rugarch::ugarchspec / rugarch::ugarchfit with
#   distribution.model='std'; fGarch::garchFit with cond.dist='std' is another
#   rough analogue.
#
# STEP B — EVT on standardised residuals (Irle Section 9, p. 212-225):
#   z_t = (r_t - mu_t) / sigma_t   (approximately i.i.d. if the GARCH filter is adequate)
#   z_losses = -z_t  (upper tail of losses = lower tail of standardised returns;
#                     POT is applied to z_losses so large positive values represent
#                     large standardised losses)
#
#   POT/GPD fitted to the upper tail of {z_losses}:
#     G_{xi,sigma}(x) = 1 - (1 + xi*x/sigma)^{-1/xi}    (Irle p.212, Def.7)
#   Threshold: u_z = quantile(z_losses, threshold_q)
#   POT quantile on residuals (dimensionless):
#     q_EVT(alpha) = u_z + (sigma/xi) * [(T_w/N_u*(1-alpha))^{-xi} - 1]
#   Gumbel limit (|xi| < 1e-4):
#     q_EVT = u_z - sigma * log((T_w/N_u) * (1-alpha))
#
#   scipy.stats.genpareto.fit(exceedances, floc=0) is conceptually analogous to
#   fitting excess losses with ismev::gpd.fit or evd::fpot in R, with the
#   threshold/location fixed at zero for the excess series.
#   pandas rolling/date-indexed Series are analogous to zoo/xts objects in R.
#
# STEP C — Conditional VaR (McNeil & Frey 2000, Eq. 4):
#   VaR_{alpha,t} = V0 * (-mu_t + sigma_t * q_EVT(alpha))
#
#   The conditional loss is L_t = -V0*r_t = -V0*mu_t - V0*sigma_t*z_t, so the
#   conditional mean enters with a sign flip: VaR = -mu_t + sigma_t * q_alpha(z).
#   mu_t is the GARCH-fitted constant mean active at time t (refit-boundary aware).
#
# =============================================================================
# DIAGNOSTICS
# =============================================================================
#
# GARCH adequacy (Irle p. 178-181):
#   Ljung-Box on z_t (lags 10, 20) — tests serial correlation in residuals.
#          In R: Box.test(z, lag=10, type="Ljung-Box").
#   Ljung-Box on z_t^2 (lags 10, 20) — tests remaining ARCH effects.
#   Engle ARCH-LM (10 lags) — direct test for remaining heteroscedasticity.
#   Pre-test: Ljung-Box on raw returns r_p — screens for serial correlation that
#             would motivate an AR mean specification.
#   p > 0.05 means we fail to reject the respective null; this supports but does
#   not prove GARCH adequacy (no power analysis performed).
#   Results saved to outputs/tables/garch_evt_diagnostics.csv.
#
# C5 — Engle-Ng (1993) sign-bias test on standardised residuals:
#   Regression:
#     z_t^2 = c0 + c1*I(z_{t-1}<0) + c2*I(z_{t-1}<0)*z_{t-1}
#             + c3*I(z_{t-1}>=0)*z_{t-1} + u_t
#   H0: c1 = c2 = c3 = 0.
#   Low p-value (p <= 0.05) suggests asymmetric news impact and motivates a
#   GJR-GARCH / EGARCH / APARCH challenger, but does not automatically invalidate
#   the current symmetric GARCH(1,1). Run on full sample and backtest period.
#
# C6 — GARCH parameter stability (not a formal Nyblom test):
#   Practical stability summary across refits: tracks mu, omega, alpha1, beta,
#   alpha1+beta, nu, optimizer warnings, and stationarity fallback flags.
#   raw_alpha_plus_beta = fitted value before the stationarity guard.
#   alpha_plus_beta     = value actually used after possible fallback.
#   Stationarity: alpha1 + beta >= 0.999 is treated as near-integrated; the
#   implementation reverts to the last valid parameters and recomputes sigma_t
#   under the reverted regime so subsequent recursion is self-consistent.
#
# Residual threshold diagnostics:
#   MEP and Hill estimator on backtest-period standardised residual losses,
#   analogous to evt.py plot_threshold_diagnostics() applied to z_t.
#   Provides diagnostic evidence for a GPD tail approximation on residuals.
#
# Threshold sensitivity:
#   Re-runs the EVT step at {0.85, 0.90, 0.925, 0.95} with GARCH fixed.
#   Saved to outputs/tables/garch_evt_threshold_sensitivity.csv.
#
# =============================================================================
# SCALING NOTE (matches GARCH.py / GARCH R.r convention):
#   arch_model() is numerically more stable when small decimal returns are scaled to percentage units.
#   scale_factor = 100 if series.std() < 0.1 else 1
#   Pass r_p * scale_factor; convert back:
#     cond_vol_decimal = res.conditional_volatility / scale_factor
#     forecast_var_decimal = res.forecast(...).variance / scale_factor**2
# =============================================================================
#
# Theory references:
#   [McNeil & Frey (2000)] "Estimation of tail-related risk measures for
#    heteroscedastic financial time series", J. Empirical Finance 7, 271-300
#   [Irle Lecture Notes] Section 8.3 (GARCH, p. 172-186)
#                      + Section 9 (EVT / POT, p. 212-225)
# =============================================================================

import os
import sys
import argparse
import warnings
if hasattr(sys.stdout, "reconfigure"):   # guard for non-reconfigurable stdout
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive: save to disk, no window
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import genpareto, kstest, t as student_t, probplot
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
import statsmodels.api as sm          # C5: Engle-Ng sign-bias OLS regression

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from backtesting.backtest import run_backtest

# =============================================================================
# SETTINGS
# =============================================================================

WINDOW          = 500
ALPHA           = 0.99
THRESHOLD_Q     = 0.90        # 90th pctile of residual losses -> ~50 exceedances/window
MIN_EXCEEDANCES = 10
V0              = 1_000_000
REFIT_EVERY     = 50          # re-estimate GARCH every N days

PROCESSED_DIR = os.path.join("data", "processed")
OUTPUT_FIGS   = os.path.join("outputs", "figures")
OUTPUT_TABLES = os.path.join("outputs", "tables")

# =============================================================================
# STEP 1 -- LOAD DATA
# =============================================================================

def load_data():
    """
    Load total portfolio P&L and derive total portfolio return for GARCH.

    GARCH is fit on total portfolio returns (pnl_total / V0) so that the
    conditional volatility captures all risk sources: linear positions,
    IRS (DV01 sensitivity), and ATM straddle (gamma/vega). Fitting on
    linear-only returns can materially understate VaR when backtested
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
    # Includes linear + IRS + straddle components.
    r_p = pnl / V0
    r_p.name = "portfolio_return_total"

    print(f"Loaded: {len(pnl)} days of total portfolio P&L")
    print(f"Date range : {pnl.index[0].date()} -> {pnl.index[-1].date()}")
    print(f"r_p (total): mean={r_p.mean():.6f}  std={r_p.std():.6f}")
    return r_p, pnl

# =============================================================================
# STEP 2 -- EXPANDING-WINDOW GARCH(1,1) FIT
# =============================================================================

def fit_garch_expanding(r_p, window=WINDOW, refit_every=REFIT_EVERY):
    """
    Fit GARCH(1,1) with Student-t innovations using an expanding window.

    Expanding-window design eliminates look-ahead bias: the GARCH model at time t
    is estimated only on r_p[0:t], never on future data.

    Algorithm
    ---------
    For each backtest day t in [window, T):
      - If t == window or (t - window) % refit_every == 0:
          Re-estimate GARCH on r_p[0:t] in %-units (expanding window).
          Store omega, alpha_1, beta, mu (all in decimal).
          Use one-step-ahead forecast for cond_vol[t] = sigma_t.
          On the FIRST refit (t=window): initialise z_arr[0:window] and
          mu_arr[0:window] from in-sample vols/mean so that the first EVT
          window is populated.
      - Else:
          Propagate via GARCH(1,1) recursion (decimal units):
            sigma^2_t = omega + alpha_1*(r_{t-1}-mu)^2 + beta*sigma^2_{t-1}
      After each step: z_arr[t-1] = (r_{t-1} - mu) / cond_vol[t-1]
                       (fill the standardised residual for the previous day)

    The GARCH-fitted mean mu = res.params['mu'] / scale_factor is used for residuals
    (internally consistent with the fitted GARCH model rather than the raw sample mean).

    Stationarity guard: near-integrated fits (alpha+beta >= 0.999) are rejected;
    parameters revert to the last valid estimate. When fallback fires,
    cond_vol_arr[t] and sigma_sq_curr are BOTH recomputed under the reverted
    regime so subsequent recursion is self-consistent.

    Optimizer warnings are captured (not suppressed) and counted per refit.

    Note on residual coherence across refit boundaries:
      At each refit, the GARCH parameters (mu, omega, alpha, beta) are updated.
      Historical residuals z_arr[:t-1] computed under earlier parameter regimes
      are NOT recomputed under the new parameters. This is intentional:
      recomputing would introduce look-ahead bias (using future-fit parameters
      to alter past residuals), which would invalidate the backtest. z_arr[t-1]
      is computed under the same parameter regime that produced cond_vol_arr[t-1];
      older residuals retain their historical regime. The EVT window thus contains
      residuals from possibly different regimes, but each is consistent with the
      GARCH parameters in force at its date — the walk-forward-correct choice.
      Parameter stability (typical alpha+beta drift < 0.02 across refits in this
      data) makes this approximation tight in practice.

    Parameters
    ----------
    r_p         : pd.Series  total portfolio return (decimal)
    window      : int        initial training window (days)
    refit_every : int        re-estimation frequency (days)

    Returns
    -------
    cond_vol    : pd.Series  sigma_t in decimal return units (index = r_p.index)
    z_residuals : pd.Series  standardised residuals (index = r_p.index)
    mu_series   : pd.Series  GARCH-fitted conditional mean mu_t, decimal
                             (constant within each refit interval; index = r_p.index)
    refit_dates : list       dates at which GARCH was re-estimated
    last_res    : arch ModelResult from the final refit (for diagnostics)
    param_df    : pd.DataFrame  one row per refit; includes raw pre-fallback
                                and post-fallback param columns (C6)
    """
    n        = len(r_p)
    r_vals   = r_p.values
    dates    = r_p.index

    # Index validation — duplicates or non-monotone ordering would silently
    # corrupt walk-forward semantics.
    assert r_p.index.is_monotonic_increasing, "r_p index must be sorted ascending"
    assert r_p.index.is_unique,                 "r_p index contains duplicate dates"

    # Index convention:
    #   r_p.index[t]    = date of return r_t (close-to-close return)
    #   pnl.index[t]    = date of P&L delta on day t (same convention as r_p)
    #   cond_vol[t]     = sigma_t = forecast for day t made at end of day t-1
    #   VaR[t]          = forecast for day t made at end of day t-1
    # Backtest: exception_t = 1 iff actual_loss[t] > VaR[t], same-date indexing.

    # Conditional scaling matches GARCH.py's numerical-stability pattern
    # (scale by 100 when |r| is small enough that arch's optimizer would
    # otherwise lose precision). For portfolio returns with std ~= 0.01
    # this branch always resolves to scale_factor = 100; the conditional
    # is kept for consistency with the teammate's reference implementation.
    scale_factor  = 100.0 if r_p.std() < 0.1 else 1.0
    scale_factor2 = scale_factor ** 2   # for variance-unit parameters (omega, forecast var)

    cond_vol_arr = np.full(n, np.nan)
    z_arr        = np.full(n, np.nan)
    mu_arr       = np.full(n, np.nan)   # conditional mean series

    mu_hat   = None      # signals "not yet estimated"
    omega_d  = None
    alpha1_d = None
    beta_d   = None
    sigma_sq_curr = np.nan   # sigma^2 at current t (decimal)

    # Stationarity guard state
    last_valid_params       = None
    n_stationarity_fallbacks = 0

    # Optimizer warning tracking
    n_warnings_total = 0
    refit_warnings   = []

    n_refits          = 0
    refit_dates       = []
    last_res          = None
    parameter_records = []   # C6: parameter path for stability diagnostics

    # Ljung-Box pre-test on raw returns before GARCH loop (Irle p. 178)
    lb_raw = acorr_ljungbox(r_p.dropna().values, lags=[10, 20], return_df=True)
    print(f"\nReturn autocorrelation pre-test (Irle p. 178):")
    print(f"  Ljung-Box on r_p (lag 10): p={lb_raw.iloc[0, 1]:.4f}")
    print(f"  Ljung-Box on r_p (lag 20): p={lb_raw.iloc[1, 1]:.4f}")
    if lb_raw.iloc[1, 1] < 0.05:
        print(f"  WARNING: significant autocorrelation in raw returns at 5%. "
              f"Consider AR(1) mean specification (currently using Constant).")
    else:
        print(f"  No significant raw-return autocorrelation detected; "
              f"constant-mean specification is acceptable as a baseline.")

    print(f"\n{'='*60}")
    print(f"GARCH(1,1) expanding-window fit  --  no look-ahead bias")
    print(f"dist=Student-t | scale_factor={scale_factor:.0f} | "
          f"refit every {refit_every} days | total obs={n} | backtest days={n - window}")

    for t in range(window, n):
        do_refit = (mu_hat is None) or ((t - window) % refit_every == 0)

        # Capture mu_hat BEFORE a refit overwrites it.  z_arr[t-1] must use the
        # same parameter regime that produced cond_vol_arr[t-1] — no cross-regime
        # contamination at refit boundaries.
        mu_hat_prev = mu_hat

        if do_refit:
            # ---- Re-estimate GARCH on r_p[0:t] * scale_factor ----
            train = r_p.iloc[:t] * scale_factor   # scaled units for arch
            garch = arch_model(train, vol="Garch", p=1, q=1,
                               dist="t", mean="Constant")

            # Capture warnings rather than suppressing them
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                res_t = garch.fit(disp="off")

            if caught:
                n_warnings_total += len(caught)
                refit_warnings.append({
                    "refit_idx" : n_refits,
                    "date"      : dates[t],
                    "n_warnings": len(caught),
                    "messages"  : [str(w.message) for w in caught],
                })

            # GARCH-fitted mean in decimal units (divide by scale_factor)
            mu_hat   = res_t.params["mu"]      / scale_factor
            omega_d  = res_t.params["omega"]   / scale_factor2
            alpha1_d = res_t.params["alpha[1]"]
            beta_d   = res_t.params["beta[1]"]

            # Capture raw (pre-fallback) parameters before stationarity guard may overwrite
            raw_mu     = mu_hat
            raw_omega  = omega_d
            raw_alpha1 = alpha1_d
            raw_beta   = beta_d

            # One-step-ahead forecast -> sigma_t for this date (divide by scale_factor2)
            fcast = res_t.forecast(horizon=1, reindex=False)
            sigma_sq_curr   = fcast.variance.iloc[-1, 0] / scale_factor2
            cond_vol_arr[t] = np.sqrt(max(sigma_sq_curr, 1e-10))

            # Stationarity guard: near-integrated GARCH is numerically unstable.
            # alpha1 + beta >= 0.999 triggers revert to last valid parameters.
            ab = alpha1_d + beta_d   # raw pre-fallback; used as raw_alpha_plus_beta below
            if ab >= 0.999:
                if last_valid_params is None:
                    raise RuntimeError(
                        f"Initial GARCH fit is near-integrated at t={t}: "
                        f"alpha+beta={ab:.6f}. No previous valid parameters available; "
                        f"cannot proceed."
                    )
                warnings.warn(
                    f"Near-integrated GARCH at t={t}: alpha+beta={ab:.6f}. "
                    f"Reverting to previous valid parameters and recomputing cond_vol_arr[t] "
                    f"under the reverted regime."
                )
                mu_hat   = last_valid_params["mu"]
                omega_d  = last_valid_params["omega"]
                alpha1_d = last_valid_params["alpha1"]
                beta_d   = last_valid_params["beta"]

                # Recompute cond_vol_arr[t] and sigma_sq_curr under the reverted
                # parameter regime so subsequent recursion is self-consistent.
                prev_sigma_sq = (cond_vol_arr[t - 1] ** 2
                                 if not np.isnan(cond_vol_arr[t - 1])
                                 else omega_d / max(1.0 - alpha1_d - beta_d, 1e-6))
                innov_prev    = r_vals[t - 1] - mu_hat
                sigma_sq_curr = (omega_d
                                 + alpha1_d * innov_prev ** 2
                                 + beta_d   * prev_sigma_sq)
                sigma_sq_curr   = max(sigma_sq_curr, 1e-10)
                cond_vol_arr[t] = np.sqrt(sigma_sq_curr)

                n_stationarity_fallbacks += 1
            else:
                last_valid_params = {
                    "mu"    : mu_hat,
                    "omega" : omega_d,
                    "alpha1": alpha1_d,
                    "beta"  : beta_d,
                }

            # Record both raw (pre-fallback) and post-fallback parameters for C6.
            # raw_alpha_plus_beta is the true fitted value before any stationarity guard.
            parameter_records.append({
                "refit_idx"               : n_refits,
                "date"                    : dates[t],
                "t"                       : t,
                "mu"                      : mu_hat,          # POST-fallback
                "omega"                   : omega_d,          # POST-fallback
                "alpha1"                  : alpha1_d,         # POST-fallback
                "beta"                    : beta_d,           # POST-fallback
                "alpha_plus_beta"         : alpha1_d + beta_d,  # POST-fallback
                "raw_mu"                  : raw_mu,
                "raw_omega"               : raw_omega,
                "raw_alpha1"              : raw_alpha1,
                "raw_beta"                : raw_beta,
                "raw_alpha_plus_beta"     : ab,              # = raw_alpha1 + raw_beta (pre-fallback)
                "nu"                      : res_t.params.get("nu", np.nan),
                "loglikelihood"           : getattr(res_t, "loglikelihood", np.nan),
                "convergence_flag"        : getattr(res_t, "convergence_flag", np.nan),
                "optimizer_warning_count" : len(caught),
                "stationarity_fallback_used": int(ab >= 0.999),
            })

            if n_refits == 0:
                # First refit: initialise z_arr[0:window] and mu_arr[0:window]
                cv_d = res_t.conditional_volatility.values / scale_factor
                z_arr[:t] = np.where(
                    cv_d > 1e-10,
                    (r_vals[:t] - mu_hat) / cv_d,
                    np.nan
                )
                mu_arr[:t] = mu_hat   # backfill initial window with first mu estimate

                # Print GARCH parameters once (initial fit)
                flag = "OK" if ab < 1.0 else "!! UNIT ROOT"
                print(f"\nGARCH(1,1) initial parameters (t={t}, Irle Eq. 17):")
                print(f"  mu      = {mu_hat:.8f}  (decimal)")
                print(f"  omega   = {omega_d:.10f}  (decimal)")
                print(f"  alpha_1 = {alpha1_d:.6f}")
                print(f"  beta    = {beta_d:.6f}")
                print(f"  alpha_1 + beta = {ab:.6f}  -> {flag}  (Irle p. 176)")

            refit_dates.append(dates[t])
            n_refits += 1
            last_res  = res_t

            if n_refits % 10 == 0:
                print(f"  Refit {n_refits:3d}: t={t} / {n}  "
                      f"(alpha+beta={alpha1_d+beta_d:.5f})")

        else:
            # ---- GARCH(1,1) recursion with last-fitted parameters ----
            innov         = r_vals[t - 1] - mu_hat
            sigma_sq_curr = (omega_d
                             + alpha1_d * innov ** 2
                             + beta_d   * sigma_sq_curr)
            sigma_sq_curr   = max(sigma_sq_curr, 1e-10)
            cond_vol_arr[t] = np.sqrt(sigma_sq_curr)

        # Record active conditional mean for this date
        mu_arr[t] = mu_hat

        # Fill z_arr[t-1] using the parameter regime under which cond_vol_arr[t-1]
        # was produced.  mu_hat_prev is None on the first iteration (t==window); skip
        # the fill there — z_arr[:window] is already initialised by the first-refit block.
        if t > window and mu_hat_prev is not None and cond_vol_arr[t - 1] > 1e-10:
            z_arr[t - 1] = (r_vals[t - 1] - mu_hat_prev) / cond_vol_arr[t - 1]

    cond_vol    = pd.Series(cond_vol_arr, index=dates, name="cond_vol")
    z_residuals = pd.Series(z_arr,        index=dates, name="z_residuals")
    mu_series   = pd.Series(mu_arr,       index=dates, name="mu_hat")

    z_ok = z_residuals.dropna()
    print(f"\nExpanding-window GARCH summary:")
    print(f"  Total refits              : {n_refits}")
    print(f"  Last refit                : {refit_dates[-1].date() if refit_dates else 'N/A'}")
    print(f"  z mean (window+)          : {z_ok.iloc[window:].mean():.4f}  (~=0 expected)")
    print(f"  z std  (window+)          : {z_ok.iloc[window:].std():.4f}   (~=1 expected)")
    print(f"  Stationarity fallbacks    : {n_stationarity_fallbacks}  (expected 0)")
    print(f"  Total optimizer warnings  : {n_warnings_total}  (across {n_refits} refits)")

    # Persist captured optimizer warnings for audit
    if refit_warnings:
        pd.DataFrame(refit_warnings).to_csv(
            os.path.join(OUTPUT_TABLES, "garch_evt_refit_warnings.csv"),
            index=False
        )
        print(f"  Refit warnings saved -> "
              f"{os.path.join(OUTPUT_TABLES, 'garch_evt_refit_warnings.csv')}")
    else:
        print(f"  No optimizer warnings during any refit (clean run).")

    param_df = pd.DataFrame(parameter_records) if parameter_records else pd.DataFrame()   # C6
    return cond_vol, z_residuals, mu_series, refit_dates, last_res, param_df

# =============================================================================
# STEP 3 -- GARCH DIAGNOSTICS (plots + adequacy tests)
# =============================================================================

def plot_garch_diagnostics(r_p, cond_vol, z_residuals):
    """
    Two diagnostic panels:

    Panel 1: GARCH conditional volatility (annualised *sqrt(252)) over time.
             Demonstrates volatility clustering captured by GARCH (Irle p. 172).
             Crisis spikes (GFC, COVID) are consistent with the model capturing volatility clustering.

    Panel 2: QQ-plot of standardised residuals vs Student-t distribution.
             Heavy tails in residuals motivate EVT on Z_t (Irle p. 179-180).
             If residual tails were already well described by the fitted Student-t,
             the EVT step would add little beyond parametric GARCH-t.
    """
    crises = [
        ("2008-09-15", "2009-03-09", "GFC 2008"),
        ("2020-02-19", "2020-03-23", "COVID 2020"),
        ("2022-01-01", "2022-10-01", "Rate Hikes 2022"),
    ]

    ann_vol = cond_vol.dropna() * np.sqrt(252) * 100    # annualised vol in %
    z       = z_residuals.dropna().values

    # Fit Student-t to standardised residuals for QQ reference
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dof, loc_fit, scale_fit = student_t.fit(z, floc=0)
    z_norm = (z - loc_fit) / scale_fit

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: GARCH conditional vol (expanding-window)
    axes[0].plot(ann_vol.index, ann_vol, color="#1976D2",
                 linewidth=0.8, label="sigma_t annualised (%)")
    axes[0].set_title(
        "GARCH(1,1) Conditional Volatility -- Annualised\n"
        "Expanding-window re-estimation every 50 days  |  Irle p. 174",
        fontsize=11, fontweight="bold")
    axes[0].set_ylabel("sigma_t  (%/year)")
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    first_crisis = True
    for s, e, lbl in crises:
        kwargs = {"alpha": 0.10, "color": "red"}
        if first_crisis:
            kwargs["label"] = "Crisis periods"
            first_crisis = False
        axes[0].axvspan(s, e, **kwargs)
    axes[0].legend(fontsize=8)

    # Panel 2: QQ-plot vs Student-t
    (osm, osr), (slope, intercept, _) = probplot(
        z_norm, dist=student_t, sparams=(dof,)
    )
    axes[1].scatter(osm, osr, color="#43A047", s=5, alpha=0.4,
                    label=f"Residuals  (n={len(z)})")
    line_x = np.array([min(osm), max(osm)])
    axes[1].plot(line_x, slope * line_x + intercept,
                 color="#C62828", linewidth=1.6,
                 label=f"Student-t(nu={dof:.1f}) reference line")
    axes[1].set_xlabel("Theoretical quantiles  [Student-t]")
    axes[1].set_ylabel("Sample quantiles  [Z_t]")
    axes[1].set_title(
        "QQ-Plot: Standardised Residuals vs Student-t\n"
        "Tail deviations justify EVT on Z_t  |  Irle p. 179-180",
        fontsize=11, fontweight="bold")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(OUTPUT_FIGS, "garch_evt_01_diagnostics.png")
    plt.savefig(path, dpi=150)
    plt.close("all")
    print(f"GARCH diagnostics saved -> {path}")


def engle_ng_sign_bias_test(z_series, label="full"):
    """
    C5 — Engle-Ng (1993) sign-bias test on standardised residuals.

    Regression:
        z_t^2 = c0
              + c1 * I(z_{t-1} < 0)
              + c2 * I(z_{t-1} < 0)  * z_{t-1}
              + c3 * I(z_{t-1} >= 0) * z_{t-1}
              + u_t

    H0: c1 = c2 = c3 = 0.
    Low p-value (p <= 0.05) suggests asymmetric news impact may remain and symmetric
    GARCH(1,1) may be misspecified; consider GJR-GARCH / EGARCH / APARCH challenger.
    Do not change the production model on this basis alone.
    """
    if isinstance(z_series, pd.Series):
        z = z_series.dropna().values
    else:
        z = np.asarray(z_series, dtype=float)
        z = z[np.isfinite(z)]

    n = len(z)
    nan_result = {
        f"{label}_sign_bias_p"           : np.nan,
        f"{label}_negative_size_bias_p"  : np.nan,
        f"{label}_positive_size_bias_p"  : np.nan,
        f"{label}_joint_sign_bias_p"     : np.nan,
        f"{label}_sign_bias_n"           : n,
    }

    if n < 50:
        warnings.warn(f"Engle-Ng ({label}): only {n} observations; skipping (need >= 50).")
        return nan_result

    z_s       = pd.Series(z)
    z_lag     = z_s.shift(1)
    z_sq      = z_s ** 2
    mask      = z_lag.notna()
    z_lag_v   = z_lag[mask].values

    indicator_neg   = (z_lag_v < 0).astype(float)
    neg_size_bias   = indicator_neg * z_lag_v
    pos_size_bias   = (1.0 - indicator_neg) * z_lag_v
    y               = z_sq[mask].values

    # Guard degenerate regressors: skip if any regressor has near-zero variance
    regressors = np.column_stack([indicator_neg, neg_size_bias, pos_size_bias])
    if np.any(regressors.std(axis=0) < 1e-10):
        warnings.warn(f"Engle-Ng ({label}): degenerate regressor(s); returning NaN.")
        return {**nan_result, f"{label}_sign_bias_n": int(mask.sum())}

    try:
        X = sm.add_constant(pd.DataFrame({
            "sign_bias"         : indicator_neg,
            "negative_size_bias": neg_size_bias,
            "positive_size_bias": pos_size_bias,
        }))
        model = sm.OLS(y, X).fit()

        pvals       = model.pvalues
        p_sign      = float(pvals.get("sign_bias",          np.nan))
        p_neg       = float(pvals.get("negative_size_bias", np.nan))
        p_pos       = float(pvals.get("positive_size_bias", np.nan))

        # Joint F-test: c1 = c2 = c3 = 0 via R-matrix (robust to column ordering)
        try:
            pnames  = model.params.index.tolist()
            idx_sb  = pnames.index("sign_bias")
            idx_nsb = pnames.index("negative_size_bias")
            idx_psb = pnames.index("positive_size_bias")
            k       = len(pnames)
            R       = np.zeros((3, k))
            R[0, idx_sb]  = 1.0
            R[1, idx_nsb] = 1.0
            R[2, idx_psb] = 1.0
            p_joint = float(model.f_test(R).pvalue)
        except Exception as fe:
            warnings.warn(f"Engle-Ng ({label}) joint F-test failed: {fe}")
            p_joint = np.nan

        return {
            f"{label}_sign_bias_p"          : p_sign,
            f"{label}_negative_size_bias_p" : p_neg,
            f"{label}_positive_size_bias_p" : p_pos,
            f"{label}_joint_sign_bias_p"    : p_joint,
            f"{label}_sign_bias_n"          : int(mask.sum()),
        }
    except Exception as e:
        warnings.warn(f"Engle-Ng sign-bias test ({label}) failed: {e}")
        return {**nan_result, f"{label}_sign_bias_n": int(mask.sum())}


def plot_residual_threshold_diagnostics(z_residuals):
    """
    Diagnostic threshold plots on backtest-period standardised residual losses.

    Analogous to evt.py plot_threshold_diagnostics(), applied to residual losses
    z_losses = -z_residuals (backtest period only: iloc[WINDOW:]).

    Panel 1: Mean Excess Plot (MEP) — e(u_z) = E[z_loss - u_z | z_loss > u_z].
             Linear region above u_z validates GPD on residuals.
    Panel 2: Hill estimator on strictly positive residual losses.
             Stability in k confirms heavy-tail behaviour in residuals.

    This is diagnostic only; it does not feed into VaR computation.
    Saved to outputs/figures/garch_evt_03_residual_threshold_diagnostics.png.
    """
    z_bt     = z_residuals.dropna().iloc[WINDOW:]
    if len(z_bt) < 100:
        warnings.warn(
            f"plot_residual_threshold_diagnostics: only {len(z_bt)} backtest residuals "
            f"(< 100); skipping."
        )
        return

    z_losses = -z_bt.values
    n        = len(z_losses)

    # MEP over quantile grid
    u_quantiles = np.linspace(0.50, 0.98, 200)
    u_grid      = np.quantile(z_losses, u_quantiles)
    mean_excess = []
    for u in u_grid:
        exc = z_losses[z_losses > u] - u
        mean_excess.append(exc.mean() if len(exc) >= 5 else np.nan)
    mean_excess = np.array(mean_excess)
    u_z         = np.quantile(z_losses, THRESHOLD_Q)

    # Hill estimator on strictly positive residual losses
    pos_losses = z_losses[z_losses > 0]
    if len(pos_losses) < 20:
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
    k_chosen = int((pos_losses > u_z).sum())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: MEP
    mask = ~np.isnan(mean_excess)
    axes[0].plot(u_grid[mask], mean_excess[mask],
                 color="#2196F3", linewidth=1.4, label="Mean excess e(u_z)")
    axes[0].axvline(u_z, color="#F44336", linestyle="--", linewidth=1.6,
                    label=f"Chosen u_z = {u_z:.4f}  ({THRESHOLD_Q:.0%} quantile)")
    axes[0].set_xlabel("Threshold u_z  (standardised residual loss)", fontsize=10)
    axes[0].set_ylabel("E[z_loss - u_z | z_loss > u_z]", fontsize=10)
    axes[0].set_title(
        "MEP — Residual Losses  (backtest period)\n"
        "Linear region above u_z supports GPD threshold choice on residuals (Irle p. 215)",
        fontsize=11, fontweight="bold")
    axes[0].legend(fontsize=9)

    # Panel 2: Hill estimator
    if len(hill) > 0:
        axes[1].plot(k_vals, hill, color="#4CAF50", linewidth=1.2,
                     label="xi_Hill(k)  [pos. residual losses only]")
        axes[1].axvline(k_chosen, color="#F44336", linestyle="--", linewidth=1.6,
                        label=f"k at 90th pct ~= {k_chosen}")
        axes[1].axhline(0, color="black", linewidth=0.7, linestyle=":")
    else:
        axes[1].text(0.5, 0.5, "Hill estimator skipped\n(< 20 positive residual losses)",
                     ha="center", va="center", transform=axes[1].transAxes, fontsize=11)
    axes[1].set_xlabel("k  (top-k order statistics)", fontsize=10)
    axes[1].set_ylabel("Hill estimator xi_Hill(k)", fontsize=10)
    axes[1].set_title(
        "Hill Estimator — Residual Losses  [positive losses only]\n"
        "xi > 0 -> heavy tail in GARCH residuals — Irle Section 9",
        fontsize=11, fontweight="bold")
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(OUTPUT_FIGS, "garch_evt_03_residual_threshold_diagnostics.png")
    plt.savefig(path, dpi=150)
    plt.close("all")
    print(f"Residual threshold diagnostics saved -> {path}")


def run_garch_parameter_stability_diagnostics(param_df, label="expanding"):
    """
    C6: Practical parameter-stability diagnostics for GARCH refits.

    This is NOT a formal Nyblom test. It reports the parameter path across
    refits, ranges, maximum absolute step changes, near-integrated count,
    stationarity fallback count, and total optimizer warning count.

    Saves:
      outputs/tables/garch_evt_parameter_path_<label>.csv
      outputs/tables/garch_evt_parameter_stability_<label>.csv
    """
    if param_df.empty:
        warnings.warn(f"run_garch_parameter_stability_diagnostics ({label}): empty param_df.")
        return pd.DataFrame()

    path_csv = os.path.join(OUTPUT_TABLES, f"garch_evt_parameter_path_{label}.csv")
    param_df.to_csv(path_csv, index=False)
    print(f"  C6 parameter path ({label}) saved -> {path_csv}")

    cols_of_interest = [c for c in
                        ["mu", "omega", "alpha1", "beta", "alpha_plus_beta",
                         "raw_mu", "raw_omega", "raw_alpha1", "raw_beta", "raw_alpha_plus_beta",
                         "nu"]
                        if c in param_df.columns]

    summary_rows = []
    for col in cols_of_interest:
        s = param_df[col].dropna()
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
    n_refits = len(param_df)
    # Use raw_alpha_plus_beta (pre-fallback) to count true near-integrated fits.
    # Post-fallback alpha_plus_beta is always < 0.999 by construction (reverted params),
    # so using it would always give near_integrated_count=0 after a fallback fires.
    _raw_ab = param_df.get("raw_alpha_plus_beta",
                           param_df.get("alpha_plus_beta", pd.Series(dtype=float)))
    near_integrated_count     = int((_raw_ab >= 0.999).sum())
    stationarity_fallback_count = int(param_df.get("stationarity_fallback_used", pd.Series(dtype=float)).sum()) \
                                  if "stationarity_fallback_used" in param_df.columns else 0
    optimizer_warning_total   = int(param_df.get("optimizer_warning_count", pd.Series(dtype=float)).sum()) \
                                if "optimizer_warning_count" in param_df.columns else 0

    meta_row = pd.DataFrame([{
        "parameter"          : "_meta_n_refits",
        "mean"               : n_refits,
        "std"                : np.nan,
        "min"                : np.nan,
        "max"                : np.nan,
        "max_abs_step_change": np.nan,
    }, {
        "parameter"          : "_meta_near_integrated_count",
        "mean"               : near_integrated_count,
        "std"                : np.nan,
        "min"                : np.nan,
        "max"                : np.nan,
        "max_abs_step_change": np.nan,
    }, {
        "parameter"          : "_meta_stationarity_fallback_count",
        "mean"               : stationarity_fallback_count,
        "std"                : np.nan,
        "min"                : np.nan,
        "max"                : np.nan,
        "max_abs_step_change": np.nan,
    }, {
        "parameter"          : "_meta_optimizer_warning_total",
        "mean"               : optimizer_warning_total,
        "std"                : np.nan,
        "min"                : np.nan,
        "max"                : np.nan,
        "max_abs_step_change": np.nan,
    }])
    summary_df = pd.concat([summary_df, meta_row], ignore_index=True)

    stab_csv = os.path.join(OUTPUT_TABLES, f"garch_evt_parameter_stability_{label}.csv")
    summary_df.to_csv(stab_csv, index=False)
    print(f"  C6 parameter stability ({label}) saved -> {stab_csv}")

    # Print headline stats
    ab_row = param_df.get("alpha_plus_beta", pd.Series(dtype=float)).dropna()
    if len(ab_row) >= 2:
        ab_step = ab_row.diff().abs().dropna()
        print(f"  C6 ({label}) alpha+beta range    : [{ab_row.min():.5f}, {ab_row.max():.5f}]")
        print(f"  C6 ({label}) max |delta a+b|     : {ab_step.max():.5f}")
    print(f"  C6 ({label}) near-integrated refits: {near_integrated_count}")
    print(f"  C6 ({label}) stationarity fallbacks : {stationarity_fallback_count}")
    print(f"  C6 ({label}) optimizer warnings     : {optimizer_warning_total}")

    return summary_df


def plot_garch_parameter_stability(param_df, label="expanding"):
    """
    C6: Four-panel parameter stability plot across GARCH refits.

    Panel 1: raw_alpha_plus_beta path with 0.999 boundary line
    Panel 2: raw_alpha1 and raw_beta paths
    Panel 3: omega path (post-fallback)
    Panel 4: nu (degrees of freedom) path

    Saved to outputs/figures/garch_evt_04_parameter_stability_<label>.png.
    """
    if param_df.empty:
        warnings.warn(f"plot_garch_parameter_stability ({label}): empty param_df, skipping.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()
    x = np.arange(len(param_df))

    # Panel 1: alpha+beta path (raw pre-fallback)
    col_ab = "raw_alpha_plus_beta" if "raw_alpha_plus_beta" in param_df.columns else "alpha_plus_beta"
    axes[0].plot(x, param_df[col_ab].values, color="#1976D2",
                 linewidth=1.0, marker=".", markersize=3)
    axes[0].axhline(0.999, color="#F44336", linestyle="--", linewidth=1.0,
                    label="0.999 boundary")
    axes[0].set_title("alpha1 + beta (raw pre-fallback)", fontweight="bold")
    axes[0].set_xlabel("Refit index")
    axes[0].set_ylabel("alpha1 + beta")
    axes[0].legend(fontsize=8)

    # Panel 2: alpha1 and beta
    col_a = "raw_alpha1" if "raw_alpha1" in param_df.columns else "alpha1"
    col_b = "raw_beta"   if "raw_beta"   in param_df.columns else "beta"
    axes[1].plot(x, param_df[col_a].values, color="#43A047",
                 linewidth=1.0, marker=".", markersize=3, label="alpha1 (raw)")
    axes[1].plot(x, param_df[col_b].values, color="#7B1FA2",
                 linewidth=1.0, marker=".", markersize=3, label="beta (raw)")
    axes[1].set_title("alpha1 and beta paths (raw pre-fallback)", fontweight="bold")
    axes[1].set_xlabel("Refit index")
    axes[1].set_ylabel("coefficient value")
    axes[1].legend(fontsize=8)

    # Panel 3: omega path (post-fallback)
    if "omega" in param_df.columns:
        axes[2].plot(x, param_df["omega"].values, color="#FF9800",
                     linewidth=1.0, marker=".", markersize=3)
        axes[2].set_title("omega path (post-fallback)", fontweight="bold")
        axes[2].set_xlabel("Refit index")
        axes[2].set_ylabel("omega")

    # Panel 4: nu (degrees of freedom)
    if "nu" in param_df.columns:
        nu_vals = param_df["nu"].values.astype(float)
        valid   = np.isfinite(nu_vals)
        if valid.any():
            axes[3].plot(x[valid], nu_vals[valid], color="#E53935",
                         linewidth=1.0, marker=".", markersize=3)
        axes[3].set_title("nu (degrees of freedom) path", fontweight="bold")
        axes[3].set_xlabel("Refit index")
        axes[3].set_ylabel("nu")

    plt.suptitle(f"GARCH Parameter Stability — {label} window  (C6)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUT_FIGS, f"garch_evt_04_parameter_stability_{label}.png")
    plt.savefig(path, dpi=150)
    plt.close("all")
    print(f"  C6 parameter stability plot ({label}) saved -> {path}")


def run_garch_adequacy_diagnostics(r_p, z_residuals):
    """
    Ljung-Box and ARCH-LM adequacy diagnostics on GARCH residuals (Irle p. 178-181).

    Tests:
      LB on Z_t       — tests for serial correlation in standardised residuals
      LB on Z_t^2     — tests for remaining ARCH effects after GARCH filtering
      Engle ARCH-LM   — direct test for remaining heteroscedasticity (10 lags)
      LB on r_p       — screens for serial correlation that would motivate an AR mean specification

    Results are saved to outputs/tables/garch_evt_diagnostics.csv.
    These are purely diagnostic — they do NOT gate the model.

    Interpretation: p > 0.05 across LB-Z, LB-Z^2, and ARCH-LM means we fail to
    reject the respective nulls (no serial correlation in residuals; no remaining
    ARCH effects). This supports — but does not prove — GARCH adequacy. A formal
    acceptance would require power analysis, which we do not perform.
    """
    z_clean = z_residuals.dropna().values
    r_clean = r_p.dropna().values

    lb_z  = acorr_ljungbox(z_clean,      lags=[10, 20], return_df=True)
    lb_z2 = acorr_ljungbox(z_clean ** 2, lags=[10, 20], return_df=True)
    lb_r  = acorr_ljungbox(r_clean,      lags=[10, 20], return_df=True)

    arch_lm_stat, arch_lm_pval, _, _ = het_arch(z_clean, nlags=10)

    # C5: Engle-Ng sign-bias test (full sample)
    c5_full = engle_ng_sign_bias_test(z_clean, label="c5_full")

    print(f"\nGARCH adequacy diagnostics (Irle p. 178-181):")
    print(f"  Ljung-Box on Z_t    (lag 10): stat={lb_z.iloc[0, 0]:.2f},  p={lb_z.iloc[0, 1]:.4f}")
    print(f"  Ljung-Box on Z_t    (lag 20): stat={lb_z.iloc[1, 0]:.2f},  p={lb_z.iloc[1, 1]:.4f}")
    print(f"  Ljung-Box on Z_t^2  (lag 10): stat={lb_z2.iloc[0, 0]:.2f},  p={lb_z2.iloc[0, 1]:.4f}")
    print(f"  Ljung-Box on Z_t^2  (lag 20): stat={lb_z2.iloc[1, 0]:.2f},  p={lb_z2.iloc[1, 1]:.4f}")
    print(f"  Engle ARCH-LM (10 lags):      stat={arch_lm_stat:.2f},  p={arch_lm_pval:.4f}")
    print(f"  Ljung-Box on r_p    (lag 10): p={lb_r.iloc[0, 1]:.4f}  (raw-return pre-test)")
    print(f"  Ljung-Box on r_p    (lag 20): p={lb_r.iloc[1, 1]:.4f}  (raw-return pre-test)")
    # Hypothesis-testing precision: fail-to-reject is not the same as accept
    print(f"  Interpretation: p > 0.05 means we fail to reject the null of no")
    print(f"    autocorrelation (LB) / no remaining ARCH effects (LM). This")
    print(f"    supports, but does not prove, GARCH adequacy.")
    print(f"\n  C5 Engle-Ng sign-bias (full sample, n={c5_full['c5_full_sign_bias_n']}):")
    print(f"    Sign-bias p             = {c5_full['c5_full_sign_bias_p']:.4f}")
    print(f"    Negative-size-bias p    = {c5_full['c5_full_negative_size_bias_p']:.4f}")
    print(f"    Positive-size-bias p    = {c5_full['c5_full_positive_size_bias_p']:.4f}")
    print(f"    Joint p (c1=c2=c3=0)   = {c5_full['c5_full_joint_sign_bias_p']:.4f}")
    print(f"    Interpretation: p<=0.05 indicates asymmetric news impact not captured by")
    print(f"    symmetric GARCH(1,1); consider GJR-GARCH/EGARCH/APARCH challenger,")
    print(f"    but do not change production model here.")

    diag_df = pd.DataFrame([{
        "lb_z_lag10_stat"  : lb_z.iloc[0, 0],
        "lb_z_lag10_p"     : lb_z.iloc[0, 1],
        "lb_z_lag20_stat"  : lb_z.iloc[1, 0],
        "lb_z_lag20_p"     : lb_z.iloc[1, 1],
        "lb_z2_lag10_stat" : lb_z2.iloc[0, 0],
        "lb_z2_lag10_p"    : lb_z2.iloc[0, 1],
        "lb_z2_lag20_stat" : lb_z2.iloc[1, 0],
        "lb_z2_lag20_p"    : lb_z2.iloc[1, 1],
        "arch_lm_stat"     : arch_lm_stat,
        "arch_lm_p"        : arch_lm_pval,
        "lb_raw_lag10_p"   : lb_r.iloc[0, 1],
        "lb_raw_lag20_p"   : lb_r.iloc[1, 1],
        # C5 full-sample columns
        **c5_full,
    }])

    path = os.path.join(OUTPUT_TABLES, "garch_evt_diagnostics.csv")
    diag_df.to_csv(path, index=False)
    print(f"  Diagnostics saved -> {path}")

    # Also run diagnostics on backtest-period residuals only
    z_bt = z_residuals.dropna().iloc[WINDOW:].values
    if len(z_bt) > 50:
        lb_z_bt  = acorr_ljungbox(z_bt,      lags=[10, 20], return_df=True)
        lb_z2_bt = acorr_ljungbox(z_bt ** 2, lags=[10, 20], return_df=True)
        arch_lm_bt_stat, arch_lm_bt_pval, _, _ = het_arch(z_bt, nlags=10)

        # C5: Engle-Ng sign-bias test (backtest-period residuals)
        c5_bt = engle_ng_sign_bias_test(z_bt, label="c5_bt")

        print(f"\n  Backtest-period residuals only (n={len(z_bt)}):")
        print(f"    LB Z_t   lag10 p = {lb_z_bt.iloc[0, 1]:.4f}")
        print(f"    LB Z_t   lag20 p = {lb_z_bt.iloc[1, 1]:.4f}")
        print(f"    LB Z_t^2 lag10 p = {lb_z2_bt.iloc[0, 1]:.4f}")
        print(f"    LB Z_t^2 lag20 p = {lb_z2_bt.iloc[1, 1]:.4f}")
        print(f"    ARCH-LM  lag10 p = {arch_lm_bt_pval:.4f}")
        print(f"    C5 Sign-bias joint p (bt) = {c5_bt['c5_bt_joint_sign_bias_p']:.4f}")

        diag_df["lb_z_bt_lag10_p"]  = lb_z_bt.iloc[0, 1]
        diag_df["lb_z_bt_lag20_p"]  = lb_z_bt.iloc[1, 1]
        diag_df["lb_z2_bt_lag10_p"] = lb_z2_bt.iloc[0, 1]
        diag_df["lb_z2_bt_lag20_p"] = lb_z2_bt.iloc[1, 1]
        diag_df["arch_lm_bt_p"]     = arch_lm_bt_pval
        for k, v in c5_bt.items():
            diag_df[k] = v

        diag_df.to_csv(path, index=False)

    return diag_df

# =============================================================================
# STEP 4 -- POT ON RESIDUALS (helper) + ROLLING VaR
# =============================================================================

def _pot_var_residuals(z_w, threshold_q, alpha, T_w):
    """
    Apply POT/GPD to the lower tail of a window of standardised residuals.

    Lower-tail losses in residual space: z_losses = -z_w
    Threshold: u_z = quantile(z_losses, threshold_q)
    Fit GPD on exceedances above u_z (dimensionless).

    Validity check at entry: alpha must exceed threshold_q.
    Once alpha is above the empirical threshold probability f_u_hat, the POT tail
    quantile should lie at or above u_z. q_evt_raw < u_z is treated as a degenerate
    fit; note that f_u_hat may differ slightly from threshold_q in finite samples.

    GPD fit failures are logged with warnings.warn.
    GPD shape xi is clamped to [-0.5, 1.0] as a defensive guard, not a
    constrained MLE. xi has not approached either bound in production data.
    KS goodness-of-fit added; ks_pvalue set to NaN when xi is clamped.

    Returns
    -------
    dict with keys:
      q_evt_final   : float  EVT quantile as model output (empirical fallback if fit failed)
      q_evt_raw     : float  In "none" paths: raw GPD quantile before the error guard.
                             In "q_below_threshold": GPD quantile that fell below u_z
                             (q_evt_final uses empirical fallback instead).
                             NaN in other fallback paths (gpd_fit_failed, etc.).
      xi            : float  GPD shape parameter post-clamp (NaN if any fallback used)
      xi_raw        : float  GPD shape pre-clamp (NaN if GPD not fitted)
      xi_clamped    : bool   True if xi was clamped to [-0.5, 1.0]
      ks_pvalue     : float  KS p-value (NaN if fallback or xi_clamped; indicator only)
      fallback_type : str    "none" | "few_exceedances" | "gpd_fit_failed" |
                             "q_below_threshold" | "alpha_below_threshold"
      threshold_u   : float  u_z used (threshold_q quantile of z_losses)
      N_u           : int    exceedances above u_z in this window
      f_u_hat       : float  empirical CDF at u_z = 1 - N_u/T_w

    Reference: Irle p. 223-225 applied to Z_t; McNeil & Frey (2000) Eq. 4
    """
    # POT formula requires alpha > threshold_q
    if alpha <= threshold_q:
        raise ValueError(
            f"POT VaR requires alpha > threshold_q; got alpha={alpha}, "
            f"threshold_q={threshold_q}"
        )

    z_losses    = -z_w
    u_z         = np.quantile(z_losses, threshold_q)
    exceedances = z_losses[z_losses > u_z] - u_z
    N_u         = len(exceedances)
    f_u_hat     = 1.0 - N_u / T_w   # empirical CDF at threshold

    # alpha_below_threshold: target quantile within empirical range; expected 0 in production
    if alpha <= f_u_hat:
        fb = np.quantile(z_losses, alpha)
        return {
            "q_evt_final": fb, "q_evt_raw": fb,
            "xi": np.nan, "xi_raw": np.nan, "xi_clamped": False, "ks_pvalue": np.nan,
            "fallback_type": "alpha_below_threshold",
            "threshold_u": u_z, "N_u": N_u, "f_u_hat": f_u_hat,
        }

    if N_u < MIN_EXCEEDANCES:
        fb = np.quantile(z_losses, alpha)
        return {
            "q_evt_final": fb, "q_evt_raw": fb,
            "xi": np.nan, "xi_raw": np.nan, "xi_clamped": False, "ks_pvalue": np.nan,
            "fallback_type": "few_exceedances",
            "threshold_u": u_z, "N_u": N_u, "f_u_hat": f_u_hat,
        }

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            c, _loc, sigma = genpareto.fit(exceedances, floc=0)

        # Finite guard on sigma — non-finite or non-positive sigma signals a failed fit
        if not np.isfinite(sigma) or sigma <= 0:
            warnings.warn(
                f"GPD fit returned non-finite or non-positive sigma={sigma}; "
                "using empirical quantile fallback."
            )
            fb = np.quantile(z_losses, alpha)
            return {
                "q_evt_final": fb, "q_evt_raw": np.nan,
                "xi": np.nan, "xi_raw": np.nan, "xi_clamped": False, "ks_pvalue": np.nan,
                "fallback_type": "gpd_fit_failed",
                "threshold_u": u_z, "N_u": N_u, "f_u_hat": f_u_hat,
            }

        # Record xi_raw before any clamping
        xi_raw     = c
        xi         = c
        xi_clamped = False

        # Clamp xi to [-0.5, 1.0] as a defensive VaR guardrail, not a constrained MLE.
        # At xi=1.0 the GPD mean is not finite, but VaR (a quantile) remains
        # well-defined for alpha < 1. ks_pvalue is set to NaN when xi is clamped.
        if xi > 1.0:
            warnings.warn(
                f"GPD shape xi={xi:.4f} exceeds upper guardrail 1.0 "
                f"(GPD mean undefined at xi>=1); clamped to 1.0 as a VaR guardrail."
            )
            xi         = 1.0
            xi_clamped = True
        elif xi < -0.5:
            warnings.warn(
                f"GPD shape xi={xi:.4f} below lower guardrail -0.5 "
                f"(implausibly bounded tail); clamped to -0.5 as a VaR guardrail."
            )
            xi         = -0.5
            xi_clamped = True

        ratio = (T_w / N_u) * (1.0 - alpha)

        if abs(xi) < 1e-4:
            # Gumbel limit: avoids division by near-zero xi
            q_evt_raw = u_z - sigma * np.log(ratio)
        else:
            q_evt_raw = u_z + (sigma / xi) * (ratio ** (-xi) - 1.0)

        # Guard against non-finite q_evt_raw (numerical instability in GPD formula)
        if not np.isfinite(q_evt_raw):
            warnings.warn(
                f"GPD VaR quantile is non-finite (xi={xi:.4f}, sigma={sigma:.4e}); "
                "using empirical quantile fallback."
            )
            fb = np.quantile(z_losses, alpha)
            return {
                "q_evt_final": fb, "q_evt_raw": np.nan,
                "xi": np.nan, "xi_raw": xi_raw, "xi_clamped": xi_clamped, "ks_pvalue": np.nan,
                "fallback_type": "gpd_fit_failed",
                "threshold_u": u_z, "N_u": N_u, "f_u_hat": f_u_hat,
            }

        # Error guard: q_evt_raw < u_z with alpha > f_u_hat is treated as a degenerate fit.
        if q_evt_raw < u_z:
            warnings.warn(
                f"GPD VaR quantile {q_evt_raw:.4f} below threshold u_z={u_z:.4f} "
                f"(xi={xi:.4f}, sigma={sigma:.4e}). This indicates a fit problem; "
                f"using empirical quantile fallback."
            )
            fallback = np.quantile(z_losses, alpha)
            return {
                "q_evt_final": fallback, "q_evt_raw": q_evt_raw,
                "xi": np.nan, "xi_raw": xi_raw, "xi_clamped": xi_clamped, "ks_pvalue": np.nan,
                "fallback_type": "q_below_threshold",
                "threshold_u": u_z, "N_u": N_u, "f_u_hat": f_u_hat,
            }

        q_evt_final = q_evt_raw   # no flooring applied; if we reach here, raw is valid

        # KS test invalid when xi was clamped — the clamped xi does not represent the fitted GPD
        if xi_clamped:
            ks_pvalue = np.nan
        else:
            _, ks_pvalue = kstest(exceedances, "genpareto", args=(xi, 0, sigma))

        return {
            "q_evt_final": q_evt_final, "q_evt_raw": q_evt_raw,
            "xi": xi, "xi_raw": xi_raw, "xi_clamped": xi_clamped, "ks_pvalue": ks_pvalue,
            "fallback_type": "none",
            "threshold_u": u_z, "N_u": N_u, "f_u_hat": f_u_hat,
        }

    except Exception as exc:
        # Audit trail for GPD fit failures
        warnings.warn(
            f"GPD fit on residuals failed: {type(exc).__name__}: {exc}. "
            "Using empirical quantile fallback."
        )
        fb = np.quantile(z_losses, alpha)
        return {
            "q_evt_final": fb, "q_evt_raw": fb,
            "xi": np.nan, "xi_raw": np.nan, "xi_clamped": False, "ks_pvalue": np.nan,
            "fallback_type": "gpd_fit_failed",
            "threshold_u": u_z, "N_u": N_u, "f_u_hat": f_u_hat,
        }


VALID_FALLBACK_TYPES = frozenset({
    "none", "few_exceedances", "gpd_fit_failed",
    "q_below_threshold", "alpha_below_threshold",
})


def _validate_garch_evt_results(results):
    """
    Sanity checks on compute_garch_evt_var output.
    Called before returning; raises ValueError on any violation.
    NaN is allowed in: xi, xi_raw, xi_clamped, ks_pvalue, VaR_GARCH_EVT_raw
    (VaR_GARCH_EVT_raw is NaN in gpd_fit_failed paths where q_evt_raw is unavailable;
    this is expected and does not indicate a bug).
    """
    errs = []

    if (results["VaR_GARCH_EVT"] < 0).any():
        errs.append(f"VaR_GARCH_EVT: {(results['VaR_GARCH_EVT'] < 0).sum()} negative values")

    required_finite = ["VaR_GARCH_EVT", "sigma_hat", "mu_hat",
                       "q_EVT", "threshold_u", "N_u", "f_u_hat"]
    for col in required_finite:
        if col in results.columns:
            n_bad = (~np.isfinite(results[col].values.astype(float))).sum()
            if n_bad > 0:
                errs.append(f"{col}: {n_bad} non-finite values")

    if (results["sigma_hat"] <= 0).any():
        errs.append(f"sigma_hat: {(results['sigma_hat'] <= 0).sum()} non-positive values")

    if "f_u_hat" in results.columns:
        fu = results["f_u_hat"].dropna()
        n_bad = ((fu <= 0) | (fu >= 1)).sum()
        if n_bad > 0:
            errs.append(f"f_u_hat: {n_bad} values outside (0, 1)")

    if "fallback_type" in results.columns and "q_EVT" in results.columns:
        none_mask = results["fallback_type"] == "none"
        if none_mask.any():
            q_below = results.loc[none_mask, "q_EVT"] < results.loc[none_mask, "threshold_u"]
            if q_below.any():
                errs.append(f"fallback_type='none' but q_EVT < threshold_u: {q_below.sum()} rows")

    if errs:
        raise ValueError(
            "_validate_garch_evt_results failed:\n" +
            "\n".join(f"  - {e}" for e in errs)
        )


def compute_garch_evt_var(pnl, cond_vol, z_residuals, mu_series,
                          window=WINDOW, threshold_q=THRESHOLD_Q,
                          alpha=ALPHA, V0=V0):
    """
    Rolling 500-day GARCH(1,1)+EVT conditional VaR (McNeil & Frey 2000).

    For each day t in [window, T):
      (A) sigma_t = cond_vol[t]            GARCH conditional vol for day t
                                            (determined by data up to t-1; no look-ahead)
          mu_t    = mu_series[t]            GARCH conditional mean for day t
      (B) z_w = z_residuals[t-window:t]   window of standardised residuals
          q_EVT = POT quantile of lower tail of z_w  (dimensionless)
      (C) VaR_t = V0 * (-mu_t + sigma_t * q_EVT)    (USD)
          (McNeil & Frey 2000 Eq. 4: loss L_t = -V0*mu_t - V0*sigma_t*Z_t,
          so the conditional mean enters with a sign flip)

    cond_vol[t] = sigma_t for day t, computed from data up to t-1.
    VaR floored at 0 (negative VaR is numerically nonsensical; guard only).

    Returns
    -------
    pd.DataFrame  columns: VaR_GARCH_EVT, VaR_GARCH_EVT_raw, sigma_hat,
                           mu_hat, q_EVT, xi, ks_pvalue, fallback_type,
                           threshold_u, N_u, f_u_hat
      VaR_GARCH_EVT     — policy output (error-guard + floor at 0)
      VaR_GARCH_EVT_raw — V0 * (-mu_t + sigma_t * q_evt_raw) before the error guard
                          and non-negativity floor (diagnostic). NaN in gpd_fit_failed
                          and other paths where q_evt_raw is unavailable; not the
                          policy output (use VaR_GARCH_EVT for that).
      threshold_u, N_u, f_u_hat — POT diagnostic fields from each window
    """
    # The three series must agree exactly — fail loudly if indices differ.
    if not cond_vol.index.equals(z_residuals.index):
        raise ValueError("cond_vol and z_residuals indices are not identical")
    if not cond_vol.index.equals(mu_series.index):
        raise ValueError("cond_vol and mu_series indices are not identical")
    common    = cond_vol.index
    vol_vals  = cond_vol.values
    z_vals    = z_residuals.values
    mu_vals   = mu_series.values
    dates_all = common
    n         = len(common)

    print(f"\n{'='*60}")
    print(f"GARCH(1,1)+EVT VaR  --  McNeil & Frey (2000)")
    print(f"window={window} | alpha={alpha:.0%} | "
          f"threshold={threshold_q:.0%} on residuals | refit every {REFIT_EVERY} days")
    print(f"VaR formula: V0 * (-mu_t + sigma_t * q_EVT)")
    print(f"Computing conditional VaR for {n - window} days ...")

    records, dates = [], []
    n_fallbacks = 0

    for t in range(window, n):
        sigma_hat_t = vol_vals[t]               # sigma_t for day t (decimal)
        mu_t        = mu_vals[t]                # conditional mean for day t
        z_w         = z_vals[t - window : t]    # window of standardised residuals

        # Handle any NaN in z_w (edge at start of series)
        z_w_clean = z_w[~np.isnan(z_w)]

        out = _pot_var_residuals(
            z_w_clean, threshold_q, alpha, len(z_w_clean)
        )
        q_evt     = out["q_evt_final"]
        xi_t      = out["xi"]
        ks_pval   = out["ks_pvalue"]
        q_evt_raw = out["q_evt_raw"]
        fb_type   = out["fallback_type"]
        if fb_type != "none":
            n_fallbacks += 1

        # VaR formula: McNeil & Frey (2000) Eq. 4
        var_t     = V0 * (-mu_t + sigma_hat_t * q_evt)
        var_t     = max(var_t, 0.0)                                  # floor at 0
        var_raw_t = V0 * (-mu_t + sigma_hat_t * q_evt_raw)          # unfloored diagnostic

        records.append({
            "VaR_GARCH_EVT"     : var_t,
            "VaR_GARCH_EVT_raw" : var_raw_t,
            "sigma_hat"         : sigma_hat_t,
            "mu_hat"            : mu_t,
            "q_EVT"             : q_evt,
            "xi"                : xi_t,
            "xi_raw"            : out["xi_raw"],
            "xi_clamped"        : out["xi_clamped"],
            "ks_pvalue"         : ks_pval,
            "fallback_type"     : fb_type,
            "threshold_u"       : out["threshold_u"],
            "N_u"               : out["N_u"],
            "f_u_hat"           : out["f_u_hat"],
        })
        dates.append(dates_all[t])

    results = pd.DataFrame(records, index=dates)

    # Enum validation — catch any unexpected fallback_type values immediately
    unexpected_fb = set(results["fallback_type"].unique()) - VALID_FALLBACK_TYPES
    if unexpected_fb:
        raise ValueError(f"Unexpected fallback_type values: {unexpected_fb}")

    # Result sanity checks before returning
    _validate_garch_evt_results(results)

    xi_ok   = results["xi"].dropna()

    n_poor_fit = int((results["ks_pvalue"] < 0.05).sum())
    n_bt       = n - window

    print(f"\nRolling GARCH+EVT summary:")
    print(f"  Mean VaR      : ${results['VaR_GARCH_EVT'].mean():>12,.0f}")
    print(f"  Min  VaR      : ${results['VaR_GARCH_EVT'].min():>12,.0f}")
    print(f"  Max  VaR      : ${results['VaR_GARCH_EVT'].max():>12,.0f}")
    print(f"  Mean sigma_hat: {results['sigma_hat'].mean():.6f}  (decimal return units)")
    print(f"  Mean mu_hat   : {results['mu_hat'].mean():.8f}  (decimal return units)")
    print(f"  Mean q_EVT    : {results['q_EVT'].mean():.4f}  (dimensionless residual quantile)")
    print(f"  Mean xi       :  {xi_ok.mean():.4f}  (>0 -> heavy tail confirmed)")
    print(f"  Fallback days : {n_fallbacks} / {n_bt}")
    fb_counts = results["fallback_type"].value_counts()
    print(f"  Fallback breakdown:")
    for fb_name in ("none", "few_exceedances", "gpd_fit_failed",
                    "q_below_threshold", "alpha_below_threshold"):
        print(f"    {fb_name:>22}: {fb_counts.get(fb_name, 0)}")
    n_xi_clamped = int(results["xi_clamped"].sum()) if "xi_clamped" in results.columns else 0
    print(f"  xi clamped    : {n_xi_clamped} / {n_bt}  (ks_pvalue=NaN when clamped)")
    print(f"  Poor GPD fit  : {n_poor_fit} / {n_bt}  (KS p<0.05 -- "
          "anti-conservative, relative indicator only)")

    return results

# =============================================================================
# STEP 5 -- BACKTEST
# =============================================================================

def _assert_results_aligned_to_pnl(pnl, results, context=""):
    """
    Date alignment guard: raises ValueError if any results date is missing from pnl.

    results.index must be a subset of pnl.index (results starts at WINDOW; pnl starts
    earlier). Replaces silent intersection() calls that can mask misalignment bugs.
    """
    missing = results.index.difference(pnl.index)
    if len(missing) > 0:
        raise ValueError(
            f"{context}: {len(missing)} result dates not found in pnl index. "
            f"First few: {missing[:5].tolist()}"
        )


def backtest_garch_evt(pnl, results):
    """
    Kupiec + Christoffersen backtests via shared framework.
    (Irle Chapter 8, p. 183-185)
    """
    _assert_results_aligned_to_pnl(pnl, results, context="backtest_garch_evt")
    bt = run_backtest(
        pnl=pnl,
        var=results["VaR_GARCH_EVT"],
        confidence=ALPHA,
        method_name="GARCH+EVT",
    )
    print(bt)
    return bt

# =============================================================================
# STEP 6 -- PLOTS
# =============================================================================

def plot_garch_evt_results(pnl, results, bt):
    """
    Two-panel figure:
      Panel 1: GARCH+EVT conditional VaR vs actual loss with exceptions.
      Panel 2: Rolling sigma_t (GARCH conditional vol) showing
               VaR = V0 * (-mu_t + sigma_t * q_EVT)

    The GARCH component makes VaR spike sharply during crises — unlike static
    EVT — illustrating the key advantage of the conditional approach.
    (McNeil & Frey 2000: GARCH+EVT dominates unconditional EVT in backtests)
    """
    crises = [
        ("2008-09-15", "2009-03-09", "GFC 2008"),
        ("2020-02-19", "2020-03-23", "COVID 2020"),
        ("2022-01-01", "2022-10-01", "Rate Hikes 2022"),
    ]

    _assert_results_aligned_to_pnl(pnl, results, context="plot_garch_evt_results")
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
                 linewidth=0.6, alpha=0.7, label="Actual loss (-DeltaV)")
    if exc_idx is not None and len(exc_idx) > 0:
        axes[0].scatter(exc_idx, actual_loss.loc[exc_idx],
                        color="#F44336", s=20, zorder=5,
                        label=f"Exceptions  N={bt.N}  ({bt.exception_rate:.2%})")
    axes[0].axhline(0, color="black", linewidth=0.5, linestyle="--")
    axes[0].set_title(
        "GARCH(1,1)+EVT Conditional VaR 99% vs Actual Portfolio Loss\n"
        "VaR_t = V0*(-mu_t + sigma_t*q_EVT(alpha))   |   McNeil & Frey (2000) Eq. 4",
        fontsize=12, fontweight="bold")
    axes[0].set_ylabel("USD")
    axes[0].legend(fontsize=9)

    # Panel 2: GARCH conditional vol sigma_t
    vol_pct = results["sigma_hat"] * np.sqrt(252) * 100   # annualised %
    axes[1].plot(results.index, vol_pct, color="#7B1FA2",
                 linewidth=0.8, label="sigma_t annualised (%)")
    axes[1].set_title(
        "GARCH Conditional Vol sigma_t  |  "
        "Expanding-window re-estimation every 50 days  |  Irle Eq. 17",
        fontsize=12, fontweight="bold")
    axes[1].set_ylabel("sigma_t  (%/year)")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[1].legend(fontsize=9)

    for ax in axes:
        first_crisis = True
        for s, e, lbl in crises:
            kwargs = {"alpha": 0.07, "color": "red"}
            if first_crisis:
                kwargs["label"] = lbl
                first_crisis = False
            ax.axvspan(s, e, **kwargs)

    plt.tight_layout()
    path = os.path.join(OUTPUT_FIGS, "garch_evt_02_var_results.png")
    plt.savefig(path, dpi=150)
    plt.close("all")
    print(f"GARCH+EVT results plot saved -> {path}")

# =============================================================================
# STEP 7 -- SAVE RESULTS
# =============================================================================

def save_results(pnl, results, bt):
    """
    Save VaR time series and backtest detail table.

    Convention matches delta_normal.py and evt.py. Columns in backtest CSV:
      VaR, VaR_raw, actual_loss, exception, sigma_hat, mu_hat, q_EVT, xi,
      xi_raw, xi_clamped, ks_pvalue, fallback_type, threshold_u, N_u, f_u_hat.

    VaR_raw  — V0*(-mu_t + sigma_t*q_EVT_raw) before error-guard (diagnostic).
    mu_hat   — GARCH conditional mean used in the VaR formula.
    ks_pvalue — KS p-value for GPD fit quality (anti-conservative; relative indicator).
    """
    _assert_results_aligned_to_pnl(pnl, results, context="save_results")
    results.to_csv(os.path.join(PROCESSED_DIR, "var_garch_evt.csv"))
    print(f"VaR saved     -> {os.path.join(PROCESSED_DIR, 'var_garch_evt.csv')}")

    loss_aligned = -pnl.reindex(results.index)
    pd.DataFrame({
        "VaR"           : results["VaR_GARCH_EVT"].values,
        "VaR_raw"       : results["VaR_GARCH_EVT_raw"].values,
        "actual_loss"   : loss_aligned.values,
        "exception"     : (loss_aligned > results["VaR_GARCH_EVT"]).astype(int).values,
        "sigma_hat"     : results["sigma_hat"].values,
        "mu_hat"        : results["mu_hat"].values,
        "q_EVT"         : results["q_EVT"].values,
        "xi"            : results["xi"].values,
        "xi_raw"        : results["xi_raw"].values,
        "xi_clamped"    : results["xi_clamped"].values,
        "ks_pvalue"     : results["ks_pvalue"].values,
        "fallback_type" : results["fallback_type"].values,
        "threshold_u"   : results["threshold_u"].values,
        "N_u"           : results["N_u"].values,
        "f_u_hat"       : results["f_u_hat"].values,
    }, index=results.index).to_csv(
        os.path.join(OUTPUT_TABLES, "backtest_garch_evt.csv")
    )
    print(f"Backtest      -> {os.path.join(OUTPUT_TABLES, 'backtest_garch_evt.csv')}")

# =============================================================================
# THRESHOLD SENSITIVITY
# =============================================================================

def threshold_sensitivity(pnl, cond_vol, z_residuals, mu_series,
                           thresholds=(0.85, 0.90, 0.925, 0.95),
                           window=WINDOW, alpha=ALPHA, V0=V0):
    """
    Re-run the EVT step at alternative threshold_q values with GARCH fixed.

    cond_vol and z_residuals are unchanged across threshold values. Addresses
    the sensitivity-roadmap entry for assumption A.2 in the appendix.

    For each threshold_q:
      - compute_garch_evt_var() with that threshold
      - run_backtest()
      - collect: mean VaR, exceptions, Kupiec p, Christoffersen p, mean xi, fallback count

    Saves results to outputs/tables/garch_evt_threshold_sensitivity.csv.
    """
    print(f"\n{'='*60}")
    print(f"Threshold sensitivity sweep (GARCH fixed)")
    print(f"  Thresholds tested: {thresholds}")

    rows = []
    for tq in thresholds:
        res_tq = compute_garch_evt_var(
            pnl, cond_vol, z_residuals, mu_series,
            window=window, threshold_q=tq, alpha=alpha, V0=V0
        )
        _assert_results_aligned_to_pnl(pnl, res_tq, context=f"threshold_sensitivity(tq={tq})")
        bt_tq = run_backtest(
            pnl=pnl,
            var=res_tq["VaR_GARCH_EVT"],
            confidence=alpha,
            method_name=f"GARCH+EVT (tq={tq:.3f})",
        )
        xi_ok      = res_tq["xi"].dropna()
        fb_counts  = res_tq["fallback_type"].value_counts() if "fallback_type" in res_tq.columns else {}
        n_poor_ks  = int((res_tq["ks_pvalue"] < 0.05).sum()) if "ks_pvalue" in res_tq.columns else 0
        rows.append({
            "threshold_q"          : tq,
            "mean_VaR"             : res_tq["VaR_GARCH_EVT"].mean(),
            "exceptions"           : bt_tq.N,
            "exception_rate"       : bt_tq.exception_rate,
            "kupiec_p"             : bt_tq.pvalue_uc,
            "christoffersen_p"     : getattr(bt_tq, "pvalue_cc", np.nan),
            "mean_xi"              : xi_ok.mean() if len(xi_ok) > 0 else np.nan,
            "fallback_days"        : int(res_tq["xi"].isna().sum()),
            "n_q_below_threshold"  : int(fb_counts.get("q_below_threshold", 0)),
            "n_alpha_below_threshold": int(fb_counts.get("alpha_below_threshold", 0)),
            "n_few_exceedances"    : int(fb_counts.get("few_exceedances", 0)),
            "n_poor_ks"            : n_poor_ks,
        })

    sens_df = pd.DataFrame(rows)
    path = os.path.join(OUTPUT_TABLES, "garch_evt_threshold_sensitivity.csv")
    sens_df.to_csv(path, index=False)

    print(f"\n  Threshold sensitivity summary:")
    print(f"  {'threshold_q':>12}  {'mean_VaR':>12}  {'exceptions':>10}  "
          f"{'kupiec_p':>10}  {'chris_p':>10}  {'mean_xi':>8}")
    for row in rows:
        print(f"  {row['threshold_q']:>12.3f}  ${row['mean_VaR']:>11,.0f}  "
              f"{row['exceptions']:>10}  {row['kupiec_p']:>10.4f}  "
              f"{row['christoffersen_p']:>10.4f}  {row['mean_xi']:>8.4f}")
    print(f"  Saved -> {path}")
    return sens_df

# =============================================================================
# ROLLING-WINDOW GARCH (CHALLENGER)
# =============================================================================

def fit_garch_rolling(r_p, window=WINDOW, refit_every=REFIT_EVERY):
    """
    Rolling-window GARCH(1,1) challenger, mirroring fit_garch_expanding()
    but using r_p.iloc[t-window:t] instead of r_p.iloc[:t].

    All other logic is identical: Student-t, scaling convention, residual
    coherence fix at refit boundaries, stationarity guard, warning capture.

    Returns 6-tuple (same as fit_garch_expanding), with param_df as 6th value.
    """
    n        = len(r_p)
    r_vals   = r_p.values
    dates    = r_p.index

    # Index validation
    assert r_p.index.is_monotonic_increasing, "r_p index must be sorted ascending"
    assert r_p.index.is_unique,                 "r_p index contains duplicate dates"

    scale_factor  = 100.0 if r_p.std() < 0.1 else 1.0
    scale_factor2 = scale_factor ** 2

    cond_vol_arr = np.full(n, np.nan)
    z_arr        = np.full(n, np.nan)
    mu_arr       = np.full(n, np.nan)

    mu_hat   = None
    omega_d  = None
    alpha1_d = None
    beta_d   = None
    sigma_sq_curr = np.nan

    last_valid_params        = None
    n_stationarity_fallbacks = 0
    n_warnings_total         = 0
    refit_warnings           = []

    n_refits          = 0
    refit_dates       = []
    last_res          = None
    parameter_records = []   # C6: parameter path for stability diagnostics

    print(f"\n{'='*60}")
    print(f"GARCH(1,1) ROLLING-window fit  --  challenger")
    print(f"dist=Student-t | window={window} | refit every {refit_every} days | "
          f"scale_factor={scale_factor:.0f}")

    for t in range(window, n):
        do_refit = (mu_hat is None) or ((t - window) % refit_every == 0)
        mu_hat_prev = mu_hat

        if do_refit:
            train = r_p.iloc[t - window : t] * scale_factor   # ROLLING window
            garch = arch_model(train, vol="Garch", p=1, q=1,
                               dist="t", mean="Constant")

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                res_t = garch.fit(disp="off")

            if caught:
                n_warnings_total += len(caught)
                refit_warnings.append({
                    "refit_idx" : n_refits,
                    "date"      : dates[t],
                    "n_warnings": len(caught),
                    "messages"  : [str(w.message) for w in caught],
                })

            mu_hat   = res_t.params["mu"]      / scale_factor
            omega_d  = res_t.params["omega"]   / scale_factor2
            alpha1_d = res_t.params["alpha[1]"]
            beta_d   = res_t.params["beta[1]"]

            # Capture raw (pre-fallback) parameters before stationarity guard may overwrite
            raw_mu     = mu_hat
            raw_omega  = omega_d
            raw_alpha1 = alpha1_d
            raw_beta   = beta_d

            fcast = res_t.forecast(horizon=1, reindex=False)
            sigma_sq_curr   = fcast.variance.iloc[-1, 0] / scale_factor2
            cond_vol_arr[t] = np.sqrt(max(sigma_sq_curr, 1e-10))

            ab = alpha1_d + beta_d   # raw pre-fallback; used as raw_alpha_plus_beta below
            if ab >= 0.999:
                if last_valid_params is None:
                    raise RuntimeError(
                        f"[rolling] Initial GARCH fit is near-integrated at t={t}: "
                        f"alpha+beta={ab:.6f}. No previous valid parameters available; "
                        f"cannot proceed."
                    )
                warnings.warn(
                    f"[rolling] Near-integrated GARCH at t={t}: alpha+beta={ab:.6f}. "
                    f"Reverting to previous valid parameters and recomputing cond_vol_arr[t] "
                    f"under the reverted regime."
                )
                mu_hat   = last_valid_params["mu"]
                omega_d  = last_valid_params["omega"]
                alpha1_d = last_valid_params["alpha1"]
                beta_d   = last_valid_params["beta"]

                # Recompute cond_vol_arr[t] under reverted parameters
                prev_sigma_sq = (cond_vol_arr[t - 1] ** 2
                                 if not np.isnan(cond_vol_arr[t - 1])
                                 else omega_d / max(1.0 - alpha1_d - beta_d, 1e-6))
                innov_prev    = r_vals[t - 1] - mu_hat
                sigma_sq_curr = (omega_d
                                 + alpha1_d * innov_prev ** 2
                                 + beta_d   * prev_sigma_sq)
                sigma_sq_curr   = max(sigma_sq_curr, 1e-10)
                cond_vol_arr[t] = np.sqrt(sigma_sq_curr)

                n_stationarity_fallbacks += 1
            else:
                last_valid_params = {
                    "mu"    : mu_hat,
                    "omega" : omega_d,
                    "alpha1": alpha1_d,
                    "beta"  : beta_d,
                }

            # C6: record both raw (pre-fallback) and post-fallback parameters
            parameter_records.append({
                "refit_idx"               : n_refits,
                "date"                    : dates[t],
                "t"                       : t,
                "mu"                      : mu_hat,          # POST-fallback
                "omega"                   : omega_d,          # POST-fallback
                "alpha1"                  : alpha1_d,         # POST-fallback
                "beta"                    : beta_d,           # POST-fallback
                "alpha_plus_beta"         : alpha1_d + beta_d,  # POST-fallback
                "raw_mu"                  : raw_mu,
                "raw_omega"               : raw_omega,
                "raw_alpha1"              : raw_alpha1,
                "raw_beta"                : raw_beta,
                "raw_alpha_plus_beta"     : ab,              # = raw_alpha1 + raw_beta (pre-fallback)
                "nu"                      : res_t.params.get("nu", np.nan),
                "loglikelihood"           : getattr(res_t, "loglikelihood", np.nan),
                "convergence_flag"        : getattr(res_t, "convergence_flag", np.nan),
                "optimizer_warning_count" : len(caught),
                "stationarity_fallback_used": int(ab >= 0.999),
            })

            if n_refits == 0:
                cv_d = res_t.conditional_volatility.values / scale_factor
                z_arr[:t] = np.where(cv_d > 1e-10, (r_vals[:t] - mu_hat) / cv_d, np.nan)
                mu_arr[:t] = mu_hat

            refit_dates.append(dates[t])
            n_refits += 1
            last_res  = res_t

        else:
            innov         = r_vals[t - 1] - mu_hat
            sigma_sq_curr = (omega_d
                             + alpha1_d * innov ** 2
                             + beta_d   * sigma_sq_curr)
            sigma_sq_curr   = max(sigma_sq_curr, 1e-10)
            cond_vol_arr[t] = np.sqrt(sigma_sq_curr)

        mu_arr[t] = mu_hat

        if t > window and mu_hat_prev is not None and cond_vol_arr[t - 1] > 1e-10:
            z_arr[t - 1] = (r_vals[t - 1] - mu_hat_prev) / cond_vol_arr[t - 1]

    cond_vol    = pd.Series(cond_vol_arr, index=dates, name="cond_vol")
    z_residuals = pd.Series(z_arr,        index=dates, name="z_residuals")
    mu_series   = pd.Series(mu_arr,       index=dates, name="mu_hat")

    z_ok = z_residuals.dropna()
    print(f"\nRolling-window GARCH summary:")
    print(f"  Total refits              : {n_refits}")
    print(f"  z mean (window+)          : {z_ok.iloc[window:].mean():.4f}")
    print(f"  z std  (window+)          : {z_ok.iloc[window:].std():.4f}")
    print(f"  Stationarity fallbacks    : {n_stationarity_fallbacks}  (expected 0)")
    print(f"  Total optimizer warnings  : {n_warnings_total}")

    # Persist captured optimizer warnings (challenger; separate file)
    if refit_warnings:
        pd.DataFrame(refit_warnings).to_csv(
            os.path.join(OUTPUT_TABLES, "garch_evt_refit_warnings_rolling.csv"),
            index=False
        )
        print(f"  Refit warnings saved -> "
              f"{os.path.join(OUTPUT_TABLES, 'garch_evt_refit_warnings_rolling.csv')}")
    else:
        print(f"  No optimizer warnings during any refit (clean run).")

    param_df = pd.DataFrame(parameter_records) if parameter_records else pd.DataFrame()   # C6
    return cond_vol, z_residuals, mu_series, refit_dates, last_res, param_df


def run_challenger(r_p, pnl):
    """
    Run rolling-window GARCH challenger pipeline and print side-by-side
    comparison with the production (expanding-window) results.
    Saves rolling results to separate output files.
    """
    print(f"\n{'='*60}")
    print(f"Running rolling-window GARCH challenger pipeline ...")

    cond_vol_r, z_res_r, mu_ser_r, refit_dates_r, _, param_df_r = fit_garch_rolling(r_p)  # C6
    results_r = compute_garch_evt_var(pnl, cond_vol_r, z_res_r, mu_ser_r)
    bt_r = run_backtest(
        pnl=pnl,
        var=results_r["VaR_GARCH_EVT"],
        confidence=ALPHA,
        method_name="GARCH+EVT rolling",
    )

    _assert_results_aligned_to_pnl(pnl, results_r, context="run_challenger")
    results_r.to_csv(os.path.join(PROCESSED_DIR, "var_garch_evt_rolling.csv"))
    loss_r = -pnl.reindex(results_r.index)
    pd.DataFrame({
        "VaR"           : results_r["VaR_GARCH_EVT"].values,
        "VaR_raw"       : results_r["VaR_GARCH_EVT_raw"].values,
        "actual_loss"   : loss_r.values,
        "exception"     : (loss_r > results_r["VaR_GARCH_EVT"]).astype(int).values,
        "sigma_hat"     : results_r["sigma_hat"].values,
        "mu_hat"        : results_r["mu_hat"].values,
        "q_EVT"         : results_r["q_EVT"].values,
        "xi"            : results_r["xi"].values,
        "xi_raw"        : results_r["xi_raw"].values,
        "xi_clamped"    : results_r["xi_clamped"].values,
        "ks_pvalue"     : results_r["ks_pvalue"].values,
        "fallback_type" : results_r["fallback_type"].values,
        "threshold_u"   : results_r["threshold_u"].values,
        "N_u"           : results_r["N_u"].values,
        "f_u_hat"       : results_r["f_u_hat"].values,
    }, index=results_r.index).to_csv(
        os.path.join(OUTPUT_TABLES, "backtest_garch_evt_rolling.csv")
    )

    # C6: parameter stability for rolling challenger
    print(f"\n  C6 parameter stability for rolling challenger:")
    run_garch_parameter_stability_diagnostics(param_df_r, label="rolling")
    plot_garch_parameter_stability(param_df_r, label="rolling")
    print(f"  Rolling results saved.")

    # Side-by-side comparison (requires production results already printed above)
    print(f"\n  Side-by-side comparison (expanding vs rolling GARCH):")
    print(f"  {'Metric':<30} {'Expanding':>14} {'Rolling':>14}")
    print(f"  {'-'*60}")

    def _safe_cc(bt):
        return getattr(bt, "pvalue_cc", np.nan)

    # Production stats are only available if called from main after the prod run
    print(f"  (Rolling stats — compare to production run above)")
    print(f"  {'Mean VaR':<30} {'':>14} ${results_r['VaR_GARCH_EVT'].mean():>13,.0f}")
    print(f"  {'Exceptions':<30} {'':>14} {bt_r.N:>14}")
    print(f"  {'Kupiec p':<30} {'':>14} {bt_r.pvalue_uc:>14.4f}")
    print(f"  {'Christoffersen p':<30} {'':>14} {_safe_cc(bt_r):>14.4f}")
    print(f"  {'Mean sigma_hat':<30} {'':>14} {results_r['sigma_hat'].mean():>14.6f}")
    print(f"  {'Mean xi':<30} {'':>14} {results_r['xi'].dropna().mean():>14.4f}")

    return results_r, bt_r

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GARCH(1,1)+EVT VaR")
    parser.add_argument("--challenger", action="store_true",
                        help="also run rolling-window GARCH challenger")
    args = parser.parse_args()

    os.makedirs(OUTPUT_FIGS,   exist_ok=True)
    os.makedirs(OUTPUT_TABLES, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"GARCH(1,1)+EVT VaR  |  alpha={ALPHA:.0%}  |  window={WINDOW} days")
    print(f"Theory: McNeil & Frey (2000) / Irle Section 8.3 + Section 9")
    print(f"{'='*60}")

    # Step 1: Load data
    r_p, pnl = load_data()

    # Step 2: Expanding-window GARCH(1,1) fit
    cond_vol, z_residuals, mu_series, refit_dates, last_garch_res, param_df = \
        fit_garch_expanding(r_p)

    # C6: GARCH parameter stability diagnostics
    print(f"\n{'='*60}")
    print(f"C6: GARCH parameter stability diagnostics (expanding window)")
    param_stability_df = run_garch_parameter_stability_diagnostics(param_df, label="expanding")
    plot_garch_parameter_stability(param_df, label="expanding")

    # Step 3: GARCH diagnostics plots
    plot_garch_diagnostics(r_p, cond_vol, z_residuals)

    # Residual threshold diagnostics (MEP + Hill on backtest-period residual losses)
    plot_residual_threshold_diagnostics(z_residuals)

    # GARCH adequacy diagnostics (Ljung-Box + ARCH-LM + C5 Engle-Ng on residuals)
    diag_df = run_garch_adequacy_diagnostics(r_p, z_residuals)

    # Step 4: Rolling GARCH+EVT VaR
    results = compute_garch_evt_var(pnl, cond_vol, z_residuals, mu_series)

    # Step 5: Backtest
    bt = backtest_garch_evt(pnl, results)

    # Step 6: VaR results plot
    plot_garch_evt_results(pnl, results, bt)

    # Step 7: Save
    save_results(pnl, results, bt)

    # Error-guard activation count from fallback_type column
    n_error_guard = int((results["fallback_type"] == "q_below_threshold").sum())
    print(f"  Error-guard activations (q_below_threshold): {n_error_guard} / {len(results)} "
          f"(expected 0 in production data)")

    # Threshold sensitivity sweep
    sens_df = threshold_sensitivity(pnl, cond_vol, z_residuals, mu_series)

    # Optional challenger run (rolling-window GARCH)
    if args.challenger:
        results_r, bt_r = run_challenger(r_p, pnl)

    print(f"\n{'='*60}")
    print(f"GARCH(1,1)+EVT VaR complete!")
    print(f"  Exceptions       : {bt.N}  (expected {bt.expected_N:.1f} at 99% VaR)")
    print(f"  Kupiec H0        : {'NOT rejected' if not bt.reject_uc else 'REJECTED'}  "
          f"(p={bt.pvalue_uc:.4f})")
    cc_p = getattr(bt, "pvalue_cc", float("nan"))
    print(f"  Christoffersen p : {cc_p:.4f}")
    print(f"  Outputs:")
    print(f"    data/processed/var_garch_evt.csv")
    print(f"    outputs/tables/backtest_garch_evt.csv")
    print(f"    outputs/tables/garch_evt_diagnostics.csv")
    print(f"    outputs/tables/garch_evt_threshold_sensitivity.csv")
    print(f"    outputs/tables/garch_evt_parameter_path_expanding.csv")
    print(f"    outputs/tables/garch_evt_parameter_stability_expanding.csv")
    print(f"    outputs/figures/garch_evt_01_diagnostics.png")
    print(f"    outputs/figures/garch_evt_02_var_results.png")
    print(f"    outputs/figures/garch_evt_03_residual_threshold_diagnostics.png")
    print(f"    outputs/figures/garch_evt_04_parameter_stability_expanding.png")
    print(f"{'='*60}")
