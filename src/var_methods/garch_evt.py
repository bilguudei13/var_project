# =============================================================================
# garch_evt.py  —  1-day 99% GARCH(1,1) + EVT Conditional VaR
#
# CHANGELOG
# ---------
# 2026-04-18  B9  Separate regulatory floor from statistical estimator in
#                 _pot_var_residuals(): return both q_evt_raw (pure GPD quantile)
#                 and q_evt_floored (policy output clamped at u_z).  For xi < 0
#                 the GPD legitimately yields q_EVT < u_z; the floor is a
#                 conservative regulatory choice, not a statistical property.
#                 VaR_GARCH_EVT_raw added as diagnostic column in all CSVs.
#             B10 Fix refit-boundary residual consistency: z_arr[t-1] was being
#                 computed with the NEW mu_hat after a refit, while
#                 cond_vol_arr[t-1] was produced under the OLD parameters.
#                 Fix: capture mu_hat_prev before the refit block; use it for
#                 the z_arr fill so residuals are always consistent with the
#                 parameter regime that produced their conditional volatility.
# 2026-04-14  B1  CRITICAL: Replace full-sample GARCH with expanding-window
#                 GARCH, re-estimated every 50 days on r_p[0:t]. Between refits,
#                 conditional vol propagated via GARCH(1,1) recursion. Eliminates
#                 look-ahead bias that was present in the original fit_garch_full().
#             B2  Use GARCH-fitted mean res.params['mu']/100 in residual
#                 computation (was: sample mean, slightly biased).
#             B3  Rename sigma_hat_t1 -> sigma_hat_t throughout (naming was
#                 misleading; cond_vol[t] is sigma_t, not sigma_{t+1}).
#             B4  Remove dead WEIGHTS constant (np.ones(4)/4, never used).
#             B5  Log GPD fit failures with warnings.warn for audit trail.
#             B6  Clamp GPD shape xi to [-0.5, 1.0] (xi > 1 -> infinite mean,
#                 xi < -0.5 -> bounded tail; see Cont & Tankov 2004, p.93).
#             B7  Floor VaR at 0 in compute_garch_evt_var (negative VaR is
#                 nonsensical; numerical guard only).
#             B8  KS goodness-of-fit test on each GPD fit (anti-conservative
#                 when params estimated from data, but useful relative quality
#                 indicator). Return ks_pvalue; store in results; print summary.
#
# Theory references:
#   [McNeil & Frey (2000)] "Estimation of tail-related risk measures for
#    heteroscedastic financial time series", J. Empirical Finance 7, 271-300
#   [Irle Lecture Notes] Section 8.3 (GARCH, p. 172-186)
#                      + Section 9 (EVT / POT, p. 212-225)
#
# TWO-STEP PROCEDURE:
#
# STEP A -- GARCH(1,1) volatility filter (Irle Section 8.3, p. 172-186):
#   Model: X_t = sigma_t * Z_t,  Z_t iid, E[Z_t]=0, Var[Z_t]=1
#   Variance: sigma^2_t = omega + alpha_1*X^2_{t-1} + beta*sigma^2_{t-1}
#   (Irle Eq. 17)
#   Innovations: Student-t(nu) -- fat tails in daily returns (Irle p. 179-180)
#   Stationarity: alpha_1 + beta < 1  (Irle p. 176)
#   EXPANDING WINDOW: GARCH re-estimated every 50 days on r_p[0:t]; between
#   refits conditional vol propagated via GARCH recursion (no look-ahead bias).
#
# STEP B -- EVT on standardised residuals (Irle Section 9, p. 212-225):
#   Z_t = (X_t - mu) / sigma_t   (approximately iid after GARCH filtering)
#   Fit GPD to lower tail of {Z_t} via POT:
#     G_{xi,sigma}(x) = 1 - (1 + xi*x/sigma)^{-1/xi}   (Irle p.212, Def.7)
#   Gumbel limit (|xi| < 1e-4):
#     q_EVT = u_z - sigma * log(T_w/N_u * (1-alpha))
#   POT quantile on residuals (dimensionless):
#     q_EVT(alpha) = u_z + (sigma/xi)*[(T_w/N_u*(1-alpha))^{-xi} - 1]
#
# STEP C -- Conditional VaR (McNeil & Frey 2000, Eq. 4):
#   VaR_{alpha,t} = V0 * sigma_t * q_EVT(alpha)
#
# Note on arch scaling:
#   arch_model() requires %-returns for numerical stability.
#   Pass r_p * 100; convert back:
#     cond_vol_decimal = res.conditional_volatility / 100
#     forecast_var_decimal = res.forecast(...).variance / 10000
# =============================================================================

import os
import sys
import warnings
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # handle box-drawing chars on Windows
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive: save to disk, no window
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import genpareto, kstest, t as student_t, probplot
from arch import arch_model

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
REFIT_EVERY     = 50          # re-estimate GARCH every N days (B1)

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
    # Includes linear + IRS + straddle components.
    r_p = pnl / V0
    r_p.name = "portfolio_return_total"

    print(f"Loaded: {len(pnl)} days of total portfolio P&L")
    print(f"Date range : {pnl.index[0].date()} -> {pnl.index[-1].date()}")
    print(f"r_p (total): mean={r_p.mean():.6f}  std={r_p.std():.6f}")
    return r_p, pnl

# =============================================================================
# STEP 2 -- EXPANDING-WINDOW GARCH(1,1) FIT   (B1, B2)
# =============================================================================

def fit_garch_expanding(r_p, window=WINDOW, refit_every=REFIT_EVERY):
    """
    Fit GARCH(1,1) with Student-t innovations using an expanding window.
    (B1: replaces full-sample fit_garch_full to eliminate look-ahead bias)

    Algorithm
    ---------
    For each backtest day t in [window, T):
      - If t == window or (t - window) % refit_every == 0:
          Re-estimate GARCH on r_p[0:t] in %-units (expanding window).
          Store omega, alpha_1, beta, mu (all in decimal).
          Use one-step-ahead forecast for cond_vol[t] = sigma_t.
          On the FIRST refit (t=window): initialise z_arr[0:window] from
          in-sample conditional vols so that the first EVT window is populated.
      - Else:
          Propagate via GARCH(1,1) recursion (decimal units):
            sigma^2_t = omega + alpha_1*(r_{t-1}-mu)^2 + beta*sigma^2_{t-1}
      After each step: z_arr[t-1] = (r_{t-1} - mu) / cond_vol[t-1]
                       (fill the standardised residual for the previous day)

    B2: GARCH-fitted mean mu = res.params['mu'] / 100 is used for residuals
        (more accurate than the sample mean of r_p).

    Parameters
    ----------
    r_p         : pd.Series  total portfolio return (decimal)
    window      : int        initial training window (days)
    refit_every : int        re-estimation frequency (days)

    Returns
    -------
    cond_vol    : pd.Series  sigma_t in decimal return units (index = r_p.index)
    z_residuals : pd.Series  standardised residuals (index = r_p.index)
    refit_dates : list       dates at which GARCH was re-estimated
    last_res    : arch ModelResult from the final refit (for diagnostics)
    """
    n        = len(r_p)
    r_vals   = r_p.values
    dates    = r_p.index

    cond_vol_arr = np.full(n, np.nan)
    z_arr        = np.full(n, np.nan)

    mu_hat   = None      # signals "not yet estimated"
    omega_d  = None
    alpha1_d = None
    beta_d   = None
    sigma_sq_curr = np.nan   # sigma^2 at current t (decimal)

    n_refits    = 0
    refit_dates = []
    last_res    = None

    print(f"\n{'='*60}")
    print(f"GARCH(1,1) expanding-window fit  --  B1 (no look-ahead bias)")
    print(f"dist=Student-t | refit every {refit_every} days | "
          f"total obs={n} | backtest days={n - window}")

    for t in range(window, n):
        do_refit = (mu_hat is None) or ((t - window) % refit_every == 0)

        # B10: Capture mu_hat BEFORE a refit overwrites it.  z_arr[t-1] must use the
        # same parameter regime that produced cond_vol_arr[t-1] — no cross-regime
        # contamination at refit boundaries.
        mu_hat_prev = mu_hat

        if do_refit:
            # ---- Re-estimate GARCH on r_p[0:t] * 100 ----
            train = r_p.iloc[:t] * 100.0   # %-units for arch
            garch = arch_model(train, vol="Garch", p=1, q=1,
                               dist="t", mean="Constant")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res_t = garch.fit(disp="off")

            # B2: GARCH-fitted mean in decimal units
            mu_hat   = res_t.params["mu"]      / 100.0
            omega_d  = res_t.params["omega"]   / 10000.0
            alpha1_d = res_t.params["alpha[1]"]
            beta_d   = res_t.params["beta[1]"]

            # One-step-ahead forecast -> sigma_t for this date
            fcast = res_t.forecast(horizon=1, reindex=False)
            sigma_sq_curr   = fcast.variance.iloc[-1, 0] / 10000.0
            cond_vol_arr[t] = np.sqrt(max(sigma_sq_curr, 1e-10))

            if n_refits == 0:
                # First refit: initialise z_arr[0:window] from in-sample vols
                cv_d = res_t.conditional_volatility.values / 100.0
                z_arr[:t] = np.where(
                    cv_d > 1e-10,
                    (r_vals[:t] - mu_hat) / cv_d,
                    np.nan
                )
                # Print GARCH parameters once (initial fit)
                ab = alpha1_d + beta_d
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

        # B10: Fill z_arr[t-1] using the parameter regime under which cond_vol_arr[t-1]
        # was produced.  mu_hat_prev is None on the first iteration (t==window); skip
        # the fill there — z_arr[:window] is already initialised by the first-refit block.
        if t > window and mu_hat_prev is not None and cond_vol_arr[t - 1] > 1e-10:
            z_arr[t - 1] = (r_vals[t - 1] - mu_hat_prev) / cond_vol_arr[t - 1]

    cond_vol    = pd.Series(cond_vol_arr, index=dates, name="cond_vol")
    z_residuals = pd.Series(z_arr,        index=dates, name="z_residuals")

    z_ok = z_residuals.dropna()
    print(f"\nExpanding-window GARCH summary:")
    print(f"  Total refits : {n_refits}")
    print(f"  Last refit   : {refit_dates[-1].date() if refit_dates else 'N/A'}")
    print(f"  z mean (window+)  : {z_ok.iloc[window:].mean():.4f}  (~=0 expected)")
    print(f"  z std  (window+)  : {z_ok.iloc[window:].std():.4f}   (~=1 expected)")

    return cond_vol, z_residuals, refit_dates, last_res

# =============================================================================
# STEP 3 -- GARCH DIAGNOSTICS (plots)
# =============================================================================

def plot_garch_diagnostics(r_p, cond_vol, z_residuals):
    """
    Two diagnostic panels:

    Panel 1: GARCH conditional volatility (annualised *sqrt(252)) over time.
             Demonstrates volatility clustering captured by GARCH (Irle p. 172).
             Crisis spikes (GFC, COVID) confirm model responsiveness.
             (Conditional vols come from expanding-window re-estimation; B1)

    Panel 2: QQ-plot of standardised residuals vs Student-t distribution.
             Heavy tails in residuals justify EVT on Z_t (Irle p. 179-180).
             If residuals were perfectly Student-t, GARCH+EVT would reduce to
             pure GARCH. QQ-plot typically shows heavier tails, confirming EVT
             adds value.
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

# =============================================================================
# STEP 4 -- POT ON RESIDUALS (helper) + ROLLING VaR   (B5, B6, B8)
# =============================================================================

def _pot_var_residuals(z_w, threshold_q, alpha, T_w):
    """
    Apply POT/GPD to the lower tail of a window of standardised residuals.

    Lower-tail losses in residual space: z_losses = -z_w
    Threshold: u_z = quantile(z_losses, threshold_q)
    Fit GPD on exceedances above u_z (dimensionless).

    B5: GPD fit failures logged with warnings.warn.
    B6: GPD shape xi clamped to [-0.5, 1.0] with warning.
    B8: KS goodness-of-fit test added; ks_pvalue returned.

    Returns (B9: two quantile outputs — statistical estimator and regulatory output)
    -------
    q_evt_floored : float  EVT quantile floored at max(raw, u_z) — regulatory output,
                           used for backtest and capital (= VaR / (V0*sigma_t))
    xi            : float  GPD shape parameter (NaN if fallback used)
    ks_pvalue     : float  KS test p-value (NaN if fallback used; anti-conservative
                           when params are estimated from the same data, use as
                           relative quality indicator only)
    q_evt_raw     : float  Pure GPD quantile before flooring — statistical estimator.
                           For xi < 0 (bounded tail) this may legitimately be < u_z.
                           Equal to q_evt_floored in fallback paths (empirical quantile
                           >= u_z since alpha=0.99 > threshold_q=0.90).

    Reference: Irle p. 223-225 applied to Z_t; McNeil & Frey (2000) Eq. 4
    """
    z_losses    = -z_w
    u_z         = np.quantile(z_losses, threshold_q)
    exceedances = z_losses[z_losses > u_z] - u_z
    N_u         = len(exceedances)

    if N_u < MIN_EXCEEDANCES:
        fb = np.quantile(z_losses, alpha)
        return fb, np.nan, np.nan, fb   # B9: raw == floored for empirical fallback

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            c, _loc, sigma = genpareto.fit(exceedances, floc=0)
        xi = c

        # B6: clamp xi to financially plausible range
        if xi > 1.0:
            warnings.warn(
                f"GPD shape xi={xi:.4f} exceeds upper cap 1.0 (infinite mean); "
                "clamped to 1.0."
            )
            xi = 1.0
        elif xi < -0.5:
            warnings.warn(
                f"GPD shape xi={xi:.4f} below lower cap -0.5 (implausible bounded tail); "
                "clamped to -0.5."
            )
            xi = -0.5

        ratio = (T_w / N_u) * (1.0 - alpha)

        if abs(xi) < 1e-4:
            # Gumbel limit: avoids division by near-zero xi
            q_evt_raw = u_z - sigma * np.log(ratio)
        else:
            q_evt_raw = u_z + (sigma / xi) * (ratio ** (-xi) - 1.0)

        # B9: Conservative floor — for xi < 0, GPD can legitimately yield q_EVT < u_z.
        # Clamping upward is a deliberate regulatory-conservative policy choice, NOT a
        # statistical property of the estimator. q_evt_raw preserves the unmodified value.
        q_evt_floored = max(q_evt_raw, u_z)

        # B8: KS goodness-of-fit test (anti-conservative -- use as relative indicator)
        _, ks_pvalue = kstest(exceedances, "genpareto", args=(xi, 0, sigma))

        return q_evt_floored, xi, ks_pvalue, q_evt_raw   # B9

    except Exception as exc:
        # B5: audit trail for GPD fit failures
        warnings.warn(
            f"GPD fit on residuals failed: {type(exc).__name__}: {exc}. "
            "Using empirical quantile fallback."
        )
        fb = np.quantile(z_losses, alpha)
        return fb, np.nan, np.nan, fb   # B9: raw == floored for empirical fallback


def compute_garch_evt_var(pnl, cond_vol, z_residuals,
                          window=WINDOW, threshold_q=THRESHOLD_Q,
                          alpha=ALPHA, V0=V0):
    """
    Rolling 500-day GARCH(1,1)+EVT conditional VaR (McNeil & Frey 2000).

    For each day t in [window, T):
      (A) sigma_t = cond_vol[t]            GARCH conditional vol for day t
                                            (determined by data up to t-1; no look-ahead)
      (B) z_w = z_residuals[t-window:t]   window of standardised residuals
          q_EVT = POT quantile of lower tail of z_w  (dimensionless)
      (C) VaR_t = V0 * sigma_t * q_EVT    (USD)

    B3: Renamed sigma_hat_t1 -> sigma_hat_t (cond_vol[t] = sigma_t for day t,
        computed from data up to t-1 -- correct one-period conditional vol).
    B7: VaR floored at 0 (negative VaR is nonsensical; numerical guard only).
    B8: Stores ks_pvalue per day; prints poor-fit count in summary.
    B9: Returns both VaR_GARCH_EVT (floored, policy) and VaR_GARCH_EVT_raw
        (unfloored GPD quantile scaled by sigma_t, diagnostic).

    Returns
    -------
    pd.DataFrame  columns: VaR_GARCH_EVT, VaR_GARCH_EVT_raw, sigma_hat, q_EVT, xi, ks_pvalue
      VaR_GARCH_EVT     — regulatory/policy output (floored at u_z and 0)
      VaR_GARCH_EVT_raw — V0 * sigma_t * q_EVT_raw before flooring (diagnostic, B9)
    """
    # Align cond_vol and z_residuals on shared dates
    common    = cond_vol.index.intersection(z_residuals.index)
    vol_vals  = cond_vol.loc[common].values
    z_vals    = z_residuals.loc[common].values
    dates_all = common
    n         = len(common)

    print(f"\n{'='*60}")
    print(f"GARCH(1,1)+EVT VaR  --  McNeil & Frey (2000)")
    print(f"window={window} | alpha={alpha:.0%} | "
          f"threshold={threshold_q:.0%} on residuals | refit every {REFIT_EVERY} days")
    print(f"Computing conditional VaR for {n - window} days ...")

    records, dates = [], []
    n_fallbacks = 0

    for t in range(window, n):
        sigma_hat_t = vol_vals[t]               # B3: sigma_t for day t (decimal)
        z_w         = z_vals[t - window : t]    # window of standardised residuals

        # Handle any NaN in z_w (edge at start of series)
        z_w_clean = z_w[~np.isnan(z_w)]

        q_evt, xi_t, ks_pval, q_evt_raw = _pot_var_residuals(   # B9: unpack 4 values
            z_w_clean, threshold_q, alpha, len(z_w_clean)
        )
        if np.isnan(xi_t):
            n_fallbacks += 1

        var_t     = V0 * sigma_hat_t * q_evt        # VaR in USD (McNeil & Frey Eq. C)
        var_t     = max(var_t, 0.0)                  # B7: floor at 0
        var_raw_t = V0 * sigma_hat_t * q_evt_raw     # B9: unfloored diagnostic

        records.append({
            "VaR_GARCH_EVT"     : var_t,
            "VaR_GARCH_EVT_raw" : var_raw_t,    # B9
            "sigma_hat"         : sigma_hat_t,
            "q_EVT"             : q_evt,
            "xi"                : xi_t,
            "ks_pvalue"         : ks_pval,       # B8
        })
        dates.append(dates_all[t])

    results = pd.DataFrame(records, index=dates)
    xi_ok   = results["xi"].dropna()

    # B8: poor-fit summary
    n_poor_fit = int((results["ks_pvalue"] < 0.05).sum())
    n_bt       = n - window

    print(f"\nRolling GARCH+EVT summary:")
    print(f"  Mean VaR      : ${results['VaR_GARCH_EVT'].mean():>12,.0f}")
    print(f"  Min  VaR      : ${results['VaR_GARCH_EVT'].min():>12,.0f}")
    print(f"  Max  VaR      : ${results['VaR_GARCH_EVT'].max():>12,.0f}")
    print(f"  Mean sigma_hat: {results['sigma_hat'].mean():.6f}  (decimal return units)")
    print(f"  Mean q_EVT    : {results['q_EVT'].mean():.4f}  (dimensionless residual quantile)")
    print(f"  Mean xi       :  {xi_ok.mean():.4f}  (>0 -> heavy tail confirmed)")
    print(f"  Fallback days : {n_fallbacks} / {n_bt}")
    print(f"  Poor GPD fit  : {n_poor_fit} / {n_bt}  (KS p<0.05 -- "
          "anti-conservative, relative indicator only)")

    return results

# =============================================================================
# STEP 5 -- BACKTEST
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
# STEP 6 -- PLOTS
# =============================================================================

def plot_garch_evt_results(pnl, results, bt):
    """
    Two-panel figure:
      Panel 1: GARCH+EVT conditional VaR vs actual loss with exceptions.
      Panel 2: Rolling sigma_t (GARCH conditional vol) showing
               VaR = V0 * sigma_t * q_EVT

    The GARCH component makes VaR spike sharply during crises -- unlike static
    EVT -- illustrating the key advantage of the conditional approach.
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
                 linewidth=0.6, alpha=0.7, label="Actual loss (-DeltaV)")
    if exc_idx is not None and len(exc_idx) > 0:
        axes[0].scatter(exc_idx, actual_loss.loc[exc_idx],
                        color="#F44336", s=20, zorder=5,
                        label=f"Exceptions  N={bt.N}  ({bt.exception_rate:.2%})")
    axes[0].axhline(0, color="black", linewidth=0.5, linestyle="--")
    axes[0].set_title(
        "GARCH(1,1)+EVT Conditional VaR 99% vs Actual Portfolio Loss\n"
        "VaR_t = V0 * sigma_t * q_EVT(alpha)   |   McNeil & Frey (2000) Eq. 4",
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
    Convention matches delta_normal.py and evt.py.
    B8: ks_pvalue column included in backtest CSV.
    B9: VaR_GARCH_EVT_raw (unfloored diagnostic) included in both CSVs.
    """
    results.to_csv(os.path.join(PROCESSED_DIR, "var_garch_evt.csv"))
    print(f"VaR saved     -> {os.path.join(PROCESSED_DIR, 'var_garch_evt.csv')}")

    common = pnl.index.intersection(results.index)
    loss_aligned = -pnl.reindex(common)
    pd.DataFrame({
        "VaR"         : results.loc[common, "VaR_GARCH_EVT"].values,
        "VaR_raw"     : results.loc[common, "VaR_GARCH_EVT_raw"].values,  # B9
        "actual_loss" : loss_aligned.values,
        "exception"   : (loss_aligned > results.loc[common, "VaR_GARCH_EVT"]).astype(int).values,
        "sigma_hat"   : results.loc[common, "sigma_hat"].values,
        "q_EVT"       : results.loc[common, "q_EVT"].values,
        "xi"          : results.loc[common, "xi"].values,
        "ks_pvalue"   : results.loc[common, "ks_pvalue"].values,   # B8
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

    # Step 2: Expanding-window GARCH(1,1) fit  (B1)
    cond_vol, z_residuals, refit_dates, last_garch_res = fit_garch_expanding(r_p)

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

    # B9: Regulatory floor impact diagnostic
    n_floored  = (results["VaR_GARCH_EVT"] > results["VaR_GARCH_EVT_raw"]).sum()
    floor_mean = (results["VaR_GARCH_EVT"] - results["VaR_GARCH_EVT_raw"]).mean()
    print(f"  Days where regulatory floor was binding: {n_floored} / {len(results)} "
          f"({n_floored/len(results):.1%})")
    print(f"  Mean floor impact on VaR: ${floor_mean:,.0f}")

    print(f"\n{'='*60}")
    print(f"GARCH(1,1)+EVT VaR complete!")
    print(f"  Exceptions  : {bt.N}  (expected {bt.expected_N:.1f}  at 99% VaR)")
    print(f"  Kupiec H0   : {'NOT rejected' if not bt.reject_uc else 'REJECTED'}  "
          f"(p={bt.pvalue_uc:.4f})")
    print(f"{'='*60}")
