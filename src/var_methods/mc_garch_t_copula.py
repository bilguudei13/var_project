# =============================================================================
# monte_carlo_garch_copula_v2.py
# 1-day 99% Monte Carlo VaR: GARCH(1,1)-t marginals + Student-t Copula
#                            with EWMA time-varying correlation
#
# CHANGES FROM v1 (monte_carlo_garch_copula.py)
# ----------------------------------------------
# V1-BUG-1  HORIZON=5 removed → HORIZON=1 (single-step simulation).
#           v1's 5-step intraday GARCH used daily-calibrated alpha/beta at
#           1/5-daily frequency, giving 5× faster mean-reversion than intended.
#           During high-vol regimes sigma_init > unconditional, so the sub-step
#           GARCH aggressively pulled vol back toward the sub-daily steady state
#           within the simulated day, compressing the tail and making VaR too
#           low exactly when crises required higher VaR → too many breaches.
#           With HORIZON=1: r_sim = sigma_forecast × z_std (clean, calibration-
#           consistent, distribution is exactly t_std(nu) scaled by sigma_t).
#
# V1-BUG-2  REFIT_EVERY reduced from 250 → 50 days.
#           v1 held GARCH parameters (omega, alpha, beta) fixed for up to 250
#           days. omega sets the unconditional variance target; when estimated
#           in calm markets and a crisis hits 200 days later, sigma_current
#           could only climb as fast as alpha*r^2 allows while mean-reverting
#           toward a stale low omega. More frequent refitting recalibrates omega
#           to the current regime faster, reducing pro-cyclical underestimation.
#
# RETAINED  GARCH residuals used consistently for copula pseudo-obs and EWMA.
# RETAINED  Profile-MLE t-copula for ν estimation.
# RETAINED  EWMA R_t with λ=0.94 for daily correlation updates.
# RETAINED  Full revaluation: linear + IRS + straddle.
# RETAINED  All shared functions (GARCH fitting, copula, backtest) unchanged.
#
# PIPELINE (unchanged from v1)
# ----------------------------
#   Step 1  Fit GARCH(1,1)-t per factor on rolling WINDOW returns
#   Step 2  Compute rank-based pseudo-observations from GARCH residuals z_t
#   Step 3  Fit 6-dim t-copula (R_static, ν_copula) on residual pseudo-obs
#   Step 3b Init Q_ewma = sample covariance of GARCH residuals
#   Step 4  Update R_dynamic daily via EWMA using z_{t-1} = r_{t-1}/sigma_{t-1}
#   Step 5  Simulate M joint z from t-copula(R_dynamic, ν_copula)
#   Step 6  Reconstruct: r_sim = sigma_forecast × z_std   [single step, HORIZON=1]
#   Step 7  Full revaluation: linear + IRS + straddle
#   Step 8  VaR = -Q_{1%} of simulated P&L
#   Rolling: refit every REFIT_EVERY=50 days; propagate sigma daily between refits
# =============================================================================

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from scipy.special import gammaln
from scipy.stats import norm
from arch import arch_model

# ── Project root and data paths ───────────────────────────────────────────────
_ROOT        = Path(__file__).resolve().parent.parent.parent
DATA_DIR     = _ROOT / "data" / "processed"
RAW_DIR_PATH = _ROOT / "data" / "raw"
FIG_DIR      = _ROOT / "outputs" / "figures"
TAB_DIR      = _ROOT / "outputs" / "tables"

sys.path.insert(0, str(_ROOT / "src" / "data"))
sys.path.insert(0, str(_ROOT))
from config import (V0, WEIGHTS_DICT, IRS_NOTIONAL, IRS_FIXED_RATE,
                    STRADDLE_DAYS, STRADDLE_SHARES, RF_RATE)
from backtesting.backtest import run_backtest
from backtesting.plot_backtest import plot_all
from portfolio_pricing import price_irs as _price_irs_prod
from portfolio_pricing import build_straddle_state

# ── Settings ──────────────────────────────────────────────────────────────────
WINDOW      = 750
ALPHA       = 0.99
M           = 10_000
REFIT_EVERY = 50          # V1-BUG-2: was 250; 50 recalibrates omega faster
NU_GRID     = list(range(2, 21))
SEED        = 42

INSTRUMENTS = ["EURUSD", "GLD", "IEF", "SPY", "VIX_ret", "DGS10_chg"]
N_ASSETS    = len(INSTRUMENTS)

IDX_LINEAR = [0, 1, 2, 3]
IDX_SPY    = 3
IDX_VIX    = 4
IDX_DGS10  = 5

# Linear asset tickers matching INSTRUMENTS[IDX_LINEAR] order
# Used to compute static inception shares (matching compute_pnl.py convention)
LINEAR_TICKERS = ["EURUSD", "GLD", "IEF", "SPY"]

# GARCH_SCALE converts raw decimal log-returns into percentage-point returns
# before passing them to the GARCH optimizer.  For example, a daily SPY return
# of 0.008 (0.8%) becomes 0.8 after scaling.  This matters because numerical
# optimizers (scipy.minimize used internally by the 'arch' library) struggle
# when the parameters being estimated span many orders of magnitude: in decimal
# units, omega (the long-run variance floor) would be on the order of 1e-6
# while alpha and beta are on the order of 0.05–0.90.  In percentage units,
# omega is on the order of 0.01–0.10, which is far closer to alpha and beta
# in magnitude, making the loss surface better-conditioned.  After fitting,
# omega is divided back by GARCH_SCALE^2 (= 10,000) to convert it to decimal
# variance units, while alpha and beta (dimensionless ratios) are unchanged.
GARCH_SCALE = 100.0       # multiply returns before GARCH fit for numerical stability
EWMA_LAMBDA = 0.94        # RiskMetrics decay; half-life ≈ 12 trading days


# =============================================================================
# VECTORISED PRICING FUNCTIONS
# =============================================================================

def price_straddle_vec(S, K, T, r, sigma):
    d1   = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2   = d1 - sigma * np.sqrt(T)
    call = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put  = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return call + put


def price_irs_vec(notional, fixed_rate, swap_rates, maturity=10):
    value, _ = _price_irs_prod(notional, fixed_rate, swap_rates, maturity=maturity)
    return value


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data():
    factors = pd.read_csv(DATA_DIR / "all_factor_returns.csv",
                          index_col=0, parse_dates=True)
    pnl     = pd.read_csv(DATA_DIR / "total_portfolio_pnl.csv",
                          index_col=0, parse_dates=True)
    prices  = pd.read_csv(RAW_DIR_PATH / "prices.csv",
                          index_col=0, parse_dates=True)
    vix     = pd.read_csv(RAW_DIR_PATH / "vix.csv",
                          index_col=0, parse_dates=True).squeeze()
    dgs10   = pd.read_csv(RAW_DIR_PATH / "dgs10.csv",
                          index_col=0, parse_dates=True).squeeze()

    common  = factors.index.intersection(pnl.index)
    prices  = prices.reindex(common, method="ffill")
    vix     = vix.reindex(common, method="ffill")
    dgs10   = dgs10.reindex(common, method="ffill")

    factors = factors.loc[common, INSTRUMENTS]
    pnl_s   = pnl.loc[common, "pnl_total"]

    print(f"Data loaded : {len(common)} days  "
          f"({common[0].date()} → {common[-1].date()})")
    print(f"Factors     : {INSTRUMENTS}")
    print(f"P&L std     : ${pnl_s.std():,.0f}")
    return factors, pnl_s, prices, vix, dgs10


# =============================================================================
def _fit_vix_skew_dist(resids):
    """Placeholder — Johnson SU reverted (made breaches worse 83→88)."""
    return None


# =============================================================================
# STEP 1 — FIT GARCH(1,1)-t MARGINALS  (unchanged from v1)
# =============================================================================

def fit_garch_marginals(window_returns):
    """
    Fit GARCH(1,1) with Student-t innovations to each factor independently.

    Aligned with garch_evt.py: uses mean='Constant' so that GARCH estimates
    a location parameter mu_hat per factor.  Standardised residuals are then
    z_t = (r_t - mu_hat) / sigma_t, matching the lecture decomposition
    Y_t = mu_t + X_t, X_t = sigma_t * Z_t  (Irle §8.6, p. 160).

    Returns std_resids (n, 6) and params_list (one dict per factor) with
    keys: omega, alpha, beta, nu, mu, sigma_last — all in decimal units.
    """
    n, d = window_returns.shape
    std_resids  = np.zeros((n, d))
    params_list = []

    for j in range(d):
        ret_pct = window_returns[:, j] * GARCH_SCALE

        try:
            am  = arch_model(ret_pct, vol='Garch', p=1, q=1,
                             dist='t', mean='Constant', rescale=False)
            res = am.fit(disp='off', show_warning=False)

            p         = res.params
            # Extract mu (location parameter, in %-units)
            mu_pct    = float(p.get('mu', 0.0))
            omega_pct = float(p.get('omega', p.iloc[0]))
            alpha_keys = [k for k in p.index if 'alpha' in k.lower()]
            beta_keys  = [k for k in p.index if 'beta'  in k.lower()]
            nu_keys    = [k for k in p.index if k.lower() in ('nu', 'df', 'eta')]
            alpha = float(p[alpha_keys[0]]) if alpha_keys else 0.05
            beta  = float(p[beta_keys[0]])  if beta_keys  else 0.90
            nu    = float(p[nu_keys[0]])    if nu_keys    else 5.0

            # Physical interpretation of the four GARCH(1,1)-t parameters:
            #
            # omega (ω): the long-run baseline variance contribution — the
            #   unconditional variance floor.  Even in a perfectly calm market
            #   with no recent shocks, sigma^2 cannot fall below omega/(1-alpha-beta).
            #   It is estimated in %-squared units and divided by GARCH_SCALE^2
            #   before storing (see params_list below).
            #
            # alpha (α): the ARCH coefficient — measures how strongly last
            #   period's squared return shock (x_{t-1}^2) feeds into today's
            #   variance forecast.  A high alpha (e.g., 0.15) means the model
            #   reacts quickly and aggressively to new shocks; vol "spikes" fast.
            #
            # beta (β): the GARCH coefficient — measures how much of yesterday's
            #   conditional variance persists into today's forecast.  A high beta
            #   (e.g., 0.85) means volatility is highly persistent: once elevated,
            #   it decays slowly even in the absence of new shocks.
            #
            # alpha + beta: the total persistence of a volatility shock.  If
            #   alpha + beta = 0.97, a 1% vol shock decays to 0.5% only after
            #   log(0.5)/log(0.97) ≈ 23 days — consistent with empirical "vol
            #   clustering" documented by Mandelbrot and Fama.  The stationarity
            #   condition requires alpha + beta < 1; the rescaling below enforces
            #   this if the optimizer produces a non-stationary solution.
            #
            # nu (ν): degrees of freedom for the t-distributed innovations.
            #   Controls the tail fatness of daily shocks AFTER removing the
            #   time-varying GARCH volatility.  A low nu (e.g., 4–6) captures
            #   the "excess kurtosis" of the standardised residuals that remains
            #   even after GARCH filtering.
            ab = alpha + beta
            if ab >= 1.0:
                sf    = 0.9999 / ab
                alpha *= sf
                beta  *= sf
            # Enforce nu > 2 so that the variance of the innovation distribution
            # is finite: Var(t_nu) = nu/(nu-2), which diverges as nu approaches 2.
            # The threshold 2.1 is set just above 2 to allow very fat-tailed fits
            # while keeping the model mathematically well-defined.  No upper bound
            # on nu is imposed, so in calm periods nu may be estimated at 15–20,
            # approaching Gaussian behaviour — which is economically reasonable.
            nu = max(nu, 2.1)

            cond_vol = np.asarray(res.conditional_volatility, dtype=float).copy()
            cond_vol = np.where(cond_vol < 1e-8, 1e-8, cond_vol)
            # Residuals: r_t - mu_hat (arch already subtracts mu when mean='Constant')
            std_resids[:, j] = np.asarray(res.resid, dtype=float) / cond_vol

            params_list.append({
                'omega'      : omega_pct / (GARCH_SCALE ** 2),
                'alpha'      : alpha,
                'beta'       : beta,
                'nu'         : nu,
                'mu'         : mu_pct / GARCH_SCALE,    # decimal units
                'sigma_last' : float(cond_vol[-1]) / GARCH_SCALE,
            })

        except Exception as exc:
            warnings.warn(
                f"GARCH fit failed for {INSTRUMENTS[j]}: {exc}. "
                "Using constant-vol fallback."
            )
            std_dec = max(float(np.std(window_returns[:, j])), 1e-6)
            mu_dec  = float(np.mean(window_returns[:, j]))
            std_resids[:, j] = (window_returns[:, j] - mu_dec) / std_dec
            params_list.append({
                'omega'      : std_dec ** 2 * 0.05,
                'alpha'      : 0.05,
                'beta'       : 0.90,
                'nu'         : 5.0,
                'mu'         : mu_dec,
                'sigma_last' : std_dec,
            })

    # Fit skewed marginal for VIX on top of GARCH residuals
    params_list[IDX_VIX]['vix_skew_dist'] = _fit_vix_skew_dist(
        std_resids[:, IDX_VIX]
    )

    return std_resids, params_list


# =============================================================================
# STEP 2 — PSEUDO-OBSERVATIONS  (unchanged from v1)
# =============================================================================

def compute_pseudo_observations(data):
    n, d = data.shape
    U = np.zeros((n, d))
    for j in range(d):
        ranks   = stats.rankdata(data[:, j])
        U[:, j] = ranks / (n + 1)
    return U


# =============================================================================
# STEP 3 — FIT t-COPULA  (unchanged from v1)
# =============================================================================

def _nearest_pd(A):
    from scipy.linalg import eigh
    B          = (A + A.T) / 2
    vals, vecs = eigh(B)
    return vecs @ np.diag(np.maximum(vals, 1e-8)) @ vecs.T


def _ewma_corr_update(Q_prev, z_t, lam):
    Q_new = lam * Q_prev + (1.0 - lam) * np.outer(z_t, z_t)
    diag  = np.sqrt(np.diag(Q_new))
    diag  = np.where(diag < 1e-8, 1e-8, diag)
    R     = Q_new / np.outer(diag, diag)
    np.fill_diagonal(R, 1.0)
    if np.linalg.eigvalsh(R).min() < 1e-8:
        R = _nearest_pd(R)
    return Q_new, R


def _multivariate_t_logpdf(X, R, nu):
    n, d = X.shape
    try:
        R_inv        = np.linalg.inv(R)
        _, log_det_R = np.linalg.slogdet(R)
    except np.linalg.LinAlgError:
        return -np.inf
    q = np.einsum('ij,jk,ik->i', X, R_inv, X)
    const = (gammaln((nu + d) / 2)
             - gammaln(nu / 2)
             - (d / 2) * np.log(nu * np.pi)
             - 0.5 * log_det_R)
    return n * const - (nu + d) / 2 * np.sum(np.log1p(q / nu))


def fit_t_copula(U_emp, nu_grid=NU_GRID):
    eps        = 1e-6
    best_nu    = nu_grid[0]
    best_ll    = -np.inf
    best_R     = None
    ll_profile = {}

    for nu in nu_grid:
        T_emp = stats.t.ppf(np.clip(U_emp, eps, 1 - eps), df=nu)
        R = np.corrcoef(T_emp.T)
        if np.linalg.eigvalsh(R).min() < 1e-8:
            R = _nearest_pd(R)
        ll_joint     = _multivariate_t_logpdf(T_emp, R, nu)
        ll_marginals = float(np.sum(stats.t.logpdf(T_emp, df=nu)))
        ll_copula    = ll_joint - ll_marginals
        ll_profile[nu] = ll_copula
        if np.isfinite(ll_copula) and ll_copula > best_ll:
            best_ll = ll_copula
            best_nu = nu
            best_R  = R.copy()

    if best_R is None:
        warnings.warn("fit_t_copula: all ν failed — using ν=5, Pearson R.")
        best_nu = 5
        T_emp   = stats.t.ppf(np.clip(U_emp, eps, 1 - eps), df=best_nu)
        best_R  = np.corrcoef(T_emp.T)

    return best_R, best_nu, ll_profile


# =============================================================================
# STEP 3b — COPULA GOODNESS-OF-FIT (Irle §7.2, Cramér-von Mises)
# =============================================================================

def copula_gof_cvm(U_emp, R, nu, n_mc=500, rng_seed=123):
    """
    Cramér-von Mises goodness-of-fit test for the fitted t-copula.
    (Irle Lecture Notes, §7.2, p. 124: S_n = sum (C_X(U_i) - C_n(U_i))^2)

    Uses a parametric bootstrap to obtain the p-value:
    1. Compute the CvM statistic on the observed pseudo-observations.
    2. Simulate n_mc datasets from the fitted t-copula and recompute
       CvM for each → null distribution.
    3. p-value = fraction of bootstrap statistics >= observed.

    Parameters
    ----------
    U_emp : (n, d) array of pseudo-observations in [0, 1]
    R     : (d, d) fitted correlation matrix
    nu    : int    fitted copula degrees of freedom
    n_mc  : int    number of bootstrap samples (default 500)
    rng_seed : int seed for reproducibility

    Returns
    -------
    cvm_stat : float  observed Cramér-von Mises statistic
    p_value  : float  parametric bootstrap p-value
    """
    n, d = U_emp.shape
    eps  = 1e-6

    def _empirical_copula(U, eval_points):
        """C_n(u) = (1/n) * sum_i I(U_i1 <= u1, ..., U_id <= ud)."""
        # Vectorised: for each eval point, count how many observations
        # have all coordinates <= that point
        # eval_points: (m, d), U: (n, d)
        return np.array([
            np.mean(np.all(U <= u, axis=1)) for u in eval_points
        ])

    def _parametric_copula(eval_points, R, nu):
        """C_theta(u) evaluated via multivariate t CDF (Monte Carlo approx)."""
        # For high dimensions, exact CDF is intractable; use large MC sample
        rng_inner = np.random.default_rng(42)
        m_mc = 10_000
        # Draw from t-copula
        try:
            L = np.linalg.cholesky(R)
        except np.linalg.LinAlgError:
            R_pd = _nearest_pd(R)
            L = np.linalg.cholesky(R_pd)
        Z = rng_inner.standard_normal((m_mc, d)) @ L.T
        W = rng_inner.chisquare(df=nu, size=m_mc)
        T_samp = Z * np.sqrt(nu / W[:, np.newaxis])
        U_samp = stats.t.cdf(T_samp, df=nu)

        return np.array([
            np.mean(np.all(U_samp <= u, axis=1)) for u in eval_points
        ])

    def _cvm_stat(U_obs, R, nu):
        C_n     = _empirical_copula(U_obs, U_obs)
        C_theta = _parametric_copula(U_obs, R, nu)
        return float(np.sum((C_theta - C_n) ** 2))

    # Observed statistic
    cvm_obs = _cvm_stat(U_emp, R, nu)

    # Parametric bootstrap
    rng_boot    = np.random.default_rng(rng_seed)
    boot_stats  = np.zeros(n_mc)
    try:
        L = np.linalg.cholesky(R)
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(_nearest_pd(R))

    for b in range(n_mc):
        Z_b = rng_boot.standard_normal((n, d)) @ L.T
        W_b = rng_boot.chisquare(df=nu, size=n)
        T_b = Z_b * np.sqrt(nu / W_b[:, np.newaxis])
        U_b = stats.t.cdf(T_b, df=nu)
        # Re-fit on bootstrap sample (same nu, re-estimate R)
        T_refit = stats.t.ppf(np.clip(U_b, eps, 1 - eps), df=nu)
        R_b = np.corrcoef(T_refit.T)
        if np.linalg.eigvalsh(R_b).min() < 1e-8:
            R_b = _nearest_pd(R_b)
        boot_stats[b] = _cvm_stat(U_b, R_b, nu)

    p_value = float(np.mean(boot_stats >= cvm_obs))
    return cvm_obs, p_value


# =============================================================================
# STEP 4 — SIMULATE FROM t-COPULA  (unchanged from v1)
# =============================================================================

def simulate_t_copula(R, nu, n_samples, rng):
    d = R.shape[0]
    try:
        L = np.linalg.cholesky(R)
    except np.linalg.LinAlgError:
        R = _nearest_pd(R)
        L = np.linalg.cholesky(R)
    Z = rng.standard_normal((n_samples, d)) @ L.T
    W = rng.chisquare(df=nu, size=n_samples)
    T = Z * np.sqrt(nu / W[:, np.newaxis])
    U = stats.t.cdf(T, df=nu)
    return U


# =============================================================================
# STEPS 5-7 — RECONSTRUCT RETURNS AND COMPUTE P&L  (V1-BUG-1 fix: HORIZON=1)
# =============================================================================

def scenarios_to_pnl(nu_marginals, sigma_forecast, mu_arr,
                     R, nu_copula, rng,
                     linear_prices_now, linear_shares,
                     S_now, sigma_now, rate_now, K, T_now):
    """
    Single-step GARCH simulation for 1-day VaR.

    V1-BUG-1 FIX — HORIZON removed:
    v1 ran 5 sub-steps with daily alpha/beta at 1/5 frequency, causing
    vol to mean-revert 5× too fast within the day.  In high-vol regimes
    this compressed the tail, making VaR too low.

    LINEAR P&L FIX — static inception shares:
    v1 used (V0/4)×(exp(r)-1), implicitly rebalancing to equal dollar weights
    each day.  The backtest target (total_portfolio_pnl.csv) uses static
    inception shares (compute_pnl.py: shares = V0 * w / initial_prices).
    Now: PnL_j = shares_j × price_now_j × (exp(r_sim_j) - 1), which equals
    shares_j × (price_sim_j - price_now_j).  Matches historical simulation.

    Standard-t → standardised-t correction:
        scipy.t.ppf returns standard t (var = nu/(nu-2)).
        std_scale = sqrt((nu-2)/nu) converts to unit-variance so that
        r = sigma_forecast * z_std has variance sigma_forecast^2.
    """
    eps = 1e-6

    # Step 4: simulate M joint uniform samples from t-copula
    U = simulate_t_copula(R, nu_copula, M, rng)          # (M, N_ASSETS)

    # Step 5: invert marginal CDFs — unit-variance standardised Student-t.
    # scipy's stats.t.ppf(u, df) returns a draw from a standard t-distribution
    # whose variance is df/(df-2), NOT 1.  This means a raw ppf draw has more
    # dispersion than a unit-variance variable.  We need unit-variance innovations
    # because sigma_forecast is already the one-step-ahead forecast of the
    # standard deviation (in decimal units); the GARCH model is calibrated under
    # the convention Var(z_t) = 1.  Therefore we must rescale:
    #   std_scale = sqrt((nu - 2) / nu)
    # so that Var(ppf_draw * std_scale) = nu/(nu-2) * (nu-2)/nu = 1.
    # Without this correction, the reconstructed return r_sim = sigma * z would
    # have variance sigma^2 * nu/(nu-2), consistently overestimating volatility
    # by the factor nu/(nu-2) — e.g., about 25% too high for nu=8.
    std_scale = np.sqrt((nu_marginals - 2.0) / nu_marginals)   # (N_ASSETS,)
    U_clipped = np.clip(U, eps, 1 - eps)
    z = stats.t.ppf(U_clipped, df=nu_marginals[np.newaxis, :]) * std_scale[np.newaxis, :]

    # Step 6: reconstruct 1-day returns (log-returns for each factor)
    r_sim = mu_arr[np.newaxis, :] + sigma_forecast[np.newaxis, :] * z   # (M, N_ASSETS)

    # Straddle prices for next day (1 trading day elapsed)
    T_next = max(T_now - 1.0 / 252.0, 1.0 / 252.0)

    # Step 7: full revaluation
    # Linear P&L: static inception shares × price_now × (exp(r) - 1)
    # = shares × (price_sim - price_now), matching compute_pnl.py
    dollar_position = linear_shares * linear_prices_now   # (4,) = shares × current price
    pnl_linear = (dollar_position[np.newaxis, :] *
                  (np.exp(r_sim[:, IDX_LINEAR]) - 1.0)).sum(axis=1)

    S_sim     = S_now    * np.exp(r_sim[:, IDX_SPY])
    sigma_sim = sigma_now * np.exp(r_sim[:, IDX_VIX])
    rate_sim  = rate_now  + r_sim[:, IDX_DGS10]

    v_strad_now = price_straddle_vec(S_now, K, T_now, rate_now, sigma_now)
    v_irs_now   = price_irs_vec(IRS_NOTIONAL, IRS_FIXED_RATE, rate_now)

    pnl_irs      = price_irs_vec(IRS_NOTIONAL, IRS_FIXED_RATE, rate_sim) - v_irs_now
    pnl_straddle = (price_straddle_vec(S_sim, K, T_next, rate_sim, sigma_sim)
                    - v_strad_now) * STRADDLE_SHARES

    return pnl_linear + pnl_irs + pnl_straddle


# =============================================================================
# STEP 8 — EXTRACT VaR
# =============================================================================

def extract_var(pnl_sim, alpha=ALPHA):
    q = np.percentile(pnl_sim, (1.0 - alpha) * 100.0)
    var = -q
    if var < 0:
        warnings.warn(f"Negative VaR ({var:.2f}): 1% quantile of simulated P&L is positive.")
    return float(var)


# =============================================================================
# ROLLING LOOP
# =============================================================================

def compute_rolling_var(returns_arr, dates, linear_prices, linear_shares,
                        spy_prices, vix_series, dgs10_series):
    """
    Walk-forward GARCH(1,1)-t copula VaR.

    Refit schedule (every REFIT_EVERY=50 days):
      (a) Fit GARCH(1,1)-t to each factor → std_resids, params, sigma_last
      (b) Fit 6-dim t-copula on residual pseudo-observations → nu_copula
      (c) Initialise Q_ewma from sample covariance of GARCH residuals

    Daily update (between refits):
      GARCH: sigma_{t,j}^2 = omega_j + alpha_j*r_{t-1,j}^2 + beta_j*sigma_{t-1,j}^2
      EWMA:  Q_t = λ*Q_{t-1} + (1-λ)*z_{t-1}*z_{t-1}'  where z_{t-1} = r_{t-1}/sigma_{t-1}
             R_t = normalize(Q_t)

    Each day:
      (d) Update R_t via EWMA using z_{t-1}
      (e) Compute sigma_forecast = GARCH one-step-ahead vol
      (f) Simulate M scenarios from t-copula(R_t, nu_copula) → r_sim = sigma_forecast * z_std
      (g) Full revaluation, extract VaR
    """
    rng       = np.random.default_rng(SEED)
    T         = len(returns_arr)
    var_arr   = np.full(T, np.nan)
    refit_log = []

    omega_arr     = None
    alpha_arr     = None
    beta_arr      = None
    mu_arr        = None
    nu_marginals  = None
    sigma_current = None

    nu_copula_cached  = None
    Q_ewma            = None

    straddle_state = build_straddle_state(spy_prices, straddle_days=STRADDLE_DAYS)

    print(f"\n{'='*65}")
    print(f"GARCH(1,1)-t Copula Monte Carlo VaR  v2  (HORIZON=1, REFIT={REFIT_EVERY}d)")
    print(f"Window={WINDOW} | M={M:,} | alpha={ALPHA:.0%} | EWMA λ={EWMA_LAMBDA}")
    print(f"6-factor GARCH + t-copula: {INSTRUMENTS}")
    print(f"Full revaluation: linear + IRS + straddle")
    print(f"Computing VaR for {T - WINDOW} days …")
    print(f"{'='*65}")

    for t in range(WINDOW, T):

        need_refit = (omega_arr is None) or ((t - WINDOW) % REFIT_EVERY == 0)

        if need_refit:
            W = returns_arr[t - WINDOW : t]

            std_resids, garch_params = fit_garch_marginals(W)

            omega_arr     = np.array([p['omega']      for p in garch_params])
            alpha_arr     = np.array([p['alpha']      for p in garch_params])
            beta_arr      = np.array([p['beta']       for p in garch_params])
            mu_arr        = np.array([p['mu']         for p in garch_params])
            nu_marginals  = np.array([p['nu']         for p in garch_params])
            sigma_current = np.array([p['sigma_last'] for p in garch_params])

            U_resid = compute_pseudo_observations(std_resids)
            R_static, nu_copula_cached, _ = fit_t_copula(U_resid)

            # Copula GoF test (Irle §7.2) — run on first refit and every 10th
            # to balance rigour vs runtime (parametric bootstrap is expensive)
            n_refits_so_far = len(refit_log)
            if n_refits_so_far % 10 == 0:
                cvm_stat, cvm_p = copula_gof_cvm(
                    U_resid, R_static, nu_copula_cached,
                    n_mc=200, rng_seed=SEED + n_refits_so_far
                )
                cvm_msg = (f"  CvM GoF: S={cvm_stat:.4f}  p={cvm_p:.3f}  "
                           f"{'OK' if cvm_p >= 0.05 else 'REJECT at 5%'}")
                print(cvm_msg)

            if Q_ewma is None:
                # First refit: initialise Q_ewma from the sample covariance of
                # the GARCH standardised residuals in this window.  This gives
                # a sensible starting point that is consistent with the fitted
                # GARCH parameters from the very first day of the backtest.
                Q_ewma = np.cov(std_resids.T) + 1e-8 * np.eye(N_ASSETS)
            else:
                # Subsequent refits: a new set of GARCH parameters has been
                # estimated, which produces a different series of standardised
                # residuals (std_resids) for the same historical window.  If we
                # simply kept the old Q_ewma, there would be a discontinuous jump
                # in R_dynamic at the refit date because the incoming z_prev
                # vectors are now computed with different (omega, alpha, beta)
                # and therefore have a different scale than the z vectors that
                # built up Q_ewma under the old parameters.
                #
                # The warm-up loop "replays" all 750 residuals from the new
                # window through the EWMA recursion starting from the new sample
                # covariance.  This aligns Q_ewma's internal state with the
                # current GARCH parameter set before the next daily step, so
                # that R_dynamic transitions smoothly across refit boundaries
                # rather than jumping discontinuously.
                Q_warm = np.cov(std_resids.T) + 1e-8 * np.eye(N_ASSETS)
                for _z_row in std_resids:
                    Q_warm = EWMA_LAMBDA * Q_warm + (1.0 - EWMA_LAMBDA) * np.outer(_z_row, _z_row)
                Q_ewma = Q_warm

            spy_idx = INSTRUMENTS.index("SPY")
            gld_idx = INSTRUMENTS.index("GLD")
            ief_idx = INSTRUMENTS.index("IEF")
            refit_log.append({
                "date"        : dates[t].date(),
                "nu_copula"   : nu_copula_cached,
                "rho_SPY_GLD" : R_static[spy_idx, gld_idx],
                "rho_SPY_IEF" : R_static[spy_idx, ief_idx],
                "nu_SPY"      : nu_marginals[spy_idx],
                "alpha_SPY"   : alpha_arr[spy_idx],
                "beta_SPY"    : beta_arr[spy_idx],
                "sigma_SPY"   : sigma_current[spy_idx],
            })
            print(f"  Refit {len(refit_log):3d} | {dates[t].date()} | "
                  f"ν_cop={nu_copula_cached:2d} | "
                  f"ρ(SPY,GLD)={R_static[spy_idx,gld_idx]:+.3f} | "
                  f"α_SPY={alpha_arr[spy_idx]:.3f} "
                  f"β_SPY={beta_arr[spy_idx]:.3f}")

        # EWMA correlation update using yesterday's standardised innovation.
        r_prev = returns_arr[t - 1]
        # x_prev is the mean-adjusted return shock (the "innovation" in GARCH
        # terminology): x_t = r_t - mu.  This is the raw surprise relative to
        # the model's drift, before scaling by volatility.
        x_prev = r_prev - mu_arr   # innovation: r - mu (Irle §8.6)
        # z_prev is the fully standardised residual: yesterday's return shock
        # divided by yesterday's estimated conditional standard deviation.
        # If GARCH is correctly specified, z_t should be approximately iid with
        # mean 0 and variance 1 across time — the GARCH filter has removed the
        # time-varying heteroskedasticity from r_t, leaving a "cleaned" residual.
        # Using z_prev (rather than raw returns x_prev) for the EWMA correlation
        # update is crucial: correlations estimated on raw returns would confound
        # changing correlations with changing variances.  By standardising first,
        # we ensure that the EWMA tracks the evolution of the pure correlation
        # structure, independently of whether each asset is in a high- or low-vol
        # regime on any given day.
        z_prev            = x_prev / np.where(sigma_current > 1e-8, sigma_current, 1e-8)
        Q_ewma, R_dynamic = _ewma_corr_update(Q_ewma, z_prev, EWMA_LAMBDA)

        # GARCH(1,1) one-step-ahead variance forecast:
        #   sigma^2_{t+1} = omega + alpha * x_t^2 + beta * sigma^2_t
        #
        # Each term has a distinct economic role:
        #   omega:           the long-run variance floor (ensures vol > 0 always).
        #   alpha * x_t^2:  the ARCH effect — amplifies variance in response to
        #                   last period's shock.  A large return surprise (x_t^2
        #                   large) immediately increases next-period vol forecast.
        #   beta * sigma^2: the GARCH effect — carries forward yesterday's
        #                   estimated variance level.  High beta means volatility
        #                   decays slowly ("vol clustering"); low beta means it
        #                   reverts quickly to the long-run mean.
        #
        # Together, the forecast is a weighted combination of the long-run
        # unconditional variance, the most recent squared shock, and the current
        # conditional variance estimate.  Taking the square root of sigma_sq_next
        # gives the one-step-ahead daily standard deviation (in decimal units),
        # which is the volatility scaling factor applied to the copula draws in
        # the scenarios_to_pnl step.  The maximum with 1e-12 prevents numerical
        # issues if any component is slightly negative due to floating-point errors.
        sigma_sq_next  = (omega_arr
                          + alpha_arr * x_prev ** 2
                          + beta_arr  * sigma_current ** 2)
        sigma_forecast = np.sqrt(np.maximum(sigma_sq_next, 1e-12))
        sigma_current  = sigma_forecast.copy()

        # Instrument state — use t-1 to avoid look-ahead bias
        # VaR at t forecasts the move t-1 → t, so market state must come from t-1
        linear_prices_now = linear_prices.iloc[t - 1].values
        S_now     = float(spy_prices.iloc[t - 1])
        sigma_now = float(vix_series.iloc[t - 1]) / 100.0
        rate_now  = float(dgs10_series.iloc[t - 1]) / 100.0

        K     = float(straddle_state.iloc[t - 1]["strike_spy"])
        T_now = float(straddle_state.iloc[t - 1]["tenor_years"])

        # Single-step simulation (V1-BUG-1 fix)
        pnl_sim = scenarios_to_pnl(
            nu_marginals, sigma_forecast, mu_arr,
            R_dynamic, nu_copula_cached, rng,
            linear_prices_now, linear_shares,
            S_now, sigma_now, rate_now, K, T_now,
        )
        var_arr[t] = extract_var(pnl_sim)

    return var_arr[WINDOW:], refit_log


# =============================================================================
# BACKTESTING — delegates to shared framework (Kupiec + Christoffersen)
# =============================================================================

def backtest_mc_garch_copula(pnl, var_series):
    """
    Kupiec + Christoffersen backtests via shared framework.
    (Irle Chapter 8, p. 183-185 — same interface as evt.py / garch_evt.py)
    """
    bt = run_backtest(
        pnl=pnl,
        var=var_series,
        confidence=ALPHA,
        method_name="MC-GARCH-t-Copula",
    )
    print(bt)
    return bt


# =============================================================================
# REFIT LOG PRINTER  (unique to MC — other methods don't have copula params)
# =============================================================================

def print_refit_log(refit_log):
    """Print GARCH + copula parameter evolution across refits."""
    if not refit_log:
        return
    print(f"\n  GARCH + COPULA PARAMETER EVOLUTION (selected refits)")
    print(f"  {'─'*65}")
    print(f"  {'Date':<12} {'ν_cop':>6} {'ρ(SPY,GLD)':>11} "
          f"{'ρ(SPY,IEF)':>11} {'α_SPY':>7} {'β_SPY':>7} {'ν_SPY':>7}")
    step = max(1, len(refit_log) // 10)
    for rec in refit_log[::step]:
        print(f"  {str(rec['date']):<12} {rec['nu_copula']:>6d} "
              f"{rec['rho_SPY_GLD']:>11.4f} "
              f"{rec['rho_SPY_IEF']:>11.4f} "
              f"{rec['alpha_SPY']:>7.4f} "
              f"{rec['beta_SPY']:>7.4f} "
              f"{rec['nu_SPY']:>7.2f}")


# =============================================================================
# VISUALIZATION — delegates to shared framework (same as hist_sim, evt, etc.)
# =============================================================================

# plot_all (imported from backtesting.plot_backtest) generates:
#   - Fig 11: exception timeline (VaR vs actual loss + breach markers)
#   - Fig 13: transition matrix heatmap (n00, n01, n10, n11)


# =============================================================================
# SAVE OUTPUTS
# =============================================================================

def save_outputs(dates_bt, var_series, pnl_bt, bt):
    """Save VaR series and backtest details to CSV."""
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    var_path = DATA_DIR / "var_mc_garch_t_copula.csv"
    pd.DataFrame({"VaR_MC_GARCH_T_COPULA": var_series.values}, index=dates_bt).to_csv(var_path)

    # Build exception series aligned to backtest dates
    from backtesting.backtest import compute_exceptions
    exceptions = compute_exceptions(pnl_bt, var_series)

    bt_path = TAB_DIR / "backtest_mc_garch_t_copula.csv"
    pd.DataFrame({
        "VaR_MC_GARCH_T_COPULA" : var_series.values,
        "actual_loss"            : -pnl_bt.values,
        "exception"              : exceptions.values,
    }, index=dates_bt).to_csv(bt_path)

    print(f"VaR series → {var_path.relative_to(_ROOT)}")
    print(f"Backtest   → {bt_path.relative_to(_ROOT)}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print(f"{'='*65}")
    print(f"GARCH(1,1)-t Copula Monte Carlo VaR  —  v2")
    print(f"  Fix 1: HORIZON=1  (single-step, no intraday vol mean-reversion)")
    print(f"  Fix 2: REFIT_EVERY={REFIT_EVERY}  (was 250; faster omega recalibration)")
    print(f"  Retained: EWMA R_t, GARCH residual copula, full revaluation")
    print(f"{'='*65}")

    factors, pnl_series, prices, vix_series, dgs10_series = load_data()
    returns_arr = factors.values
    dates       = factors.index

    # Static inception shares — matching compute_pnl.py convention exactly
    # shares_j = V0 * w_j / initial_price_j, fixed at inception, never rebalanced
    prices_linear = prices.rename(columns={"EURUSD=X": "EURUSD"})[LINEAR_TICKERS]
    prices_linear = prices_linear.reindex(dates, method="ffill")
    weights = pd.Series(
        {("EURUSD" if k == "EURUSD=X" else k): v for k, v in WEIGHTS_DICT.items()}
    )[LINEAR_TICKERS]
    initial_prices = prices_linear.iloc[0]
    linear_shares  = (V0 * weights / initial_prices).values   # (4,) fixed
    print(f"Static inception shares: "
          + ", ".join(f"{t}={s:.2f}" for t, s in zip(LINEAR_TICKERS, linear_shares)))

    var_arr, refit_log = compute_rolling_var(
        returns_arr, dates, prices_linear, linear_shares,
        prices["SPY"], vix_series, dgs10_series
    )

    dates_bt   = dates[WINDOW:]
    pnl_bt     = pnl_series.iloc[WINDOW:]
    var_series  = pd.Series(var_arr, index=dates_bt)

    # Backtest via shared framework (Kupiec + Christoffersen)
    bt = backtest_mc_garch_copula(pnl_bt, var_series)
    print_refit_log(refit_log)

    # Plots via shared framework (exception timeline + transition matrix)
    plot_all(bt, pnl=pnl_bt, var=var_series, save=True)

    save_outputs(dates_bt, var_series, pnl_bt, bt)

    print(f"\n{'='*65}")
    print(f"GARCH-Copula v2 complete!")
    print(f"  Breaches  : {bt.N}  (expected {bt.expected_N})")
    print(f"  Kupiec    : LR={bt.lr_uc:.4f}  p={bt.pvalue_uc:.4f}  "
          f"{'RETAINED' if not bt.reject_uc else 'REJECTED'}")
    print(f"  Christoff.: LR_IND={bt.lr_ind:.4f}  p={bt.pvalue_ind:.4f}  "
          f"LR_CC={bt.lr_cc:.4f}  p={bt.pvalue_cc:.4f}")
    print(f"  Mean VaR  : ${np.nanmean(var_arr):,.0f}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
