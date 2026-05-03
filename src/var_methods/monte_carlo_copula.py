# =============================================================================
# monte_carlo_copula.py
# 1-day 99% Monte Carlo VaR using a Student-t Copula
#
# THEORY REFERENCES
# -----------------
# Module 6 (Simulation.html) — Simulation Methods for VaR:
#   §1.3  Three-step simulation recipe: generate → reprice → quantile
#   §3.2  Multivariate normal as the naive baseline
#   §3.4  Stability requires M ≥ 10,000 scenarios for 99% VaR
#   §5.2  Rolling daily parameter updates (Approach B)
#   §2.4  Backtesting: counting exceptions against actual P&L
#   §2.5  Binomial(T, 1-α) distribution of breach counts
#
# Module 7 (Copulas.html) — Copulas and Dependence:
#   §1.2  Three problems with correlation (tail dependence failure)
#   §4.3  t-Copula: ρ controls normal-day co-movement, ν controls tail dependence
#   §3.3  Sklar's theorem: F(x,y) = C(F_X(x), F_Y(y))
#   §5.1  Complete workflow: marginals → pseudo-obs → copula → simulate → invert → VaR
#
# WHY t-COPULA INSTEAD OF MULTIVARIATE NORMAL?
# ---------------------------------------------
# The multivariate normal (Gaussian copula) has ZERO tail dependence: as losses
# become more extreme, the model predicts assets behave as if independent.
# Module 7 §4.2 calls this "The Fatal Flaw" — it's why 2008 models failed.
#
# The t-copula adds tail dependence via ν (degrees of freedom). When ν is small
# (e.g., ν=4), assets crash together with meaningful probability even in the
# extreme tail. Module 7 §4.3: at ν=5, ρ=0.7, joint 1% crash probability is
# 0.52% — 52× higher than independence, and 4.7× higher than Gaussian copula.
#
# KEY DIFFERENCE FROM var_copula.py
# ----------------------------------
# var_copula.py: AIC-selected marginals, fixed ν=4 (course reference approach)
# THIS FILE:     Always Student-t marginals, ν estimated by profile likelihood
#                — follows Module 7's theory more closely and is more pedagogical
#
# PORTFOLIO (from config.py)
# --------------------------
#   Linear ($1M, equal 25% weight): SPY, IEF, GLD, EURUSD
#   Nonlinear (full revaluation per scenario): IRS + ATM straddle
#   Risk factors in copula: EURUSD, GLD, IEF, SPY, VIX_ret, DGS10_chg
#   IRS and straddle are repriced using simulated (S, σ, rate) per scenario —
#   no historical bootstrap. Correlation between equity, vol, and rate shocks
#   is captured inside the 6-dimensional t-copula.
#
# WORKFLOW SUMMARY
# ----------------
#   Step 1  Fit Student-t(df, loc, scale) to each of 6 factor returns
#   Step 2  Transform returns to uniform pseudo-observations via rank transform
#   Step 3  Estimate 6-dim t-copula (R, ν) by profile maximum likelihood
#   Step 4  Generate M scenarios from t-copula: Z→Y→T→U (Cholesky algorithm)
#   Step 5  Invert marginal CDFs to recover simulated factor shocks
#   Step 6  Compute portfolio P&L for all M scenarios:
#             Linear:   (V0/4) × Σ(exp(r_j)−1)  for j ∈ {EURUSD,GLD,IEF,SPY}
#             IRS:      price_irs(rate+Δr) − price_irs(rate)
#             Straddle: BS(S·exp(r_SPY),K,T−dt,r,σ·exp(r_VIX)) − BS(S,K,T,r,σ)
#   Step 7  VaR = −Q_{1%} of the simulated P&L distribution
#   Rolling: re-estimate Steps 1-3 every REFIT_EVERY days; fresh draws daily
#   Backtest: count exception days, Kupiec LR test vs Binomial null
# =============================================================================

import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from scipy import stats
from scipy.special import gammaln
from scipy.stats import norm, chi2 as chi2_dist

# ── Project root and data paths ───────────────────────────────────────────────
_ROOT        = Path(__file__).resolve().parent.parent.parent   # var_methods/ → src/ → var_project/
DATA_DIR     = _ROOT / "data" / "processed"
RAW_DIR_PATH = _ROOT / "data" / "raw"
FIG_DIR      = _ROOT / "outputs" / "figures"
TAB_DIR      = _ROOT / "outputs" / "tables"

sys.path.insert(0, str(_ROOT / "src" / "data"))
from config import (V0, PROCESSED_DIR,
                    IRS_NOTIONAL, IRS_FIXED_RATE,
                    STRADDLE_DAYS, STRADDLE_SHARES, RF_RATE)

# ── Settings ──────────────────────────────────────────────────────────────────
WINDOW      = 500           # rolling estimation window (trading days)
ALPHA       = 0.99          # VaR confidence level
M           = 10_000        # MC scenarios per day  (Module 6 §3.4: ≥10k for 99%)
REFIT_EVERY = 250           # refit marginals + copula every N days
NU_GRID     = list(range(2, 21))  # profile likelihood grid: ν ∈ {2, 3, …, 20}
SEED        = 42

INSTRUMENTS = ["EURUSD", "GLD", "IEF", "SPY", "VIX_ret", "DGS10_chg"]
N_ASSETS    = len(INSTRUMENTS)          # 6

IDX_LINEAR = [0, 1, 2, 3]              # columns contributing to linear P&L
IDX_SPY    = 3                          # SPY log return
IDX_VIX    = 4                          # VIX log return  (σ_new = σ × exp(r_VIX))
IDX_DGS10  = 5                          # DGS10 absolute change in decimal


# =============================================================================
# VECTORISED PRICING FUNCTIONS  (mirrors monte_carlo.py)
# =============================================================================

def price_straddle_vec(S, K, T, r, sigma):
    """
    Vectorised ATM straddle pricer (Black-Scholes).
    S, sigma : arrays (M,) or scalars — one per scenario
    K, T, r  : scalars
    Returns  : array (M,) or scalar — straddle value per scenario
    """
    d1   = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2   = d1 - sigma * np.sqrt(T)
    call = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put  = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return call + put


def price_irs_vec(notional, fixed_rate, swap_rates, maturity=10):
    """
    Vectorised IRS mark-to-market pricer.
    swap_rates : array (M,) or scalar — one per scenario
    Returns    : array (M,) or scalar — IRS value per scenario
    """
    return notional * (swap_rates - fixed_rate) * maturity / (1 + swap_rates)


# =============================================================================
# STEP 0 — DATA LOADING
# =============================================================================

def load_data():
    """
    Load 6-factor returns (EURUSD, GLD, IEF, SPY, VIX_ret, DGS10_chg),
    raw state series (SPY prices, VIX levels, DGS10 yields) for per-scenario
    instrument repricing, and total portfolio P&L for backtesting.

    Returns
    -------
    factors   : DataFrame (T × 6)  all_factor_returns aligned to common dates
    pnl_total : Series (T,)        total portfolio P&L (linear + nonlinear)
    prices    : DataFrame          raw prices (SPY column used for straddle state)
    vix       : Series             raw VIX levels in index points (sigma = VIX/100)
    dgs10     : Series             raw DGS10 yields in % (rate = DGS10/100)
    """
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
# STEP 1 — FIT STUDENT-t MARGINALS  (Module 7 §5.1 Step 1)
# =============================================================================

def fit_student_t_marginals(window_returns):
    """
    Fit a Student-t(df, loc, scale) distribution to each asset independently.

    Module 7 §5.1 Step 1:
      "For SPY, fit a Student-t distribution to its historical returns.
       For Gold, fit a Student-t distribution (possibly different parameters)."

    Student-t has three parameters:
      df    — degrees of freedom  (controls tail thickness; lower = fatter tails)
      loc   — location            (≈ mean return)
      scale — scale               (≈ daily volatility)

    Why Student-t over Normal?
      Normal has thin tails: a 5-sigma event should occur once in 3.5M years.
      In equity markets, 5-sigma events happen every few years.
      Student-t with df=5 matches observed tail frequencies far better.

    Parameters
    ----------
    window_returns : ndarray (n, 4)  — raw log-returns for one rolling window

    Returns
    -------
    params : list of (df, loc, scale) tuples, one per asset
    """
    params = []
    for j in range(N_ASSETS):
        x = window_returns[:, j]
        df, loc, scale = stats.t.fit(x)
        df = max(df, 2.01)   # df < 2 → infinite variance; clip for stability
        params.append((df, loc, scale))
    return params


# =============================================================================
# STEP 2 — PSEUDO-OBSERVATIONS  (Module 7 §3.1 and §5.1 Step 2)
# =============================================================================

def compute_pseudo_observations(window_returns):
    """
    Transform each asset's returns to uniform [0, 1] pseudo-observations.

    Module 7 §3.1 (Probability Integral Transform):
      "Every random variable, no matter what distribution it follows, can be
       converted to a uniform random variable on [0,1]."

    We use RANK-BASED pseudo-observations (not the parametric CDF) because:
      - Avoids contaminating the copula fit with marginal mis-specification
      - Matches the standard academic approach (matches R's pobs() function)
      - More robust when the marginal fit is imperfect in the tail

    Formula: U[i, j] = rank(r_{i,j}) / (n + 1)
      The +1 Hazen correction keeps values strictly inside (0, 1), preventing
      infinite values when we later apply the t^{-1} quantile function.

    Parameters
    ----------
    window_returns : ndarray (n, 4)

    Returns
    -------
    U : ndarray (n, 4)  — values in (0, 1)
    """
    n, d = window_returns.shape
    U = np.zeros((n, d))
    for j in range(d):
        ranks   = stats.rankdata(window_returns[:, j])   # 1-based ranks
        U[:, j] = ranks / (n + 1)
    return U


# =============================================================================
# STEP 3 — FIT t-COPULA  (Module 7 §4.3 and §5.1 Step 3)
# =============================================================================

def _nearest_pd(A):
    """Project a symmetric matrix to the nearest positive-definite matrix."""
    from scipy.linalg import eigh
    B         = (A + A.T) / 2
    vals, vecs = eigh(B)
    return vecs @ np.diag(np.maximum(vals, 1e-8)) @ vecs.T


def _multivariate_t_logpdf(X, R, nu):
    """
    Sum of log-pdf values of multivariate Student-t(ν, 0, R) over rows of X.

    Formula:
      log f(x) = log Γ((ν+d)/2) − log Γ(ν/2) − (d/2) log(νπ) − (1/2) log|R|
                 − (ν+d)/2 × log(1 + x' R⁻¹ x / ν)

    Used in the profile likelihood for copula ν estimation (Step 3b below).
    """
    n, d = X.shape
    try:
        R_inv         = np.linalg.inv(R)
        _, log_det_R  = np.linalg.slogdet(R)
    except np.linalg.LinAlgError:
        return -np.inf

    # Mahalanobis distance q_i = x_i' R^{-1} x_i  for all n rows at once
    q = np.einsum('ij,jk,ik->i', X, R_inv, X)   # shape (n,)

    const = (gammaln((nu + d) / 2)
             - gammaln(nu / 2)
             - (d / 2) * np.log(nu * np.pi)
             - 0.5 * log_det_R)

    return n * const - (nu + d) / 2 * np.sum(np.log1p(q / nu))


def fit_t_copula(U_emp, nu_grid=NU_GRID):
    """
    Fit t-copula parameters (R, ν) by profile maximum likelihood.

    Module 7 §4.3 explains the two parameters:
      ρ  (correlation matrix R) — controls normal-day co-movement
      ν  (degrees of freedom)   — controls tail dependence strength
        ν → ∞  : Gaussian copula, ZERO tail dependence
        ν = 10 : moderate tail dependence
        ν = 4  : strong tail dependence (assets crash together often)
        ν = 3  : very strong tail dependence

    Algorithm (profile MLE):
      For each candidate ν in nu_grid:
        (a) Transform U → t-scale:   T = t_ν^{-1}(U)
        (b) Estimate R = corr(T)     (sample correlation — near-optimal MLE for d ≤ 4)
        (c) Copula log-likelihood:
              log ℒ_copula(ν) = log f_{multivar-t}(T; R, ν) − Σ_j log f_{t,ν}(T_j)
              (joint density minus marginal densities — this is the copula density)
      Choose ν* = argmax log ℒ_copula(ν)

    The subtracted marginal term removes the contribution of the marginals,
    isolating the dependence structure per Sklar's theorem (Module 7 §3.3).

    Returns
    -------
    R          : ndarray (4, 4)   — correlation matrix at optimal ν
    nu_best    : int              — estimated degrees of freedom
    ll_profile : dict {ν: log-likelihood}  — for diagnostic output
    """
    eps        = 1e-6
    best_nu    = nu_grid[0]
    best_ll    = -np.inf
    best_R     = None
    ll_profile = {}

    for nu in nu_grid:
        # (a) Transform to t-scale
        T_emp = stats.t.ppf(np.clip(U_emp, eps, 1 - eps), df=nu)   # (n, 4)

        # (b) Sample correlation matrix
        R = np.corrcoef(T_emp.T)                                     # (4, 4)
        if np.linalg.eigvalsh(R).min() < 1e-8:
            R = _nearest_pd(R)

        # (c) Copula log-likelihood = joint multivar-t minus univariate t marginals
        ll_joint     = _multivariate_t_logpdf(T_emp, R, nu)
        ll_marginals = float(np.sum(stats.t.logpdf(T_emp, df=nu)))
        ll_copula    = ll_joint - ll_marginals

        ll_profile[nu] = ll_copula

        if np.isfinite(ll_copula) and ll_copula > best_ll:
            best_ll = ll_copula
            best_nu = nu
            best_R  = R.copy()

    # Fallback if all candidates failed
    if best_R is None:
        warnings.warn("fit_t_copula: all ν candidates failed — using ν=5, Pearson R.")
        best_nu = 5
        T_emp   = stats.t.ppf(np.clip(U_emp, eps, 1 - eps), df=best_nu)
        best_R  = np.corrcoef(T_emp.T)

    return best_R, best_nu, ll_profile


# =============================================================================
# STEP 4 — SIMULATE FROM t-COPULA  (Module 7 §5.1 Step 4)
# =============================================================================

def simulate_t_copula(R, nu, n_samples, rng):
    """
    Generate n_samples draws from t-copula(R, ν).

    Algorithm (Module 7 §5.1 Step 4 code block, McNeil-Frey-Embrechts §7.3):
      1. L  = chol(R)          — Cholesky decomposition  (Module 6 §4(b))
      2. Z  ~ N(0, R)          — L @ standard-normal = correlated normals
      3. W  ~ χ²(ν)            — chi-squared mixing variable (creates fat tails)
      4. T  = Z × √(ν / W)    — multivariate-t(ν, 0, R) sample
      5. U  = F_{t,ν}(T)       — apply univariate t-CDF → back to [0,1] copula space

    The mixing variable W is what creates tail dependence:
      When W is small (happens with probability ∝ 1/ν), ALL components of T
      are scaled UP simultaneously → joint extreme events (Module 7 §4.3).
      A single W per row means all assets share the same scaling in each scenario,
      forcing them to have extreme outcomes together.

    Returns
    -------
    U : ndarray (n_samples, 4)  — copula samples, values in (0, 1)
    """
    d = R.shape[0]

    # Cholesky: Σ = L L'  so  L z ~ N(0, Σ)  when z ~ N(0, I)
    try:
        L = np.linalg.cholesky(R)
    except np.linalg.LinAlgError:
        R = _nearest_pd(R)
        L = np.linalg.cholesky(R)

    Z = rng.standard_normal((n_samples, d)) @ L.T       # (M, 4) correlated normals
    W = rng.chisquare(df=nu, size=n_samples)             # (M,)   mixing variable
    T = Z * np.sqrt(nu / W[:, np.newaxis])               # (M, 4) multivariate-t draws
    U = stats.t.cdf(T, df=nu)                            # (M, 4) uniform copula output

    return U


# =============================================================================
# STEP 5-6 — INVERT MARGINALS AND COMPUTE P&L  (Module 7 §5.1 Steps 5-6)
# =============================================================================

def scenarios_to_pnl(U_sim, marginal_params, S_now, sigma_now, rate_now, K, T_now):
    """
    Convert t-copula uniform samples to portfolio P&L via full revaluation.

    Step 5 — Invert marginals (Module 7 §5.1 Step 5):
      r_j = F_{t, df_j, loc_j, scale_j}^{-1}(U_j)   for j = 0..5

    Step 6 — Compute P&L (Module 7 §5.1 Step 6):
      Linear:
        pnl_linear = (V0 / 4) × Σ_{j ∈ IDX_LINEAR} (exp(r_j) − 1)

      Simulated next-day risk factor states (vectorised over M scenarios):
        S_sim     = S_now    × exp(r_SPY)
        sigma_sim = sigma_now × exp(r_VIX)
        rate_sim  = rate_now  + Δr_DGS10

      IRS full revaluation:
        pnl_irs = price_irs(IRS_NOTIONAL, IRS_FIXED_RATE, rate_sim)
                  − price_irs(IRS_NOTIONAL, IRS_FIXED_RATE, rate_now)

      Straddle full revaluation (time decay by 1 day):
        pnl_strad = [BS(S_sim, K, T_next, RF_RATE, sigma_sim)
                     − BS(S_now, K, T_now, RF_RATE, sigma_now)]
                    × STRADDLE_SHARES

    Parameters
    ----------
    U_sim          : ndarray (M, 6)  — copula samples in [0, 1]
    marginal_params: list of (df, loc, scale) tuples, one per factor
    S_now          : float — current SPY spot price
    sigma_now      : float — current implied vol (VIX / 100)
    rate_now       : float — current 10Y rate (DGS10 / 100)
    K              : float — current straddle strike
    T_now          : float — current straddle time-to-expiry (years)

    Returns
    -------
    pnl_total : ndarray (M,)  — simulated portfolio P&L in USD
    """
    eps = 1e-6

    # Step 5: invert each marginal CDF to recover simulated factor shocks  (M, 6)
    r_sim = np.column_stack([
        stats.t.ppf(
            np.clip(U_sim[:, j], eps, 1 - eps),
            df    = marginal_params[j][0],
            loc   = marginal_params[j][1],
            scale = marginal_params[j][2],
        )
        for j in range(N_ASSETS)
    ])

    # Linear P&L: equal $250k weight per asset, log-return repricing
    pnl_linear = (V0 / len(IDX_LINEAR)) * (np.exp(r_sim[:, IDX_LINEAR]) - 1.0).sum(axis=1)

    # Simulated next-day risk factor states (vectorised over M scenarios)
    S_sim     = S_now     * np.exp(r_sim[:, IDX_SPY])    # (M,)
    sigma_sim = sigma_now * np.exp(r_sim[:, IDX_VIX])    # (M,)
    rate_sim  = rate_now  + r_sim[:, IDX_DGS10]          # (M,)
    T_next    = max(T_now - 1.0 / 252.0, 1.0 / 252.0)

    # Current instrument values (scalars, same for all M scenarios)
    v_strad_now = price_straddle_vec(S_now, K, T_now, RF_RATE, sigma_now)
    v_irs_now   = price_irs_vec(IRS_NOTIONAL, IRS_FIXED_RATE, rate_now)

    # IRS full revaluation — single vectorised call over (M,) rate array
    pnl_irs = price_irs_vec(IRS_NOTIONAL, IRS_FIXED_RATE, rate_sim) - v_irs_now

    # Straddle full revaluation — single vectorised call over (M,) S and sigma arrays
    pnl_straddle = (price_straddle_vec(S_sim, K, T_next, RF_RATE, sigma_sim)
                    - v_strad_now) * STRADDLE_SHARES

    return pnl_linear + pnl_irs + pnl_straddle


# =============================================================================
# STEP 7 — EXTRACT VaR  (Module 6 §1.3 Step 3, §4(d))
# =============================================================================

def extract_var(pnl_sim, alpha=ALPHA):
    """
    VaR = −Q_{1-α} of the simulated P&L distribution.

    Module 6 §1.3 Step 3:
      "Sort the k simulated P&Ls from worst to best.
       The empirical (1-α) quantile is the [(1-α)×k+1]-smallest value."

    Module 6 §3.4: "For 99% VaR, use at least 10,000 scenarios."
      With M=10,000, the 1% tail contains 100 observations → stable estimate.

    The negation converts a loss (negative P&L) to a positive VaR number.
    max(..., 0) prevents reporting a negative VaR on days when every scenario
    shows a gain.
    """
    q = np.percentile(pnl_sim, (1.0 - alpha) * 100.0)   # 1st percentile of P&L
    return float(max(-q, 0.0))


# =============================================================================
# ROLLING LOOP  (Module 6 §5.2 — Rolling Daily Update)
# =============================================================================

def compute_rolling_var(returns_arr, dates, spy_prices, vix_series, dgs10_series):
    """
    Walk-forward rolling t-copula VaR with full nonlinear revaluation.

    Module 6 §5.2 (Rolling Daily Update):
      Re-estimate ALL parameters (marginals + copula) every REFIT_EVERY=250
      days using the most recent WINDOW=500 observations. Fresh M=10,000
      scenarios are drawn each day.

    The 6-factor copula (EURUSD, GLD, IEF, SPY, VIX_ret, DGS10_chg) captures
    correlation between equity returns, volatility shocks, and rate changes
    within each scenario. IRS and straddle are fully repriced per scenario —
    no historical bootstrap.

    Straddle state tracks the rolling 30-day ATM reset (mirrors monte_carlo.py):
      K reset to S_now every STRADDLE_DAYS trading days.
      T_now = (STRADDLE_DAYS − days_held) / 252, floored at 1/252.

    Returns
    -------
    var_arr   : ndarray (T − WINDOW,)  — daily VaR series
    refit_log : list of dicts          — parameter snapshots at each refit
    """
    rng       = np.random.default_rng(SEED)
    T         = len(returns_arr)
    var_arr   = np.full(T, np.nan)
    fit_cache = None
    refit_log = []

    # Straddle state (mirrors compute_pnl.py / monte_carlo.py)
    K         = float(spy_prices.iloc[0])
    days_held = 0

    print(f"\n{'='*65}")
    print(f"t-Copula Monte Carlo VaR  —  Module 6 §3 + Module 7 §4.3")
    print(f"Window={WINDOW} | M={M:,} | alpha={ALPHA:.0%} | "
          f"Refit every {REFIT_EVERY} days")
    print(f"6-factor copula: {INSTRUMENTS}")
    print(f"Full revaluation: linear + IRS + straddle (no bootstrap)")
    print(f"ν estimated by profile MLE over grid ν ∈ {{{NU_GRID[0]}..{NU_GRID[-1]}}}")
    print(f"Computing VaR for {T - WINDOW} days …")
    print(f"{'='*65}")

    for t in range(WINDOW, T):

        need_refit = (fit_cache is None) or ((t - WINDOW) % REFIT_EVERY == 0)

        if need_refit:
            W = returns_arr[t - WINDOW : t]         # (500, 6) rolling window

            # Steps 1–3: fit marginals, compute pseudo-obs, fit copula
            marg_params = fit_student_t_marginals(W)
            U_emp       = compute_pseudo_observations(W)
            R, nu, _    = fit_t_copula(U_emp)
            fit_cache   = (marg_params, R, nu)

            spy_idx = INSTRUMENTS.index("SPY")
            gld_idx = INSTRUMENTS.index("GLD")
            ief_idx = INSTRUMENTS.index("IEF")
            refit_log.append({
                "date"        : dates[t].date(),
                "nu"          : nu,
                "rho_SPY_GLD" : R[spy_idx, gld_idx],
                "rho_SPY_IEF" : R[spy_idx, ief_idx],
                "df_SPY"      : marg_params[spy_idx][0],
                "df_GLD"      : marg_params[gld_idx][0],
                "df_IEF"      : marg_params[ief_idx][0],
                "df_EURUSD"   : marg_params[0][0],
            })
            print(f"  Refit {len(refit_log):3d} | {dates[t].date()} | "
                  f"ν={nu:2d} | "
                  f"ρ(SPY,GLD)={R[spy_idx,gld_idx]:+.3f} | "
                  f"ρ(SPY,IEF)={R[spy_idx,ief_idx]:+.3f} | "
                  f"df_SPY={marg_params[spy_idx][0]:.1f}")

        marg_params, R, nu = fit_cache

        # Current instrument state
        S_now     = float(spy_prices.iloc[t])
        sigma_now = float(vix_series.iloc[t]) / 100.0
        rate_now  = float(dgs10_series.iloc[t]) / 100.0

        # Straddle reset logic
        if days_held >= STRADDLE_DAYS:
            K, days_held = S_now, 0
        T_now     = max((STRADDLE_DAYS - days_held) / 252.0, 1.0 / 252.0)
        days_held += 1

        # Step 4: simulate M scenarios from t-copula
        U_sim = simulate_t_copula(R, nu, M, rng)

        # Steps 5-6: invert marginals → factor shocks → full revaluation P&L
        pnl_sim = scenarios_to_pnl(
            U_sim, marg_params, S_now, sigma_now, rate_now, K, T_now
        )

        # Step 7: extract VaR
        var_arr[t] = extract_var(pnl_sim)

    return var_arr[WINDOW:], refit_log


# =============================================================================
# BACKTESTING  (Module 6 §2.4-§2.5)
# =============================================================================

def run_backtest(var_series, pnl_bt, alpha=ALPHA):
    """
    Count VaR exceptions and run the Kupiec (1995) unconditional coverage test.

    Module 6 §2.4 (Exception definition):
      "If Actual P&L_t < −VaR_t → this is a VaR breach."

    Module 6 §2.5 (Statistical null):
      Under a correctly calibrated VaR model:
        N ~ Binomial(T, 1 − α)
        E[N] = T × (1 − α)
        95% CI = [T×p ± 1.96 × √(T×p×(1−p))]

    Kupiec LR test (likelihood-ratio, asymptotically χ²(1) under H0):
      LR = −2 × [T_0 log(1−p̂) + T_1 log(p̂) − T_0 log(1−p₀) − T_1 log(p₀)]
      where p₀ = 1−α (theoretical rate), p̂ = N/T (observed rate)

    Returns
    -------
    dict with keys: T, N, expected, p_hat, ci_lo, ci_hi, within_ci,
                    lr_stat, kupiec_p, reject_uc, exceptions (boolean Series)
    """
    losses     = -pnl_bt.values
    exceptions = pd.Series(losses > var_series.values, index=pnl_bt.index)

    T       = int(exceptions.notna().sum())
    N       = int(exceptions.sum())
    p0      = 1.0 - alpha
    p_hat   = N / T if T > 0 else 0.0
    expected = T * p0

    # 95% binomial confidence interval (Module 6 §2.5)
    se    = np.sqrt(T * p0 * (1.0 - p0))
    ci_lo = max(0, int(np.floor(expected - 1.96 * se)))
    ci_hi = int(np.ceil(expected + 1.96 * se))

    # Kupiec likelihood-ratio statistic
    eps = 1e-10
    T0  = T - N
    T1  = N
    if 0 < p_hat < 1:
        lr_stat = -2.0 * (
            T0 * np.log(max(1 - p0,   eps) / max(1 - p_hat, eps))
            + T1 * np.log(max(p0,     eps) / max(p_hat,     eps))
        )
    else:
        lr_stat = np.nan

    kupiec_p  = float(1 - chi2_dist.cdf(lr_stat, df=1)) if np.isfinite(lr_stat) else np.nan
    reject_uc = (kupiec_p < 0.05) if np.isfinite(kupiec_p) else None

    return {
        "T"          : T,
        "N"          : N,
        "expected"   : expected,
        "p_hat"      : p_hat,
        "p0"         : p0,
        "ci_lo"      : ci_lo,
        "ci_hi"      : ci_hi,
        "within_ci"  : ci_lo <= N <= ci_hi,
        "lr_stat"    : lr_stat,
        "kupiec_p"   : kupiec_p,
        "reject_uc"  : reject_uc,
        "exceptions" : exceptions,
    }


# =============================================================================
# SAMPLE DAY WALKTHROUGH
# =============================================================================

def print_sample_day(returns_arr, dates, spy_prices, vix_series, dgs10_series):
    """
    Print a detailed step-by-step VaR calculation for one representative day.

    Uses the midpoint of the backtest period so the window includes a
    representative mix of calm and volatile days.
    """
    sample_t    = WINDOW + (len(dates) - WINDOW) // 2
    sample_date = dates[sample_t]
    W           = returns_arr[sample_t - WINDOW : sample_t]   # (500, 6)
    rng_s       = np.random.default_rng(9999)

    print(f"\n{'='*65}")
    print(f"SAMPLE DAY WALKTHROUGH — {sample_date.date()}")
    print(f"Implements Module 7 §5.1 complete workflow  (6-factor t-copula)")
    print(f"Rolling window: last {WINDOW} trading days  |  {N_ASSETS} factors")
    print(f"{'='*65}")

    # ── Step 1: Marginals ────────────────────────────────────────────────────
    print(f"\n[Step 1] Fit Student-t Marginals  (Module 7 §5.1 Step 1)")
    marg_params = fit_student_t_marginals(W)
    for j, name in enumerate(INSTRUMENTS):
        df, loc, scale = marg_params[j]
        tail_desc = ("Very fat tails" if df < 5
                     else "Fat tails" if df < 10
                     else "Near-normal tails")
        print(f"  {name:<12}:  df={df:6.2f}  loc={loc:+.6f}  "
              f"scale={scale:.6f}  ← {tail_desc}")

    # ── Step 2: Pseudo-observations ──────────────────────────────────────────
    print(f"\n[Step 2] Compute Pseudo-Observations  (Module 7 §3.1)")
    U_emp = compute_pseudo_observations(W)
    print(f"  Shape: {U_emp.shape}  (values in (0, 1))")
    spy_idx = INSTRUMENTS.index("SPY")
    worst_spy = W[:, spy_idx].argmin()
    print(f"  Worst SPY day in window → "
          f"return={W[worst_spy, spy_idx]:.4%}, "
          f"percentile={U_emp[worst_spy, spy_idx]:.5f}")

    # ── Step 3: Fit copula ───────────────────────────────────────────────────
    print(f"\n[Step 3] Fit 6-dim t-Copula by Profile MLE  (Module 7 §4.3)")
    R, nu, ll_profile = fit_t_copula(U_emp)

    print(f"\n  Profile log-likelihood by ν:")
    ll_vals = list(ll_profile.items())
    best_ll = max(v for _, v in ll_vals if np.isfinite(v))
    for nv, ll in ll_vals:
        bar    = "█" * int(max(0, (ll - best_ll + 50) / 2)) if np.isfinite(ll) else ""
        marker = "  ← BEST" if nv == nu else ""
        print(f"    ν={nv:2d}:  {ll:10.1f}  {bar}{marker}")

    print(f"\n  Estimated ν = {nu}  → "
          + ("very strong tail dependence" if nu <= 4  else
             "strong tail dependence"      if nu <= 7  else
             "moderate tail dependence"    if nu <= 12 else
             "weak tail dependence (near Gaussian)"))

    print(f"\n  Correlation matrix R (selected rows):")
    show = ["SPY", "GLD", "IEF", "VIX_ret", "DGS10_chg"]
    hdr  = f"  {'':12}" + "".join(f"{n:>12}" for n in show)
    print(hdr)
    show_idx = [INSTRUMENTS.index(n) for n in show]
    for i in [INSTRUMENTS.index(n) for n in show]:
        row = f"  {INSTRUMENTS[i]:<12}" + "".join(f"{R[i,j]:>12.4f}" for j in show_idx)
        print(row)

    # ── Step 4: Simulate ─────────────────────────────────────────────────────
    print(f"\n[Step 4] Simulate {M:,} Scenarios from t-Copula  (Module 7 §5.1 Step 4)")
    print(f"  Z~N(0,R), W~χ²({nu}), T=Z√({nu}/W), U=F_{{t,{nu}}}(T)")
    U_sim = simulate_t_copula(R, nu, M, rng_s)
    print(f"  Output shape: {U_sim.shape}")

    gld_idx    = INSTRUMENTS.index("GLD")
    n_spy_tail = int((U_sim[:, spy_idx] < 0.01).sum())
    n_both     = int(((U_sim[:, spy_idx] < 0.01) & (U_sim[:, gld_idx] < 0.01)).sum())
    print(f"\n  Tail dependence: SPY<1% AND GLD<1% → {n_both} scenarios  "
          f"(independence: ~{int(M * 0.01 * 0.01)})")

    # ── Steps 5-6: Full revaluation ──────────────────────────────────────────
    print(f"\n[Steps 5-6] Full Revaluation  (Module 7 §5.1 Steps 5-6)")
    S_now     = float(spy_prices.iloc[sample_t])
    sigma_now = float(vix_series.iloc[sample_t]) / 100.0
    rate_now  = float(dgs10_series.iloc[sample_t]) / 100.0
    K_sample  = S_now                                # ATM for walkthrough
    T_sample  = STRADDLE_DAYS / 252.0

    print(f"  Current state: S={S_now:.2f}  σ={sigma_now:.4f}  rate={rate_now:.4f}")
    print(f"  Straddle: K={K_sample:.2f}  T={T_sample:.4f}yr")
    print(f"  IRS: notional=${IRS_NOTIONAL:,}  fixed={IRS_FIXED_RATE:.2%}")

    pnl_sim = scenarios_to_pnl(
        U_sim, marg_params, S_now, sigma_now, rate_now, K_sample, T_sample
    )
    pct_lo = np.percentile(pnl_sim, 1)
    print(f"\n  P&L over {M:,} scenarios (USD):")
    print(f"    Mean      : ${pnl_sim.mean():>12,.0f}")
    print(f"    Std dev   : ${pnl_sim.std():>12,.0f}")
    print(f"    Minimum   : ${pnl_sim.min():>12,.0f}")
    print(f"    1st pctile: ${pct_lo:>12,.0f}  ← will become −VaR")

    # ── Step 7: VaR ──────────────────────────────────────────────────────────
    VaR = extract_var(pnl_sim)
    print(f"\n[Step 7] VaR = −Q_{{1%}} = −(${pct_lo:,.0f}) = ${VaR:,.0f}")
    print(f"  With 99% confidence, loss ≤ ${VaR:,.0f} on {sample_date.date()}")
    print(f"{'='*65}")


# =============================================================================
# PRINT BACKTESTING RESULTS
# =============================================================================

def print_backtest_results(bt, refit_log, mean_var):
    """Print all required backtest output, referencing Module 6 §2.4-§2.5."""
    print(f"\n{'='*65}")
    print(f"BACKTESTING RESULTS  —  Module 6 §2.4-§2.5")
    print(f"{'='*65}")
    print(f"  Observations T             : {bt['T']:,}")
    print(f"  Theoretical breach rate    : {bt['p0']:.2%}  (= 1 − {ALPHA:.0%} confidence)")
    print(f"  Expected breaches E[N]     : {bt['expected']:.1f}  (= T × {bt['p0']:.3f})")
    print(f"  95% CI for N               : [{bt['ci_lo']}, {bt['ci_hi']}]")
    print(f"    (Binomial(T={bt['T']}, p={bt['p0']}) — Module 6 §2.5)")
    print(f"  {'─'*55}")
    print(f"  Observed breaches N        : {bt['N']}")
    print(f"  Observed breach rate       : {bt['p_hat']:.4%}")
    print(f"  Comparison to theoretical  : {bt['p_hat']:.4%} vs {bt['p0']:.2%}  "
          f"({'OVER' if bt['p_hat'] > bt['p0'] else 'UNDER'}-estimating risk)")
    print(f"  Within 95% CI?             : {'YES ✓' if bt['within_ci'] else 'NO  ✗'}")
    print(f"  {'─'*55}")
    print(f"  Kupiec LR statistic        : {bt['lr_stat']:.4f}")
    print(f"  Kupiec p-value             : {bt['kupiec_p']:.4f}")
    verdict = ("FAIL TO REJECT H0  → model is correctly calibrated"
               if not bt['reject_uc']
               else "REJECT H0  → model is mis-calibrated")
    print(f"  Kupiec verdict (α=5%)      : {verdict}")
    print(f"  Mean VaR over period       : ${mean_var:,.0f}")
    print(f"  {'─'*55}")

    if bt['N'] > bt['ci_hi']:
        print(f"  → Model UNDERESTIMATES risk ({bt['N']} breaches > upper CI {bt['ci_hi']})")
        print(f"    Likely cause: returns are NOT i.i.d. (volatility clustering")
        print(f"    is ignored here). Fix: GARCH-filtered residuals + t-copula.")
        print(f"    Module 7 warning: 'Volatility clustering violates the i.i.d.")
        print(f"    assumption of pseudo-observations.'")
    elif bt['N'] < bt['ci_lo']:
        print(f"  → Model OVERESTIMATES risk ({bt['N']} breaches < lower CI {bt['ci_lo']})")
        print(f"    VaR is too conservative. The estimated ν may be too small,")
        print(f"    forcing too much tail dependence.")
    else:
        print(f"  → Model ADEQUATELY CALIBRATED at {ALPHA:.0%} confidence level ✓")

    if refit_log:
        print(f"\n  COPULA PARAMETER EVOLUTION (selected refits)")
        print(f"  {'─'*55}")
        print(f"  {'Date':<12} {'ν':>4} {'ρ(SPY,GLD)':>11} "
              f"{'ρ(SPY,IEF)':>11} {'df_SPY':>7} {'df_GLD':>7}")
        step = max(1, len(refit_log) // 8)
        for rec in refit_log[::step]:
            print(f"  {str(rec['date']):<12} {rec['nu']:>4d} "
                  f"{rec['rho_SPY_GLD']:>11.4f} "
                  f"{rec['rho_SPY_IEF']:>11.4f} "
                  f"{rec['df_SPY']:>7.1f} "
                  f"{rec['df_GLD']:>7.1f}")

    print(f"{'='*65}")


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_var_backtest(dates_bt, var_arr, pnl_bt, bt):
    """
    Save 'mc_copula_var_backtest.png' with two panels:
      Panel 1: VaR time series, actual daily losses, and breach markers
      Panel 2: Rolling 252-day breach rate vs theoretical 1%

    The breach markers (red dots) show days where actual loss > VaR —
    the "exceptions" counted in backtesting.
    """
    losses     = -pnl_bt.values
    exc_mask   = bt["exceptions"].values.astype(bool)
    breach_x   = dates_bt[exc_mask]
    breach_y   = losses[exc_mask]

    crises = [
        ("2008-09-01", "2009-06-01", "GFC\n2008"),
        ("2020-02-01", "2020-06-01", "COVID\n2020"),
        ("2022-01-01", "2022-12-01", "Rate\nHikes"),
    ]

    fig, axes = plt.subplots(
        2, 1, figsize=(14, 9), sharex=True,
        gridspec_kw={"height_ratios": [2.5, 1]}
    )
    fig.suptitle(
        "t-Copula Monte Carlo VaR Backtesting\n"
        f"99% VaR  |  Rolling {WINDOW}-day window  |  "
        f"ν estimated by profile MLE  |  M = {M:,} scenarios/day",
        fontsize=12, fontweight="bold", y=0.99
    )

    # ── Panel 1: VaR vs actual loss ───────────────────────────────────────────
    ax1 = axes[0]
    ax1.fill_between(dates_bt, 0, var_arr,
                     alpha=0.10, color="#7B1FA2", label="VaR corridor")
    ax1.plot(dates_bt, var_arr,
             color="#7B1FA2", linewidth=1.3,
             label=f"t-Copula VaR 99%  (mean=${np.nanmean(var_arr):,.0f})")
    ax1.plot(dates_bt, losses,
             color="#90A4AE", linewidth=0.5, alpha=0.7,
             label="Actual daily loss  (−ΔPortfolio)")
    ax1.scatter(breach_x, breach_y,
                color="#F44336", s=22, zorder=5,
                label=f"VaR breach  (N={bt['N']}, rate={bt['p_hat']:.2%})")
    ax1.axhline(0, color="black", linewidth=0.4, linestyle="--")

    y_max = float(np.nanmax([var_arr.max(), losses.max()])) * 1.15
    for s, e, lbl in crises:
        try:
            ax1.axvspan(pd.Timestamp(s), pd.Timestamp(e),
                        alpha=0.07, color="#EF5350")
            mid = pd.Timestamp(s) + (pd.Timestamp(e) - pd.Timestamp(s)) / 2
            ax1.text(mid, y_max * 0.92, lbl,
                     ha="center", va="top", fontsize=8.5, color="#B71C1C",
                     fontweight="bold")
        except Exception:
            pass

    ax1.set_ylim(bottom=0)
    ax1.set_ylabel("USD", fontsize=10)
    ax1.legend(loc="upper left", fontsize=9, framealpha=0.8)
    ax1.set_title("VaR vs Realised Portfolio Loss", fontsize=10, loc="left")
    ax1.grid(axis="y", linewidth=0.4, alpha=0.4)

    # ── Panel 2: Rolling breach rate ──────────────────────────────────────────
    ax2 = axes[1]
    exc_float  = bt["exceptions"].astype(float).values
    roll_rate  = (pd.Series(exc_float, index=dates_bt)
                  .rolling(252, min_periods=63)
                  .mean() * 100)

    ax2.plot(dates_bt, roll_rate,
             color="#F57C00", linewidth=1.3,
             label="Rolling 1-year breach rate (%)")
    ax2.axhline(1.0, color="#2E7D32", linewidth=1.5, linestyle="--",
                label="Theoretical 1%  (correctly calibrated model)")

    y2_max = max(float(roll_rate.dropna().max()) * 1.3, 3.0)
    ax2.set_ylim(0, y2_max)
    ax2.set_ylabel("Breach rate (%)", fontsize=10)
    ax2.legend(fontsize=9, loc="upper right", framealpha=0.8)
    ax2.set_title(
        "Rolling 252-Day Breach Rate vs Theoretical 1%  "
        f"(Kupiec p={bt['kupiec_p']:.4f})",
        fontsize=10, loc="left"
    )
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.xaxis.set_major_locator(mdates.YearLocator(2))
    ax2.grid(axis="y", linewidth=0.4, alpha=0.4)

    for s, e, _ in crises:
        try:
            ax2.axvspan(pd.Timestamp(s), pd.Timestamp(e),
                        alpha=0.07, color="#EF5350")
        except Exception:
            pass

    fig.autofmt_xdate()
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / "mc_copula_var_backtest.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved → {path.relative_to(_ROOT)}")


# =============================================================================
# SAVE OUTPUTS
# =============================================================================

def save_outputs(dates_bt, var_series, pnl_bt, bt):
    """Save VaR series and backtest results to CSV."""
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # VaR series (for cross-model comparison)
    var_path = DATA_DIR / "var_mc_copula.csv"
    pd.DataFrame({"VaR_MC_COPULA": var_series.values}, index=dates_bt).to_csv(var_path)

    # Full backtest detail
    bt_path = TAB_DIR / "backtest_mc_copula.csv"
    pd.DataFrame({
        "VaR_MC_COPULA" : var_series.values,
        "actual_loss"   : -pnl_bt.values,
        "exception"     : bt["exceptions"].values.astype(int),
    }, index=dates_bt).to_csv(bt_path)

    print(f"VaR series → {var_path.relative_to(_ROOT)}")
    print(f"Backtest   → {bt_path.relative_to(_ROOT)}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """
    End-to-end t-Copula Monte Carlo VaR.

    HOW TO RUN
    ----------
    From the project root (var_project/):
        python src/monte_carlo_copula.py

    Expected runtime: ~3–8 minutes depending on hardware.
    The bottleneck is scipy's Student-t MLE fitting at each refit (~19 calls
    per refit × 4 assets). Between refits, each day's 10,000-scenario draw
    is fast (pure numpy).

    OUTPUTS
    -------
    Console  : sample-day walkthrough, refit log, backtest results
    PNG      : outputs/figures/mc_copula_var_backtest.png
    CSV      : data/processed/var_mc_copula.csv
               outputs/tables/backtest_mc_copula.csv
    """
    print(f"{'='*65}")
    print(f"t-Copula Monte Carlo VaR")
    print(f"Scenario generation: Module 7 (Copulas)")
    print(f"VaR extraction + backtest: Module 6 (Simulation)")
    print(f"Upgrade over var_copula.py: ν estimated (not fixed at 4)")
    print(f"{'='*65}")

    # ── Load data ─────────────────────────────────────────────────────────────
    factors, pnl_series, prices, vix_series, dgs10_series = load_data()
    returns_arr = factors.values
    dates       = factors.index

    # ── Sample day walkthrough ────────────────────────────────────────────────
    print_sample_day(returns_arr, dates, prices["SPY"], vix_series, dgs10_series)

    # ── Rolling VaR ───────────────────────────────────────────────────────────
    var_arr, refit_log = compute_rolling_var(
        returns_arr, dates, prices["SPY"], vix_series, dgs10_series
    )

    dates_bt   = dates[WINDOW:]
    pnl_bt     = pnl_series.iloc[WINDOW:]
    var_series  = pd.Series(var_arr, index=dates_bt)

    # ── Backtest ──────────────────────────────────────────────────────────────
    bt = run_backtest(var_series, pnl_bt)
    print_backtest_results(bt, refit_log, mean_var=float(np.nanmean(var_arr)))

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_var_backtest(dates_bt, var_arr, pnl_bt, bt)

    # ── Save ──────────────────────────────────────────────────────────────────
    save_outputs(dates_bt, var_series, pnl_bt, bt)

    print(f"\n{'='*65}")
    print(f"t-Copula Monte Carlo VaR complete!")
    print(f"  Breaches    : {bt['N']}  (expected {bt['expected']:.1f}, "
          f"CI [{bt['ci_lo']}, {bt['ci_hi']}])")
    print(f"  Kupiec H0   : "
          f"{'RETAINED' if not bt['reject_uc'] else 'REJECTED'}  "
          f"(p={bt['kupiec_p']:.4f})")
    print(f"  Mean VaR    : ${np.nanmean(var_arr):,.0f}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
