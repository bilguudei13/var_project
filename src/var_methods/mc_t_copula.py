# =============================================================================
# mc_t_copula.py
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
sys.path.insert(0, str(_ROOT))
from config import (V0, WEIGHTS_DICT, PROCESSED_DIR,
                    IRS_NOTIONAL, IRS_FIXED_RATE,
                    STRADDLE_DAYS, STRADDLE_SHARES, RF_RATE)
from portfolio_pricing import build_straddle_state
from portfolio_pricing import price_irs as _price_irs_prod
from backtesting.backtest import run_backtest

# ── Settings ──────────────────────────────────────────────────────────────────
WINDOW      = 750           # rolling estimation window (trading days)
ALPHA       = 0.99          # VaR confidence level
M           = 10_000        # MC scenarios per day  (Module 6 §3.4: ≥10k for 99%)
REFIT_EVERY = 50            # refit marginals + copula every N days
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
    Vectorised IRS mark-to-market pricer using full discrete annuity valuation.
    Delegates to portfolio_pricing.price_irs for consistency with the realized
    P&L pipeline and mc_garch_t_copula.py.
    swap_rates : array (M,) or scalar — one per scenario
    Returns    : array (M,) or scalar — IRS value per scenario
    """
    value, _ = _price_irs_prod(notional, fixed_rate, swap_rates, maturity=maturity)
    return value


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
        # stats.t.fit uses maximum likelihood to find the three parameters of
        # the Student-t that best match the empirical return distribution.
        # The three returned values have the following physical meaning:
        #   df    — degrees of freedom: controls how fat the tails are.
        #           Lower df = heavier tails.  A 5-sigma event is roughly
        #           100× more likely under df=5 than under the Normal, which
        #           makes df the key "tail risk amplifier" in this model.
        #   loc   — location (fitted average daily return): the centre of
        #           the distribution, approximately equal to the sample mean.
        #           For most assets this is a small positive number (~0.03%/day).
        #   scale — scale (fitted dispersion): controls the typical size of
        #           a daily move, analogous to daily volatility (but not equal
        #           to the standard deviation unless df is large; for Student-t,
        #           std = scale * sqrt(df/(df-2))).
        df, loc, scale = stats.t.fit(x)
        # Enforce df > 2 to guarantee that the variance of the distribution is
        # finite.  For df ≤ 2 the variance is infinite, which would cause
        # numerical overflow in downstream computations (e.g., the copula
        # log-likelihood and any moment-based diagnostics).  The threshold 2.01
        # is chosen to be strictly above 2 while allowing very fat tails.
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
        # Dividing by (n+1) instead of n is the Hazen continuity correction.
        # It keeps all pseudo-observation values strictly inside (0, 1):
        # the smallest rank gives 1/(n+1) > 0 and the largest gives n/(n+1) < 1.
        # This matters because the next step applies the t-distribution quantile
        # function (t.ppf), which diverges to ±∞ at 0 and 1; any value exactly
        # at the boundary would produce an infinite t-score and break the copula
        # log-likelihood computation.
        # Conceptually, these uniform values retain the full rank structure of
        # the original returns (i.e., who had the worst day, who had the best)
        # but strip away the marginal distribution shape (whether the returns
        # are t-distributed, skewed, etc.).  Feeding rank-based pseudo-
        # observations into the copula fitting step isolates the dependence
        # structure from the marginals, which is precisely the separation that
        # Sklar's theorem (Module 7 §3.3) requires.
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

    # Z is a draw of M correlated standard normals with covariance matrix R.
    # Each row of Z is an independent d-dimensional scenario; the columns are
    # correlated according to R (the copula correlation matrix).  Multiplying
    # the (M, d) independent standard-normal matrix by L.T on the right applies
    # the correlation structure: Cov(z @ L.T) = L @ I @ L.T = R per row.
    Z = rng.standard_normal((n_samples, d)) @ L.T       # (M, d) correlated normals

    # W is the chi-squared mixing variable that turns correlated normals into
    # correlated t-distributed draws.  Crucially, ONE scalar W is drawn per row
    # (per scenario), meaning ALL d assets in a given scenario share the same
    # realisation of W.  When W is small (tail event of the chi-squared
    # distribution), sqrt(nu/W) is large and ALL assets are simultaneously
    # scaled up in magnitude, creating joint extreme moves.  This shared
    # scaling is the key mechanism that generates tail dependence in the
    # t-copula: assets do not just crash together on average — they crash
    # together with a probability that is governed by nu and does not vanish
    # in the extreme tail (unlike the Gaussian copula).
    W = rng.chisquare(df=nu, size=n_samples)             # (M,)  mixing variable

    # T is the actual multivariate-t(nu, 0, R) sample.  Multiplying Z by
    # sqrt(nu/W) re-scales all components of each row by the same factor:
    # when W is small, the entire scenario vector is "stretched" outward,
    # forcing all assets to have extreme realisations at the same time.
    # This is the mathematical source of tail dependence (lambda_U > 0).
    T = Z * np.sqrt(nu / W[:, np.newaxis])               # (M, d) multivariate-t draws

    # Map the t-distributed scores back to the uniform [0,1] copula space
    # by applying the univariate t-CDF column by column.  After this step,
    # each column of U is marginally uniform on (0,1), but the joint
    # distribution retains the t-copula dependence structure.  These uniform
    # samples can then be passed through any marginal inverse CDF (quantile
    # function) to produce factor shocks in their original units.
    U = stats.t.cdf(T, df=nu)                            # (M, d) uniform copula output

    return U


# =============================================================================
# STEP 5-6 — INVERT MARGINALS AND COMPUTE P&L  (Module 7 §5.1 Steps 5-6)
# =============================================================================

def scenarios_to_pnl(U_sim, marginal_params, linear_prices_now, linear_shares,
                     S_now, sigma_now, rate_now, K, T_now):
    """
    Convert t-copula uniform samples to portfolio P&L via full revaluation.

    Step 5 — Invert marginals (Module 7 §5.1 Step 5):
      r_j = F_{t, df_j, loc_j, scale_j}^{-1}(U_j)   for j = 0..5

    Step 6 — Compute P&L (Module 7 §5.1 Step 6):
      Linear (static inception shares, matching compute_pnl.py):
        dollar_position_j = shares_j × price_now_j
        pnl_linear = Σ_{j ∈ IDX_LINEAR} dollar_position_j × (exp(r_j) − 1)

      Simulated next-day risk factor states (vectorised over M scenarios):
        S_sim     = S_now    × exp(r_SPY)
        sigma_sim = sigma_now × exp(r_VIX)
        rate_sim  = rate_now  + Δr_DGS10

      IRS full revaluation:
        pnl_irs = price_irs(IRS_NOTIONAL, IRS_FIXED_RATE, rate_sim)
                  − price_irs(IRS_NOTIONAL, IRS_FIXED_RATE, rate_now)

      Straddle full revaluation (time decay by 1 day, dynamic discount rate):
        pnl_strad = [BS(S_sim, K, T_next, rate_sim, sigma_sim)
                     − BS(S_now, K, T_now, rate_now, sigma_now)]
                    × STRADDLE_SHARES

    Parameters
    ----------
    U_sim             : ndarray (M, 6)  — copula samples in [0, 1]
    marginal_params   : list of (df, loc, scale) tuples, one per factor
    linear_prices_now : ndarray (4,)    — current prices of linear assets
    linear_shares     : ndarray (4,)    — fixed inception share counts
    S_now             : float — current SPY spot price
    sigma_now         : float — current implied vol (VIX / 100)
    rate_now          : float — current 10Y rate (DGS10 / 100)
    K                 : float — current straddle strike
    T_now             : float — current straddle time-to-expiry (years)

    Returns
    -------
    pnl_total : ndarray (M,)  — simulated portfolio P&L in USD
    """
    eps = 1e-6

    # Step 5: invert each marginal CDF to recover simulated factor shocks.
    # stats.t.ppf is the quantile function (inverse CDF) of the Student-t:
    # for a probability p in (0,1), t.ppf(p, df, loc, scale) returns the
    # value x such that P(X <= x) = p under a t(df, loc, scale) distribution.
    # Applied here, it converts each column of U_sim — which lives in [0,1]
    # copula space — back into a factor shock in the original return units:
    # log returns for EURUSD/GLD/IEF/SPY/VIX, absolute yield change for DGS10.
    # The copula captured the dependence structure (who moves together); the
    # marginal quantile function restores the correct tail shape for each factor
    # individually.  Together, this is the Sklar decomposition in reverse:
    # copula dependence + marginal shapes = joint simulated returns.
    r_sim = np.column_stack([
        stats.t.ppf(
            np.clip(U_sim[:, j], eps, 1 - eps),
            df    = marginal_params[j][0],
            loc   = marginal_params[j][1],
            scale = marginal_params[j][2],
        )
        for j in range(N_ASSETS)
    ])

    # Linear P&L: static inception shares × current price × (exp(r) − 1),
    # matching compute_pnl.py and mc_garch_t_copula.py.  This reflects the
    # actual dollar value of the buy-and-hold position rather than a
    # hypothetical equal-weight fund rebalanced to $250k per asset each day.
    dollar_position = linear_shares * linear_prices_now            # (4,)
    pnl_linear = (dollar_position[np.newaxis, :] *
                  (np.exp(r_sim[:, IDX_LINEAR]) - 1.0)).sum(axis=1)

    # Simulated next-day risk factor states (vectorised over M scenarios)
    S_sim     = S_now     * np.exp(r_sim[:, IDX_SPY])    # (M,)
    sigma_sim = sigma_now * np.exp(r_sim[:, IDX_VIX])    # (M,)
    rate_sim  = rate_now  + r_sim[:, IDX_DGS10]          # (M,)
    T_next    = max(T_now - 1.0 / 252.0, 1.0 / 252.0)

    # Current instrument values (scalars, same for all M scenarios).
    # Use rate_now (actual DGS10) as the Black-Scholes discount factor instead
    # of constant RF_RATE=5%, matching the realized P&L pipeline.
    v_strad_now = price_straddle_vec(S_now, K, T_now, rate_now, sigma_now)
    v_irs_now   = price_irs_vec(IRS_NOTIONAL, IRS_FIXED_RATE, rate_now)

    # IRS full revaluation — single vectorised call over (M,) rate array
    pnl_irs = price_irs_vec(IRS_NOTIONAL, IRS_FIXED_RATE, rate_sim) - v_irs_now

    # Straddle full revaluation — rate_sim used for the simulated discount rate
    # so that a rising-rate scenario reduces the present value of the straddle
    # consistently with its effect on the IRS.
    pnl_straddle = (price_straddle_vec(S_sim, K, T_next, rate_sim, sigma_sim)
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
    A negative VaR (all simulated scenarios profitable) is possible and is
    returned as-is — no floor applied, consistent with the lecture definition.
    """
    q = np.percentile(pnl_sim, (1.0 - alpha) * 100.0)   # 1st percentile of P&L
    var = -q
    if var < 0:
        warnings.warn(f"Negative VaR ({var:.2f}): 1% quantile of simulated P&L is positive.")
    return float(var)


# =============================================================================
# ROLLING LOOP  (Module 6 §5.2 — Rolling Daily Update)
# =============================================================================

def compute_rolling_var(returns_arr, dates, spy_prices, vix_series, dgs10_series,
                        prices_linear, linear_shares):
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
    rng            = np.random.default_rng(SEED)
    T              = len(returns_arr)
    var_arr        = np.full(T, np.nan)
    fit_cache      = None
    refit_log      = []
    straddle_state = build_straddle_state(spy_prices, straddle_days=STRADDLE_DAYS)

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
            W = returns_arr[t - WINDOW : t]         # (750, 6) rolling window

            # Steps 1–3: fit marginals, compute pseudo-obs, fit copula.
            # We cache (marg_params, R, nu) and reuse them for REFIT_EVERY days
            # before re-estimating.  The reason for caching is computational cost:
            # fitting 6 independent Student-t distributions via maximum likelihood
            # plus a profile-MLE search over 19 candidate values of nu is
            # approximately 0.5 seconds per refit.  If we refitted every day over
            # a ~15-year backtest (~3,750 days), that would cost ~30 minutes.
            # By refitting every 50 days, total refit cost falls to ~35 seconds
            # while still tracking slow-moving structural changes in the marginals
            # and correlation structure.  The parameters (marginals + copula) are
            # relatively stable over 50-day windows, so the approximation error
            # from using cached parameters is small compared to estimation noise.
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

        # Current instrument state at t-1 (no look-ahead)
        linear_prices_now = prices_linear.iloc[t - 1].values
        S_now     = float(spy_prices.iloc[t - 1])
        sigma_now = float(vix_series.iloc[t - 1]) / 100.0
        rate_now  = float(dgs10_series.iloc[t - 1]) / 100.0

        K     = float(straddle_state.iloc[t - 1]["strike_spy"])
        T_now = float(straddle_state.iloc[t - 1]["tenor_years"])

        # Step 4: simulate M scenarios from t-copula.
        # The rng object is NOT reset between days, so each day receives a
        # fresh, non-repeating draw of M scenarios even when the copula
        # parameters (R, nu) are identical to the previous day (cached).
        # This ensures that daily VaR estimates are not mechanically identical
        # during cached periods — only the parameter-driven shape of the
        # distribution is the same, while the specific realisations vary.
        U_sim = simulate_t_copula(R, nu, M, rng)

        # Steps 5-6: invert marginals → factor shocks → full revaluation P&L
        pnl_sim = scenarios_to_pnl(
            U_sim, marg_params, linear_prices_now, linear_shares,
            S_now, sigma_now, rate_now, K, T_now
        )

        # Step 7: extract VaR
        var_arr[t] = extract_var(pnl_sim)

    return var_arr[WINDOW:], refit_log


# =============================================================================
# SAMPLE DAY WALKTHROUGH
# =============================================================================

def print_sample_day(returns_arr, dates, spy_prices, vix_series, dgs10_series,
                     prices_linear, linear_shares):
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
    S_now     = float(spy_prices.iloc[sample_t - 1])
    sigma_now = float(vix_series.iloc[sample_t - 1]) / 100.0
    rate_now  = float(dgs10_series.iloc[sample_t - 1]) / 100.0
    K_sample  = S_now                                # ATM for walkthrough
    T_sample  = STRADDLE_DAYS / 252.0

    linear_prices_now = prices_linear.iloc[sample_t - 1].values
    print(f"  Current state: S={S_now:.2f}  σ={sigma_now:.4f}  rate={rate_now:.4f}")
    print(f"  Straddle: K={K_sample:.2f}  T={T_sample:.4f}yr")
    print(f"  IRS: notional=${IRS_NOTIONAL:,}  fixed={IRS_FIXED_RATE:.2%}")

    pnl_sim = scenarios_to_pnl(
        U_sim, marg_params, linear_prices_now, linear_shares,
        S_now, sigma_now, rate_now, K_sample, T_sample
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
    """Print centralised BacktestResult plus copula parameter evolution."""
    print(bt)
    print(f"  Mean VaR over period : ${mean_var:,.0f}")

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
    exc_series = pnl_bt < -pd.Series(var_arr, index=dates_bt)
    exc_mask   = exc_series.values
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
                label=f"VaR breach  (N={bt.N}, rate={bt.exception_rate:.2%})")
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
    roll_rate  = (exc_series.astype(float)
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
        f"(Kupiec p={bt.pvalue_uc:.4f})",
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
    path = FIG_DIR / "mc_t_copula_var_backtest.png"
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
    var_path = DATA_DIR / "var_mc_t_copula.csv"
    pd.DataFrame({"VaR_MC_T_COPULA": var_series.values}, index=dates_bt).to_csv(var_path)

    # Full backtest detail
    bt_path = TAB_DIR / "backtest_mc_t_copula.csv"
    pd.DataFrame({
        "VaR_MC_T_COPULA" : var_series.values,
        "actual_loss"     : -pnl_bt.values,
        "exception"       : (pnl_bt.values < -var_series.values).astype(int),
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
        python src/var_methods/mc_t_copula.py

    Expected runtime: ~3–8 minutes depending on hardware.
    The bottleneck is scipy's Student-t MLE fitting at each refit (~19 calls
    per refit × 4 assets). Between refits, each day's 10,000-scenario draw
    is fast (pure numpy).

    OUTPUTS
    -------
    Console  : sample-day walkthrough, refit log, backtest results
    PNG      : outputs/figures/mc_t_copula_var_backtest.png
    CSV      : data/processed/var_mc_t_copula.csv
               outputs/tables/backtest_mc_t_copula.csv
    """
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
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

    # Static inception shares — matching compute_pnl.py and mc_garch_t_copula.py
    _linear_tickers = [INSTRUMENTS[i] for i in IDX_LINEAR]
    prices_linear = prices.rename(columns={"EURUSD=X": "EURUSD"})[_linear_tickers]
    prices_linear = prices_linear.reindex(dates, method="ffill")
    _weights = pd.Series(
        {("EURUSD" if k == "EURUSD=X" else k): v for k, v in WEIGHTS_DICT.items()}
    )[_linear_tickers]
    linear_shares = (V0 * _weights / prices_linear.iloc[0]).values

    # ── Sample day walkthrough ────────────────────────────────────────────────
    print_sample_day(returns_arr, dates, prices["SPY"], vix_series, dgs10_series,
                     prices_linear, linear_shares)

    # ── Rolling VaR ───────────────────────────────────────────────────────────
    var_arr, refit_log = compute_rolling_var(
        returns_arr, dates, prices["SPY"], vix_series, dgs10_series,
        prices_linear, linear_shares
    )

    dates_bt   = dates[WINDOW:]
    pnl_bt     = pnl_series.iloc[WINDOW:]
    var_series  = pd.Series(var_arr, index=dates_bt)

    # ── Backtest ──────────────────────────────────────────────────────────────
    bt = run_backtest(pnl=pnl_bt, var=var_series, confidence=ALPHA, method_name="MC-t-Copula")
    print_backtest_results(bt, refit_log, mean_var=float(np.nanmean(var_arr)))

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_var_backtest(dates_bt, var_arr, pnl_bt, bt)

    # ── Save ──────────────────────────────────────────────────────────────────
    save_outputs(dates_bt, var_series, pnl_bt, bt)

    print(f"\n{'='*65}")
    print(f"t-Copula Monte Carlo VaR complete!")
    print(f"  Breaches    : {bt.N}  (expected {bt.expected_N:.1f})")
    print(f"  Kupiec H0   : "
          f"{'RETAINED' if not bt.reject_uc else 'REJECTED'}  "
          f"(p={bt.pvalue_uc:.4f})")
    print(f"  Ind H0      : "
          f"{'RETAINED' if not bt.reject_ind else 'REJECTED'}  "
          f"(p={bt.pvalue_ind:.4f})")
    print(f"  Mean VaR    : ${np.nanmean(var_arr):,.0f}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
