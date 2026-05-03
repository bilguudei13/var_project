# =============================================================================
# monte_carlo_garch_copula_v3.py
# 1-day 99% Monte Carlo VaR: GARCH(1,1)-t marginals + Student-t Copula
#                            with EWMA time-varying correlation
#
# CHANGES FROM v2 (monte_carlo_garch_copula_v2.py)
# ------------------------------------------------
# V3-CHG-1  Copula fitted on RAW RETURNS (not GARCH residuals).
#           v2 fitted the t-copula on pseudo-observations of GARCH residuals
#           z_t = (r_t − μ̂) / σ_t, which strips vol-clustering from the
#           copula.  v3 uses pseudo-observations of the raw returns directly,
#           so the copula captures joint tail events including the clustered-
#           vol contribution to extreme co-movements.  Consequence: lower ν
#           (more tail dependence) vs v2.  EWMA dynamics still use GARCH
#           residuals for the daily correlation update (z = x/σ unchanged).
#           Q_ewma is also initialised from raw-return covariance for
#           consistency with the raw-return copula.
#
# V3-CHG-2  HYBRID P&L: linear via Monte Carlo, nonlinear via hist bootstrap.
#           v2 simulated IRS and straddle P&L by feeding MC-generated SPY,
#           VIX, DGS10 shocks through pricing functions.  v3 instead:
#             (a) Simulates M GARCH-copula scenarios for the 4 linear factors.
#             (b) Bootstraps nonlinear (IRS + straddle) P&L from the last
#                 WINDOW days of re-priced historical shocks applied to the
#                 CURRENT portfolio snapshot (same repricing logic as HistSim).
#             (c) Combines via rank-matching (Iman-Conover): worst MC linear
#                 scenario is paired with worst historical nonlinear P&L —
#                 conservative but realistic tail coherence without needing
#                 a parametric model for VIX / DGS10 joint dynamics.
#
# V3-CHG-3  STRESS TEST MODULE.
#           After the rolling loop, applies every scenario in the most recent
#           WINDOW historical returns to the final day's portfolio snapshot
#           and reports the N_STRESS_WORST worst losses alongside today's VaR.
#           Loss is decomposed into linear / IRS / straddle components.
#           Results are printed and saved to stress_test_mc_garch_copula_v3.csv.
#
# RETAINED  All of v2:
#   GARCH(1,1)-t marginals, REFIT_EVERY=50, EWMA R_t, profile-MLE ν_copula,
#   CvM GoF test, single-step reconstruction (HORIZON=1), static inception
#   shares, Kupiec + Christoffersen backtesting via centralised module.
#
# PIPELINE
# --------
#   Step 1  Fit GARCH(1,1)-t per factor on rolling WINDOW returns
#   Step 2  Pseudo-obs from RAW RETURNS W  (V3-CHG-1)
#   Step 3  Fit 6-dim t-copula (R_static, ν) on U_raw
#   Step 3b Init Q_ewma from cov(W)  (V3-CHG-1)
#   Step 4  Update R_dynamic daily via EWMA:  z_{t-1} = x_{t-1}/σ_{t-1}
#   Step 5a Simulate M joint scenarios; compute MC linear P&L (V3-CHG-2)
#   Step 5b Bootstrap WINDOW nonlinear (IRS+straddle) scenarios (V3-CHG-2)
#   Step 6  Rank-match; total P&L = MC linear + bootstrapped nonlinear
#   Step 7  VaR = -Q_{1%} of total P&L
#   Stress  Top-N worst historical scenarios vs current portfolio (V3-CHG-3)
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
from backtesting.backtest import run_backtest, compute_exceptions
from backtesting.plot_backtest import plot_all

# ── Settings ──────────────────────────────────────────────────────────────────
WINDOW         = 500
ALPHA          = 0.99
M              = 10_000
REFIT_EVERY    = 50
NU_GRID        = list(range(2, 21))
SEED           = 42
N_STRESS_WORST = 10        # worst historical scenarios to report in stress test

INSTRUMENTS = ["EURUSD", "GLD", "IEF", "SPY", "VIX_ret", "DGS10_chg"]
N_ASSETS    = len(INSTRUMENTS)

IDX_LINEAR = [0, 1, 2, 3]
IDX_SPY    = 3
IDX_VIX    = 4
IDX_DGS10  = 5

LINEAR_TICKERS = ["EURUSD", "GLD", "IEF", "SPY"]

GARCH_SCALE = 100.0
EWMA_LAMBDA = 0.94


# =============================================================================
# VECTORISED PRICING FUNCTIONS  (unchanged from v2)
# =============================================================================

def price_straddle_vec(S, K, T, r, sigma):
    d1   = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2   = d1 - sigma * np.sqrt(T)
    call = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put  = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return call + put


def price_irs_vec(notional, fixed_rate, swap_rates, maturity=10):
    return notional * (swap_rates - fixed_rate) * maturity / (1 + swap_rates)


# =============================================================================
# DATA LOADING  (unchanged from v2)
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

    common = factors.index.intersection(pnl.index)
    prices = prices.reindex(common, method="ffill")
    vix    = vix.reindex(common, method="ffill")
    dgs10  = dgs10.reindex(common, method="ffill")

    factors = factors.loc[common, INSTRUMENTS]
    pnl_s   = pnl.loc[common, "pnl_total"]

    print(f"Data loaded : {len(common)} days  "
          f"({common[0].date()} → {common[-1].date()})")
    print(f"Factors     : {INSTRUMENTS}")
    print(f"P&L std     : ${pnl_s.std():,.0f}")
    return factors, pnl_s, prices, vix, dgs10


# =============================================================================
# STEP 1 — FIT GARCH(1,1)-t MARGINALS  (unchanged from v2)
# =============================================================================

def fit_garch_marginals(window_returns):
    """
    GARCH(1,1)-t per factor on window_returns (n, 6).
    Returns std_resids (n, 6) and params_list with keys:
      omega, alpha, beta, nu, mu, sigma_last  — all in decimal units.
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

            p          = res.params
            mu_pct     = float(p.get('mu', 0.0))
            omega_pct  = float(p.get('omega', p.iloc[0]))
            alpha_keys = [k for k in p.index if 'alpha' in k.lower()]
            beta_keys  = [k for k in p.index if 'beta'  in k.lower()]
            nu_keys    = [k for k in p.index if k.lower() in ('nu', 'df', 'eta')]
            alpha = float(p[alpha_keys[0]]) if alpha_keys else 0.05
            beta  = float(p[beta_keys[0]])  if beta_keys  else 0.90
            nu    = float(p[nu_keys[0]])    if nu_keys    else 5.0

            ab = alpha + beta
            if ab >= 1.0:
                sf    = 0.9999 / ab
                alpha *= sf
                beta  *= sf
            nu = max(nu, 2.1)

            cond_vol = np.asarray(res.conditional_volatility, dtype=float).copy()
            cond_vol = np.where(cond_vol < 1e-8, 1e-8, cond_vol)
            std_resids[:, j] = np.asarray(res.resid, dtype=float) / cond_vol

            params_list.append({
                'omega'      : omega_pct / (GARCH_SCALE ** 2),
                'alpha'      : alpha,
                'beta'       : beta,
                'nu'         : nu,
                'mu'         : mu_pct / GARCH_SCALE,
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

    return std_resids, params_list


# =============================================================================
# STEP 2 — PSEUDO-OBSERVATIONS  (unchanged from v2)
# =============================================================================

def compute_pseudo_observations(data):
    n, d = data.shape
    U = np.zeros((n, d))
    for j in range(d):
        ranks   = stats.rankdata(data[:, j])
        U[:, j] = ranks / (n + 1)
    return U


# =============================================================================
# STEP 3 — FIT t-COPULA  (unchanged from v2; called on raw returns in v3)
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
# STEP 3b — COPULA GoF (Cramér-von Mises)  (unchanged from v2)
# =============================================================================

def copula_gof_cvm(U_emp, R, nu, n_mc=500, rng_seed=123):
    n, d = U_emp.shape
    eps  = 1e-6

    def _ec(U, pts):
        return np.array([np.mean(np.all(U <= u, axis=1)) for u in pts])

    def _pc(pts, R, nu):
        rng_i = np.random.default_rng(42)
        try:
            L = np.linalg.cholesky(R)
        except np.linalg.LinAlgError:
            L = np.linalg.cholesky(_nearest_pd(R))
        Z = rng_i.standard_normal((10_000, d)) @ L.T
        W = rng_i.chisquare(df=nu, size=10_000)
        U_s = stats.t.cdf(Z * np.sqrt(nu / W[:, np.newaxis]), df=nu)
        return np.array([np.mean(np.all(U_s <= u, axis=1)) for u in pts])

    def _cvm(U_obs, R, nu):
        return float(np.sum((_pc(U_obs, R, nu) - _ec(U_obs, U_obs)) ** 2))

    cvm_obs  = _cvm(U_emp, R, nu)
    rng_boot = np.random.default_rng(rng_seed)
    try:
        L = np.linalg.cholesky(R)
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(_nearest_pd(R))

    boot_stats = np.zeros(n_mc)
    for b in range(n_mc):
        Z_b  = rng_boot.standard_normal((n, d)) @ L.T
        W_b  = rng_boot.chisquare(df=nu, size=n)
        U_b  = stats.t.cdf(Z_b * np.sqrt(nu / W_b[:, np.newaxis]), df=nu)
        T_rf = stats.t.ppf(np.clip(U_b, eps, 1 - eps), df=nu)
        R_b  = np.corrcoef(T_rf.T)
        if np.linalg.eigvalsh(R_b).min() < 1e-8:
            R_b = _nearest_pd(R_b)
        boot_stats[b] = _cvm(U_b, R_b, nu)

    return cvm_obs, float(np.mean(boot_stats >= cvm_obs))


# =============================================================================
# STEP 4 — SIMULATE FROM t-COPULA  (unchanged from v2)
# =============================================================================

def simulate_t_copula(R, nu, n_samples, rng):
    d = R.shape[0]
    try:
        L = np.linalg.cholesky(R)
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(_nearest_pd(R))
    Z = rng.standard_normal((n_samples, d)) @ L.T
    W = rng.chisquare(df=nu, size=n_samples)
    return stats.t.cdf(Z * np.sqrt(nu / W[:, np.newaxis]), df=nu)


# =============================================================================
# STEPS 5-7 — HYBRID P&L: MC linear + historical bootstrap nonlinear  (V3-CHG-2)
# =============================================================================

def scenarios_to_pnl_hybrid(nu_marginals, sigma_forecast,
                              R, nu_copula, rng,
                              linear_shares, linear_prices_now,
                              hist_returns_window,
                              S_now, sigma_now, rate_now, K, T_now):
    """
    Hybrid 1-day P&L for VaR (V3-CHG-2).

    Linear P&L  — M MC scenarios via GARCH-copula (4 linear factors).
    Nonlinear   — bootstrap pool: reprice IRS + straddle under each of the
                  WINDOW historical shocks applied to today's portfolio state.
    Combining   — rank-match (Iman-Conover): sort MC linear ascending, sort
                  bootstrapped nonlinear ascending, interpolate to M points,
                  pair by rank.  Worst linear ↔ worst nonlinear.

    Standard-t correction: scipy.t.ppf has var = ν/(ν-2); std_scale converts
    to unit-variance innovations consistent with GARCH(1,1)-t fitting.
    """
    eps    = 1e-6
    T_next = max(T_now - 1.0 / 252.0, 1.0 / 252.0)

    # ── Step 5a: simulate M joint scenarios, compute MC linear P&L ──────────
    U = simulate_t_copula(R, nu_copula, M, rng)              # (M, N_ASSETS)
    std_scale = np.sqrt((nu_marginals - 2.0) / nu_marginals) # (N_ASSETS,)
    z = np.column_stack([
        stats.t.ppf(np.clip(U[:, j], eps, 1 - eps), df=nu_marginals[j])
        * std_scale[j]
        for j in range(N_ASSETS)
    ])                                                        # (M, N_ASSETS)
    r_sim = sigma_forecast[np.newaxis, :] * z                # (M, N_ASSETS)

    dollar_pos  = linear_shares * linear_prices_now          # (4,)
    pnl_linear  = (dollar_pos[np.newaxis, :]
                   * (np.exp(r_sim[:, IDX_LINEAR]) - 1.0)).sum(axis=1)  # (M,)

    # ── Step 5b: bootstrap nonlinear P&L from historical shocks ─────────────
    # Reprice IRS + straddle using each historical return applied to current state.
    v_strad_now = price_straddle_vec(S_now, K, T_now, RF_RATE, sigma_now)
    v_irs_now   = price_irs_vec(IRS_NOTIONAL, IRS_FIXED_RATE, rate_now)

    S_hist     = S_now    * np.exp(hist_returns_window[:, IDX_SPY])
    sig_hist   = sigma_now * np.exp(hist_returns_window[:, IDX_VIX])
    rate_hist  = rate_now  + hist_returns_window[:, IDX_DGS10]

    pnl_irs_h  = price_irs_vec(IRS_NOTIONAL, IRS_FIXED_RATE, rate_hist) - v_irs_now
    pnl_str_h  = (price_straddle_vec(S_hist, K, T_next, RF_RATE, sig_hist)
                  - v_strad_now) * STRADDLE_SHARES
    pnl_nl_h   = pnl_irs_h + pnl_str_h                      # (WINDOW,)

    # ── Step 6: rank-match (Iman-Conover) ────────────────────────────────────
    # Sort both ascending; interpolate nonlinear to M points; pair by rank.
    lin_order  = np.argsort(pnl_linear)                      # ascending permutation
    nl_sorted  = np.sort(pnl_nl_h)                           # (WINDOW,) ascending

    resample_idx = np.round(
        np.linspace(0, len(nl_sorted) - 1, M)
    ).astype(int)
    nl_resampled = nl_sorted[resample_idx]                   # (M,) ascending

    pnl_nl_matched          = np.empty(M)
    pnl_nl_matched[lin_order] = nl_resampled                 # rank-matched

    return pnl_linear + pnl_nl_matched


# =============================================================================
# STEP 8 — EXTRACT VaR  (unchanged from v2)
# =============================================================================

def extract_var(pnl_sim, alpha=ALPHA):
    q = np.percentile(pnl_sim, (1.0 - alpha) * 100.0)
    return float(max(-q, 0.0))


# =============================================================================
# STRESS TEST MODULE  (V3-CHG-3)
# =============================================================================

def run_stress_test(returns_arr, dates,
                    linear_shares, linear_prices_now,
                    S_now, sigma_now, rate_now, K, T_now,
                    var_today, n_worst=N_STRESS_WORST):
    """
    Reprice today's portfolio under every scenario in returns_arr and report
    the N worst losses.

    All repricing uses the current-state snapshot (same convention as HistSim):
      linear:   dollar_pos × (exp(r) - 1)
      IRS:      price_irs(rate_now + Δr) - price_irs(rate_now)
      straddle: price_straddle(S_now×exp(r_SPY), sigma_now×exp(r_VIX)) - v_now

    Parameters
    ----------
    returns_arr       : (n, 6) historical log-returns in INSTRUMENTS order
    dates             : DatetimeIndex corresponding to rows of returns_arr
    linear_shares     : (4,) static inception shares
    linear_prices_now : (4,) current prices for linear instruments
    S_now, sigma_now, rate_now : current instrument state (SPY price,
                                 implied vol decimal, DGS10 decimal)
    K, T_now          : current straddle strike and time-to-maturity (years)
    var_today         : today's VaR estimate (positive, USD)
    n_worst           : number of worst scenarios to include in report

    Returns
    -------
    pd.DataFrame with columns rank, date, loss_usd, pnl_linear,
                               pnl_irs, pnl_straddle, r_SPY, r_VIX, r_DGS10
    """
    T_next = max(T_now - 1.0 / 252.0, 1.0 / 252.0)

    v_strad_now = price_straddle_vec(S_now, K, T_now, RF_RATE, sigma_now)
    v_irs_now   = price_irs_vec(IRS_NOTIONAL, IRS_FIXED_RATE, rate_now)
    dollar_pos  = linear_shares * linear_prices_now          # (4,)

    pnl_linear   = (dollar_pos[np.newaxis, :]
                    * (np.exp(returns_arr[:, IDX_LINEAR]) - 1.0)).sum(axis=1)
    rate_sim     = rate_now + returns_arr[:, IDX_DGS10]
    pnl_irs      = price_irs_vec(IRS_NOTIONAL, IRS_FIXED_RATE, rate_sim) - v_irs_now
    S_sim        = S_now    * np.exp(returns_arr[:, IDX_SPY])
    sig_sim      = sigma_now * np.exp(returns_arr[:, IDX_VIX])
    pnl_straddle = (price_straddle_vec(S_sim, K, T_next, RF_RATE, sig_sim)
                    - v_strad_now) * STRADDLE_SHARES
    pnl_total    = pnl_linear + pnl_irs + pnl_straddle

    worst_idx = np.argsort(pnl_total)[:n_worst]

    rows = []
    for rank, idx in enumerate(worst_idx, 1):
        rows.append({
            "rank"         : rank,
            "date"         : dates[idx].date(),
            "loss_usd"     : float(-pnl_total[idx]),
            "pnl_linear"   : float(pnl_linear[idx]),
            "pnl_irs"      : float(pnl_irs[idx]),
            "pnl_straddle" : float(pnl_straddle[idx]),
            "r_SPY"        : float(returns_arr[idx, IDX_SPY]),
            "r_VIX"        : float(returns_arr[idx, IDX_VIX]),
            "r_DGS10"      : float(returns_arr[idx, IDX_DGS10]),
        })

    df        = pd.DataFrame(rows)
    n_exceed  = int((pnl_total < -var_today).sum())
    n_total   = len(returns_arr)

    print(f"\n{'='*75}")
    print(f"STRESS TEST  —  Top {n_worst} worst historical scenarios "
          f"(current portfolio snapshot)")
    print(f"{'='*75}")
    print(f"  Scenarios evaluated : {n_total}")
    print(f"  VaR today           : ${var_today:>12,.0f}")
    print(f"  Scenarios > VaR     : {n_exceed} / {n_total}  "
          f"({100 * n_exceed / n_total:.1f}%)")
    print(f"  {'─'*73}")
    print(f"  {'':>1}{'Rank':>4}  {'Date':<12}  {'Loss $':>12}  "
          f"{'Linear $':>11}  {'IRS $':>11}  {'Straddle $':>11}  "
          f"{'r_SPY':>7}  {'r_VIX':>7}  {'r_DGS10':>8}")
    print(f"  {'─'*73}")
    for _, row in df.iterrows():
        flag = "✗" if row["loss_usd"] > var_today else " "
        print(f"  {flag}{int(row['rank']):>4}  {str(row['date']):<12}  "
              f"{row['loss_usd']:>12,.0f}  "
              f"{row['pnl_linear']:>11,.0f}  "
              f"{row['pnl_irs']:>11,.0f}  "
              f"{row['pnl_straddle']:>11,.0f}  "
              f"{row['r_SPY']:>7.2%}  "
              f"{row['r_VIX']:>7.2%}  "
              f"{row['r_DGS10']:>8.4f}")
    print(f"  {'─'*73}")
    print(f"  ✗ = loss exceeds today's VaR of ${var_today:,.0f}")
    print(f"{'='*75}")

    return df


# =============================================================================
# ROLLING LOOP  (V3: copula on raw returns; passes hist window to hybrid P&L)
# =============================================================================

def compute_rolling_var(returns_arr, dates, linear_prices, linear_shares,
                        spy_prices, vix_series, dgs10_series):
    """
    Walk-forward GARCH(1,1)-t copula VaR  —  v3.

    V3-CHG-1 (copula on raw returns):
      U_raw = compute_pseudo_observations(W)  — W is the raw return window.
      Q_ewma initialised from cov(W) for consistency.
      EWMA update still uses GARCH residuals z = (r-μ)/σ (innovation dynamics).

    V3-CHG-2 (hybrid P&L):
      scenarios_to_pnl_hybrid receives hist_window = returns_arr[t-WINDOW:t]
      for the historical bootstrap of IRS + straddle P&L.

    Returns
    -------
    var_arr   : (T - WINDOW,) array of daily VaR estimates
    refit_log : list of dicts with copula/GARCH parameters at each refit
    final_K   : straddle strike on the last simulation day (for stress test)
    final_T   : straddle time-to-maturity on the last simulation day
    """
    rng       = np.random.default_rng(SEED)
    T_len     = len(returns_arr)
    var_arr   = np.full(T_len, np.nan)
    refit_log = []

    omega_arr     = None
    alpha_arr     = None
    beta_arr      = None
    mu_arr        = None
    nu_marginals  = None
    sigma_current = None
    nu_copula_cached = None
    Q_ewma           = None

    K         = float(spy_prices.iloc[0])
    days_held = 0
    final_K   = K
    final_T   = max(STRADDLE_DAYS / 252.0, 1.0 / 252.0)

    print(f"\n{'='*65}")
    print(f"GARCH(1,1)-t Copula Monte Carlo VaR  v3")
    print(f"  CHG-1: copula on raw returns  |  CHG-2: hybrid P&L bootstrap")
    print(f"Window={WINDOW} | M={M:,} | α={ALPHA:.0%} | EWMA λ={EWMA_LAMBDA}")
    print(f"Factors: {INSTRUMENTS}")
    print(f"Computing VaR for {T_len - WINDOW} days …")
    print(f"{'='*65}")

    for t in range(WINDOW, T_len):

        need_refit = (omega_arr is None) or ((t - WINDOW) % REFIT_EVERY == 0)

        if need_refit:
            W = returns_arr[t - WINDOW : t]              # (WINDOW, N_ASSETS)

            std_resids, garch_params = fit_garch_marginals(W)

            omega_arr     = np.array([p['omega']      for p in garch_params])
            alpha_arr     = np.array([p['alpha']      for p in garch_params])
            beta_arr      = np.array([p['beta']       for p in garch_params])
            mu_arr        = np.array([p['mu']         for p in garch_params])
            nu_marginals  = np.array([p['nu']         for p in garch_params])
            sigma_current = np.array([p['sigma_last'] for p in garch_params])

            # V3-CHG-1: copula pseudo-obs from raw returns (not residuals)
            U_raw = compute_pseudo_observations(W)
            R_static, nu_copula_cached, _ = fit_t_copula(U_raw)

            # CvM GoF every 10th refit
            n_refits_so_far = len(refit_log)
            if n_refits_so_far % 10 == 0:
                cvm_stat, cvm_p = copula_gof_cvm(
                    U_raw, R_static, nu_copula_cached,
                    n_mc=200, rng_seed=SEED + n_refits_so_far
                )
                print(f"  CvM GoF: S={cvm_stat:.4f}  p={cvm_p:.3f}  "
                      f"{'OK' if cvm_p >= 0.05 else 'REJECT at 5%'}")

            # V3-CHG-1: EWMA init from raw return covariance (matches copula)
            Q_ewma = np.cov(W.T) + 1e-8 * np.eye(N_ASSETS)

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
                  f"ρ(SPY,GLD)={R_static[spy_idx, gld_idx]:+.3f} | "
                  f"α_SPY={alpha_arr[spy_idx]:.3f} "
                  f"β_SPY={beta_arr[spy_idx]:.3f}")

        # EWMA: innovation-based residuals (unchanged from v2)
        r_prev            = returns_arr[t - 1]
        x_prev            = r_prev - mu_arr
        z_prev            = x_prev / np.where(sigma_current > 1e-8,
                                               sigma_current, 1e-8)
        Q_ewma, R_dynamic = _ewma_corr_update(Q_ewma, z_prev, EWMA_LAMBDA)

        # GARCH one-step-ahead vol forecast
        sigma_sq_next  = (omega_arr
                          + alpha_arr * x_prev ** 2
                          + beta_arr  * sigma_current ** 2)
        sigma_forecast = np.sqrt(np.maximum(sigma_sq_next, 1e-12))
        sigma_current  = sigma_forecast.copy()

        # Instrument state
        linear_prices_now = linear_prices.iloc[t].values
        S_now     = float(spy_prices.iloc[t])
        sigma_now = float(vix_series.iloc[t]) / 100.0
        rate_now  = float(dgs10_series.iloc[t]) / 100.0

        if days_held >= STRADDLE_DAYS:
            K, days_held = S_now, 0
        T_now      = max((STRADDLE_DAYS - days_held) / 252.0, 1.0 / 252.0)
        days_held += 1

        # Track straddle state for stress test
        final_K = K
        final_T = T_now

        # V3-CHG-2: hybrid P&L
        hist_window = returns_arr[t - WINDOW : t]
        pnl_sim = scenarios_to_pnl_hybrid(
            nu_marginals, sigma_forecast,
            R_dynamic, nu_copula_cached, rng,
            linear_shares, linear_prices_now,
            hist_window,
            S_now, sigma_now, rate_now, K, T_now
        )
        var_arr[t] = extract_var(pnl_sim)

    return var_arr[WINDOW:], refit_log, final_K, final_T


# =============================================================================
# BACKTESTING  (unchanged from v2)
# =============================================================================

def backtest_var(pnl, var_series):
    bt = run_backtest(
        pnl=pnl,
        var=var_series,
        confidence=ALPHA,
        method_name="MC-GARCH-Copula-v3",
    )
    print(bt)
    return bt


# =============================================================================
# REFIT LOG PRINTER  (unchanged from v2)
# =============================================================================

def print_refit_log(refit_log):
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
# SAVE OUTPUTS
# =============================================================================

def save_outputs(dates_bt, var_series, pnl_bt, bt, stress_df):
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    var_path = DATA_DIR / "var_mc_garch_copula_v3.csv"
    pd.DataFrame(
        {"VaR_MC_GARCH_COPULA_V3": var_series.values}, index=dates_bt
    ).to_csv(var_path)

    exceptions = compute_exceptions(pnl_bt, var_series)
    bt_path = TAB_DIR / "backtest_mc_garch_copula_v3.csv"
    pd.DataFrame({
        "VaR_MC_GARCH_COPULA_V3" : var_series.values,
        "actual_loss"            : -pnl_bt.values,
        "exception"              : exceptions.values,
    }, index=dates_bt).to_csv(bt_path)

    st_path = TAB_DIR / "stress_test_mc_garch_copula_v3.csv"
    stress_df.to_csv(st_path, index=False)

    print(f"VaR series  → {var_path.relative_to(_ROOT)}")
    print(f"Backtest    → {bt_path.relative_to(_ROOT)}")
    print(f"Stress test → {st_path.relative_to(_ROOT)}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print(f"{'='*65}")
    print(f"GARCH(1,1)-t Copula Monte Carlo VaR  —  v3")
    print(f"  CHG-1: copula on raw returns (not GARCH residuals)")
    print(f"  CHG-2: hybrid P&L = MC linear + hist-bootstrap nonlinear")
    print(f"  CHG-3: stress test (top {N_STRESS_WORST} worst scenarios)")
    print(f"{'='*65}")

    factors, pnl_series, prices, vix_series, dgs10_series = load_data()
    returns_arr = factors.values
    dates       = factors.index

    # Static inception shares (same convention as v2 and compute_pnl.py)
    prices_linear = prices.rename(columns={"EURUSD=X": "EURUSD"})[LINEAR_TICKERS]
    prices_linear = prices_linear.reindex(dates, method="ffill")
    weights = pd.Series(
        {("EURUSD" if k == "EURUSD=X" else k): v for k, v in WEIGHTS_DICT.items()}
    )[LINEAR_TICKERS]
    initial_prices = prices_linear.iloc[0]
    linear_shares  = (V0 * weights / initial_prices).values
    print(f"Static inception shares: "
          + ", ".join(f"{t}={s:.2f}" for t, s in zip(LINEAR_TICKERS, linear_shares)))

    var_arr, refit_log, final_K, final_T = compute_rolling_var(
        returns_arr, dates, prices_linear, linear_shares,
        prices["SPY"], vix_series, dgs10_series
    )

    dates_bt   = dates[WINDOW:]
    pnl_bt     = pnl_series.iloc[WINDOW:]
    var_series  = pd.Series(var_arr, index=dates_bt)

    bt = backtest_var(pnl_bt, var_series)
    print_refit_log(refit_log)

    plot_all(bt, pnl=pnl_bt, var=var_series, save=True)

    # ── V3-CHG-3: stress test on final day's portfolio snapshot ──────────────
    final_t         = len(returns_arr) - 1
    stress_window   = returns_arr[final_t - WINDOW : final_t]
    stress_dates    = dates[final_t - WINDOW : final_t]
    final_prices_lin = prices_linear.iloc[final_t].values
    final_S         = float(prices["SPY"].iloc[final_t])
    final_sigma     = float(vix_series.iloc[final_t]) / 100.0
    final_rate      = float(dgs10_series.iloc[final_t]) / 100.0
    final_var       = float(var_arr[-1]) if not np.isnan(var_arr[-1]) else 0.0

    stress_df = run_stress_test(
        stress_window, stress_dates,
        linear_shares, final_prices_lin,
        final_S, final_sigma, final_rate,
        final_K, final_T,
        var_today=final_var,
        n_worst=N_STRESS_WORST,
    )

    save_outputs(dates_bt, var_series, pnl_bt, bt, stress_df)

    print(f"\n{'='*65}")
    print(f"GARCH-Copula v3 complete!")
    print(f"  Breaches  : {bt.N}  (expected {bt.expected_N})")
    print(f"  Kupiec    : LR={bt.lr_uc:.4f}  p={bt.pvalue_uc:.4f}  "
          f"{'RETAINED' if not bt.reject_uc else 'REJECTED'}")
    print(f"  Christoff.: LR_IND={bt.lr_ind:.4f}  p={bt.pvalue_ind:.4f}  "
          f"LR_CC={bt.lr_cc:.4f}  p={bt.pvalue_cc:.4f}")
    print(f"  Mean VaR  : ${np.nanmean(var_arr):,.0f}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
