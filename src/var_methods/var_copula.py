# =============================================================================
# var_copula.py  —  Copula-Based Value-at-Risk (Multivariate Dependence Model)
#
# This file implements the copula-based VaR benchmark described in Irle's
# lecture (Section 7).  It is a SEPARATE model from evt.py (unconditional EVT)
# and garch_evt.py (conditional GARCH+EVT); copula logic is not merged into
# those files.
#
# The preferred production VaR model for this portfolio is garch_evt.py when
# it passes both Kupiec and Christoffersen — its GARCH pre-filtering makes the
# i.i.d. assumption more plausible.  var_copula.py is retained because it
# implements the lecture's multivariate dependence approach: Sklar's theorem,
# a parametric t-copula, and tail dependence coefficients.
#
# =============================================================================
# METHODOLOGY
# =============================================================================
#
# Sklar's theorem decomposes a joint distribution into marginal CDFs and a
# copula, motivating separate modelling of marginals and dependence.  We fit
# AIC-selected parametric marginals and a Student-t copula with fixed ν = 4 to
# the four linear return factors (EURUSD, GLD, IEF, SPY).  Monte Carlo
# scenarios are drawn from the copula, inverted through the marginals, and
# the portfolio P&L distribution is simulated.  VaR = −Q_{1−α}(simulated P&L)
# = Q_α(simulated loss) (lecture Section 6).
#
# Marginals:
#   Six candidate families — Normal, Logistic, Student-t, Laplace,
#   Generalised-Normal, Generalised-Logistic — are fitted per instrument per
#   window and the lowest-AIC family is selected.
#   scipy.stats.dist.fit() is the Python equivalent of fitdistr() in R
#   (MASS package); .pdf()/.cdf()/.ppf() correspond to d/p/q functions.
#
# t-copula:
#   Degrees of freedom fixed at ν = 4, matching the course implementation
#   tCopula(df = 4, df.fixed = TRUE) in R (copula package).  Estimating ν
#   would be more flexible (McNeil-Frey-Embrechts 2005) but departs from the
#   course setup.  The correlation matrix R is estimated by rank-based
#   pseudo-likelihood (canonical maximum likelihood): the copula step uses
#   t_ν-transformed rank pseudo-observations rather than parametric marginal
#   CDFs.  This is an IFM-style two-step workflow — parametric marginals are
#   fitted first and used later for inverse simulation — but the copula
#   correlation step itself is rank-based.  The sample correlation of the
#   transformed pseudo-observations is the L-BFGS-B starting point and
#   fallback if optimisation does not improve the objective.
#
# GoF diagnostic:
#   Simulation-based CvM-style statistic (lecture §7): a squared distance
#   between two simulated copula samples rather than the exact parametric CDF.
#   Large values indicate poor fit (upper-tail rejection only).  p-values are
#   diagnostics, not proof of correct model specification.
#
# Rolling window: WINDOW = 500 days, refit every COPULA_REFIT = 250 days.
# RNG: daily VaR uses a deterministic per-date seed; GoF bootstrap, diagnostic
#      plots, and bootstrap CI use separate independent generators.
#
# =============================================================================
# LECTURE ALIGNMENT
# =============================================================================
#
# Section 4:  log returns imply S_t = S_0 exp(r_t); simulated log returns
#             are converted to discrete changes by exp(r) − 1 before
#             aggregating portfolio P&L.
# Section 6:  MC generates factor-change scenarios, revalues the portfolio,
#             and estimates VaR from the empirical lower-tail quantile of
#             simulated P&L.
# Section 7:  copulas separate marginals from dependence (Sklar's theorem);
#             the t-copula has positive upper and lower tail dependence for
#             ρ > −1, whereas the Gaussian copula has zero tail dependence.
#             Tail dependence coefficients are computed and saved per asset pair.
# Section 8:  financial returns are generally not i.i.d.; volatility
#             clustering and clustered extremes are known limitations when
#             applying the copula to unfiltered raw returns.
#
# =============================================================================
# LIMITATIONS
# =============================================================================
#
# - Raw returns are not volatility-filtered, so pseudo-observations may still
#   exhibit clustering.  Ljung-Box diagnostics on raw and squared returns are
#   provided; GARCH+EVT (garch_evt.py) addresses this via the GARCH filter.
# - Fixed ν = 4 may not match the true joint tail heaviness.
# - IRS and straddle P&L are not modelled within the copula; they are added by
#   an independent historical bootstrap so that simulated P&L can be
#   backtested against pnl_total.  This is a project-specific extension, not
#   part of the pure lecture copula model.
# - MC estimation uncertainty: for a 1% tail probability with 20,000 paths,
#   the binomial standard error is sqrt(0.01 × 0.99 / 20,000) ≈ 0.07 pp;
#   a rough 95% band is ±0.14 pp.  Dollar VaR uncertainty also depends on
#   tail density.
# - KS and CvM goodness-of-fit p-values are diagnostics only; parameters are
#   estimated from the same data, so nominal p-values are not calibrated for
#   post-estimation testing.
#
# =============================================================================

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from scipy.linalg import eigh
from scipy.optimize import minimize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── add project root to path so backtesting module is importable ─────────────
_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[2]
sys.path.insert(0, str(_ROOT))
from backtesting.backtest import run_backtest

# ─── SETTINGS ─────────────────────────────────────────────────────────────────
WINDOW          = 500        # rolling window length (days)
ALPHA           = 0.99       # VaR confidence level
COPULA_REFIT    = 250        # refit every N days
N_SIMS          = 20_000     # Monte Carlo paths per VaR date
NU_T_COPULA     = 4          # t-copula df, fixed — matches tCopula(df=4, df.fixed=TRUE) in R
N_GOF_BOOTSTRAP = 1_000      # CvM null-distribution samples
N_VAR_BOOTSTRAP = 1_000      # VaR bootstrap CI samples
RANDOM_SEED     = 42
V0              = 1_000_000  # portfolio value (USD)
N_ASSETS        = 4          # EURUSD, GLD, IEF, SPY
INSTRUMENT_NAMES = ["EURUSD", "GLD", "IEF", "SPY"]

# ─── PATHS ────────────────────────────────────────────────────────────────────
DATA = _ROOT / "data" / "processed"
FIGS = _ROOT / "outputs" / "figures"
TABS = _ROOT / "outputs" / "tables"
FIGS.mkdir(parents=True, exist_ok=True)
TABS.mkdir(parents=True, exist_ok=True)

# ─── MARGINAL CANDIDATES ──────────────────────────────────────────────────────
_MARGINAL_CANDIDATES = [
    ("norm",        stats.norm),
    ("logistic",    stats.logistic),
    ("t",           stats.t),
    ("laplace",     stats.laplace),
    ("gennorm",     stats.gennorm),
    ("genlogistic", stats.genlogistic),
]


# ─── DATA LOADING ─────────────────────────────────────────────────────────────

def _validate_aligned(df_or_series, name):
    """Raise ValueError if the aligned dataset fails basic quality checks."""
    idx  = df_or_series.index
    vals = df_or_series.values.astype(float)

    if len(idx) == 0:
        raise ValueError(f"load_data: {name} is empty after date alignment.")
    if not idx.is_monotonic_increasing:
        raise ValueError(f"load_data: {name} index is not sorted ascending.")
    dups = int(idx.duplicated().sum())
    if dups:
        raise ValueError(f"load_data: {name} has {dups} duplicate date(s).")
    nan_n = int(np.sum(np.isnan(vals)))
    if nan_n:
        raise ValueError(f"load_data: {name} contains {nan_n} NaN value(s).")
    inf_n = int(np.sum(np.isinf(vals)))
    if inf_n:
        raise ValueError(f"load_data: {name} contains {inf_n} Inf/-Inf value(s).")


def load_data():
    """
    Load per-asset log-returns, total portfolio P&L, and nonlinear P&L.

    Returns
    -------
    lr        : pd.DataFrame  log-returns for 4 linear assets
    pnl_total : pd.Series     total portfolio P&L (for backtesting)
    nonlinear : pd.Series or None
                IRS + straddle P&L summed; None if instrument_pnl.csv is
                unavailable.  Used in the hybrid MC extension below.
    """
    lr  = pd.read_csv(DATA / "log_returns.csv",         index_col=0, parse_dates=True)
    pnl = pd.read_csv(DATA / "total_portfolio_pnl.csv", index_col=0, parse_dates=True)

    nonlinear = None
    try:
        inst   = pd.read_csv(DATA / "instrument_pnl.csv", index_col=0, parse_dates=True)
        common = lr.index.intersection(pnl.index).intersection(inst.index)
        n_dropped = len(lr.index.union(pnl.index).union(inst.index)) - len(common)
        if n_dropped:
            print(f"  load_data: date intersection dropped {n_dropped} row(s).")
        inst   = inst.loc[common]
        if "pnl_irs" in inst.columns and "pnl_straddle" in inst.columns:
            nonlinear = inst[["pnl_irs", "pnl_straddle"]].sum(axis=1)
            nonlinear.name = "pnl_nonlinear"
        else:
            warnings.warn(
                "load_data: instrument_pnl.csv missing pnl_irs or pnl_straddle; "
                "falling back to linear-only VaR."
            )
            nonlinear = None
            common    = lr.index.intersection(pnl.index)
    except Exception as exc:
        warnings.warn(
            f"load_data: instrument_pnl.csv not found or unreadable "
            f"({type(exc).__name__}: {exc}); falling back to linear-only VaR."
        )
        common = lr.index.intersection(pnl.index)

    missing_cols = [c for c in INSTRUMENT_NAMES if c not in lr.columns]
    if missing_cols:
        raise ValueError(
            f"load_data: log_returns.csv missing required columns: {missing_cols}"
        )

    lr  = lr.loc[common, INSTRUMENT_NAMES]
    pnl = pnl.loc[common]
    if nonlinear is not None:
        nonlinear = nonlinear.loc[common]

    # Hard validation after alignment — raise on any data quality issue.
    _validate_aligned(lr,              "log_returns")
    _validate_aligned(pnl["pnl_total"], "pnl_total")
    if nonlinear is not None:
        _validate_aligned(nonlinear, "pnl_nonlinear")

    print(f"Loaded: {len(common)} days  |  {common[0].date()} -> {common[-1].date()}")
    print(f"Instruments: {INSTRUMENT_NAMES}")
    print(f"r mean (each):  {lr.mean().round(6).to_dict()}")
    print(f"pnl_total std:  ${pnl['pnl_total'].std():,.0f}")
    nl_label = f"${nonlinear.std():,.0f}" if nonlinear is not None else "not loaded"
    print(f"nonlinear std:  {nl_label}")
    return lr, pnl["pnl_total"], nonlinear


# ─── MARGINAL FITTING ─────────────────────────────────────────────────────────

def fit_best_marginal(x, instrument_name="?", refit_id="?"):
    """
    AIC-based marginal selection from six parametric families.

    scipy.stats distributions provide the same parametric families available
    through fitdistr() (MASS) in R; .fit() maximises the log-likelihood.
    Returns dict: dist_name, params, frozen, aic, ks_pvalue, aic_table.
    KS p-values are diagnostic only; parameters are estimated from the same
    data, so nominal p-values are not calibrated for post-estimation testing.
    """
    results = []
    for name, dist_cls in _MARGINAL_CANDIDATES:
        try:
            params = dist_cls.fit(x)
            loglik = np.sum(dist_cls.logpdf(x, *params))
            if not np.isfinite(loglik):
                raise ValueError("non-finite log-likelihood")
            k   = len(params)
            aic = 2 * k - 2 * loglik
            results.append(dict(name=name, dist_cls=dist_cls, params=params,
                                aic=aic, loglik=loglik))
        except Exception as exc:
            warnings.warn(
                f"fit_best_marginal [{instrument_name} @ {refit_id}]: "
                f"{name} failed: {type(exc).__name__}: {exc}"
            )

    if not results:
        warnings.warn(
            f"fit_best_marginal [{instrument_name} @ {refit_id}]: all candidates "
            f"failed; falling back to Normal."
        )
        params = stats.norm.fit(x)
        frozen = stats.norm(*params)
        return dict(dist_name="norm", params=params, frozen=frozen,
                    aic=np.nan, ks_pvalue=np.nan, aic_table=pd.DataFrame())

    results.sort(key=lambda r: r["aic"])
    best = results[0]

    try:
        _, ks_p = stats.kstest(x, best["dist_cls"].cdf, args=best["params"])
    except Exception as exc:
        warnings.warn(
            f"fit_best_marginal [{instrument_name} @ {refit_id}]: "
            f"KS test failed: {type(exc).__name__}: {exc}"
        )
        ks_p = np.nan

    frozen    = best["dist_cls"](*best["params"])
    aic_table = pd.DataFrame(
        [{"name": r["name"], "aic": r["aic"], "loglik": r["loglik"]} for r in results]
    )
    return dict(dist_name=best["name"], params=best["params"], frozen=frozen,
                aic=best["aic"], ks_pvalue=ks_p, aic_table=aic_table)


# ─── COPULA ───────────────────────────────────────────────────────────────────

def pseudo_observations(X):
    """
    Rank-based pseudo-observations using rank/(n+1), matching the common
    copula::pobs() convention in R.  Using n+1 rather than n keeps
    pseudo-observations strictly inside (0, 1), which avoids infinite
    inverse-t values at the boundary.
    Returns array (n, d) with values in (0, 1).
    """
    n, d = X.shape
    U    = np.zeros_like(X, dtype=float)
    for j in range(d):
        ranks   = stats.rankdata(X[:, j])   # 1-based
        U[:, j] = ranks / (n + 1)
    return U


def _nearest_pd(A):
    """Project symmetric matrix A to nearest positive-definite matrix (Higham 1988)."""
    B          = (A + A.T) / 2
    vals, vecs = eigh(B)
    vals       = np.maximum(vals, 1e-8)
    return vecs @ np.diag(vals) @ vecs.T


def _nearest_corr(A):
    """
    Project a symmetric matrix to a positive-definite correlation matrix.

    _nearest_pd() makes the matrix positive definite but may leave diagonal
    entries different from 1.  A copula correlation matrix must have unit
    diagonal, so we project to PD and then rescale to correlation form.
    """
    B = _nearest_pd(A)
    d = np.sqrt(np.diag(B))
    C = B / np.outer(d, d)
    C = (C + C.T) / 2      # re-symmetrise after floating-point rescaling
    np.fill_diagonal(C, 1.0)
    return C


def _t_copula_negloglik(params, T_emp, nu):
    """
    Negative pseudo log-likelihood for the t-copula with fixed nu.

    Maximising over R reduces to minimising
        n/2 * log|R| + (nu+d)/2 * sum_i log(1 + T_i' R^{-1} T_i / nu)
    where T_i = t_nu^{-1}(U_i) are the t-transformed pseudo-observations.
    Terms that do not depend on R are omitted.
    """
    n, d = T_emp.shape
    R    = np.eye(d)
    idx  = np.triu_indices(d, k=1)
    R[idx] = params
    R      = R + R.T - np.eye(d)

    eigvals = np.linalg.eigvalsh(R)
    if eigvals.min() < 1e-8:
        return 1e15

    try:
        sign, logdet = np.linalg.slogdet(R)
        if sign <= 0:
            return 1e15
        R_inv = np.linalg.inv(R)
    except np.linalg.LinAlgError:
        return 1e15

    quad   = np.einsum("ni,ij,nj->n", T_emp, R_inv, T_emp)
    loglik = -0.5 * n * logdet - 0.5 * (nu + d) * np.sum(np.log(1.0 + quad / nu))
    return -loglik


def fit_t_copula(U_emp, nu=NU_T_COPULA, refit_id="?"):
    """
    Estimate the t-copula correlation matrix R from pseudo-observations (n, d).

    Method: rank-based pseudo-likelihood (canonical maximum likelihood) with
    fixed nu.  Input U_emp must be rank pseudo-observations in (0, 1); the
    copula step does NOT use parametric marginal CDFs — it transforms ranks
    through the t_nu quantile function directly.  Parametric marginals are
    fitted separately and used later only for inverse simulation (IFM-style
    two-step workflow, but rank-based copula estimation).  R is optimised by
    minimising the negative t-copula pseudo log-likelihood (_t_copula_negloglik)
    via L-BFGS-B.  The sample correlation of t_nu-transformed pseudo-observations
    serves as both the starting point and the fallback if optimisation does not
    improve the objective.
    In R this corresponds conceptually to fitCopula(tCopula(df=4, df.fixed=TRUE), data).
    """
    t_dist = stats.t(df=nu)
    eps    = 1e-6
    T_emp  = t_dist.ppf(np.clip(U_emp, eps, 1 - eps))
    R_init = np.corrcoef(T_emp.T)   # plug-in starting point

    # Project to nearest correlation matrix if not PD; avoids returning 1e15
    # at the first likelihood evaluation on numerically near-singular windows.
    if np.linalg.eigvalsh(R_init).min() < 1e-8:
        warnings.warn(
            f"fit_t_copula [{refit_id}]: plug-in R not PD; "
            f"projecting to nearest correlation matrix before optimisation."
        )
        R_init = _nearest_corr(R_init)

    d   = T_emp.shape[1]
    idx = np.triu_indices(d, k=1)
    x0  = R_init[idx]
    f0  = _t_copula_negloglik(x0, T_emp, nu)

    R = R_init
    try:
        bounds = [(-0.999, 0.999)] * len(x0)
        res    = minimize(
            _t_copula_negloglik, x0, args=(T_emp, nu),
            method="L-BFGS-B", bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-10},
        )
        if res.success and np.isfinite(res.fun) and res.fun < f0:
            R_opt      = np.eye(d)
            R_opt[idx] = res.x
            R_opt      = R_opt + R_opt.T - np.eye(d)
            if np.linalg.eigvalsh(R_opt).min() >= 1e-8:
                R = R_opt
            else:
                warnings.warn(
                    f"fit_t_copula [{refit_id}]: optimised R not PD; "
                    f"using plug-in correlation estimator."
                )
        else:
            warnings.warn(
                f"fit_t_copula [{refit_id}]: optimisation did not converge or "
                f"did not improve over starting point; "
                f"using plug-in correlation estimator."
            )
    except Exception as exc:
        warnings.warn(
            f"fit_t_copula [{refit_id}]: optimisation raised "
            f"{type(exc).__name__}: {exc}; using plug-in correlation estimator."
        )

    if np.linalg.eigvalsh(R).min() < 1e-8:
        warnings.warn(
            f"fit_t_copula [{refit_id}]: R not PD; "
            f"projecting to nearest correlation matrix."
        )
        R = _nearest_corr(R)

    return R


def sample_t_copula(R, nu, n_samples, rng):
    """
    Sample n_samples observations from t-copula(R, ν).
    Algorithm: Y ~ N(0, R), W ~ χ²(ν), T = Y√(ν/W), U = F_{t,ν}(T).
    Equivalent to rCopula(n, tCopula(R, df=nu)) in R.
    Returns (n_samples, d) array in (0, 1).
    """
    try:
        L = np.linalg.cholesky(R)
    except np.linalg.LinAlgError:
        R = _nearest_corr(R)
        try:
            L = np.linalg.cholesky(R)
        except np.linalg.LinAlgError:
            R_j = R + 1e-6 * np.eye(R.shape[0])
            np.fill_diagonal(R_j, 1.0)
            L   = np.linalg.cholesky(R_j)
            R   = R_j

    d = R.shape[0]
    Z = rng.standard_normal((n_samples, d))
    Y = Z @ L.T
    W = rng.chisquare(nu, size=n_samples)
    T = Y * np.sqrt(nu / W[:, np.newaxis])
    U = stats.t.cdf(T, df=nu)
    return U


# ─── TAIL DEPENDENCE ──────────────────────────────────────────────────────────

def copula_tail_dependence_table(R, U_emp, nu, refit_label):
    """
    Compute t-copula dependence and tail dependence measures for all asset pairs.

    Spearman's rho (lecture-aligned primary rank measure):
        rho_S = corr(F_i(X_i), F_j(X_j))
    Empirically approximated as the Pearson correlation of rank pseudo-observations:
        spearman_rho_emp = corr(U_emp[:, i], U_emp[:, j])
    This is the standard rank correlation used in the lecture to characterise
    copula dependence; it does not assume any parametric copula form.

    t-copula tail dependence (equal upper and lower by symmetry of the t distribution):
        lambda = 2 * t_{nu+1}(-sqrt((nu+1)*(1-rho)/(1+rho)))
    The t-copula has positive upper and lower tail dependence for rho > -1,
    whereas the Gaussian copula has zero tail dependence.

    Kendall's tau (additional diagnostic; exact for elliptical copulas):
        tau = (2/pi) * arcsin(rho)
    Invariant under monotone marginal transforms.  In R: cor(u, method="kendall").

    Returns list of dicts: refit_id, asset_i, asset_j, rho, spearman_rho,
    kendall_tau, lambda_lower, lambda_upper.
    """
    rows   = []
    d      = R.shape[0]
    t_next = stats.t(df=nu + 1)
    for i in range(d):
        for j in range(i + 1, d):
            rho       = R[i, j]
            rho_c     = np.clip(rho, -0.9999, 0.9999)
            lam       = 2.0 * t_next.cdf(-np.sqrt((nu + 1) * (1 - rho_c) / (1 + rho_c)))
            tau       = (2.0 / np.pi) * np.arcsin(rho_c)
            spear_emp = float(np.corrcoef(U_emp[:, i], U_emp[:, j])[0, 1])
            rows.append({
                "refit_id":     refit_label,
                "asset_i":      INSTRUMENT_NAMES[i],
                "asset_j":      INSTRUMENT_NAMES[j],
                "rho":          float(rho),
                "spearman_rho": spear_emp,
                "kendall_tau":  float(tau),
                "lambda_lower": float(lam),
                "lambda_upper": float(lam),
            })
    return rows


# ─── CVM GOF DIAGNOSTIC ───────────────────────────────────────────────────────

def eval_empirical_copula(eval_pts, data_pts):
    """
    Vectorised empirical copula C_n(u) evaluated at multiple u.

    eval_pts : (m, d) – points to evaluate
    data_pts : (n, d) – pseudo-observations
    Returns  : (m,) array, C_n[i] = #{j : data_pts[j] ≤ eval_pts[i] componentwise} / n
    """
    mask = np.all(
        data_pts[np.newaxis, :, :] <= eval_pts[:, np.newaxis, :],
        axis=2,
    )
    return mask.mean(axis=1)


def cvm_gof_test(U_emp, R, nu, n_bootstrap, rng, refit_id="?"):
    """
    Simulation-based CvM-style GoF diagnostic for the t-copula.

    The statistic measures the squared distance between the model copula
    (approximated by a fixed reference sample U_ref) and the empirical copula,
    evaluated at the same set of points.  This is still simulation-based rather
    than exact because C_model is approximated via U_ref, not the analytical
    t-copula CDF.  Large values indicate poor fit; rejection is upper-tail only.

    Observed statistic:
        C_emp_obs   = eval_empirical_copula(U_emp, U_emp)   # empirical at data pts
        C_model_obs = eval_empirical_copula(U_emp, U_ref)   # model approx at data pts
        CvM_obs     = sum((C_model_obs - C_emp_obs)^2)

    Null distribution (under H0: data comes from t-copula(R, nu)):
        For each bootstrap b, draw U_b ~ t-copula(R, nu), then:
        C_emp_b   = eval_empirical_copula(U_b, U_b)    # empirical at U_b pts
        C_model_b = eval_empirical_copula(U_b, U_ref)  # model approx at U_b pts
        CvM_null[b] = sum((C_model_b - C_emp_b)^2)
    The same U_ref is used throughout so model approximation error is consistent.

    p-value = (1 + #{boot stats >= CvM_obs}) / (n_bootstrap + 1)
              (conservative, avoids p=0 at small sample sizes)

    Returns dict: cvm_obs, q025, q95, q975, reject_5pct, p_value, cvm_null
    """
    n, _ = U_emp.shape

    def _sample(n_draw):
        return sample_t_copula(R, nu, n_draw, rng)

    U_ref       = _sample(n)
    C_emp_obs   = eval_empirical_copula(U_emp, U_emp)
    C_model_obs = eval_empirical_copula(U_emp, U_ref)
    CvM_obs     = float(np.sum((C_model_obs - C_emp_obs) ** 2))

    CvM_null = np.zeros(n_bootstrap)
    for b in range(n_bootstrap):
        U_b          = _sample(n)
        C_emp_b      = eval_empirical_copula(U_b, U_b)
        C_model_b    = eval_empirical_copula(U_b, U_ref)
        CvM_null[b]  = np.sum((C_model_b - C_emp_b) ** 2)

    q025        = float(np.percentile(CvM_null, 2.5))
    q95         = float(np.percentile(CvM_null, 95.0))
    q975        = float(np.percentile(CvM_null, 97.5))
    reject_5pct = bool(CvM_obs > q95)
    p_val       = (1 + int(np.sum(CvM_null >= CvM_obs))) / (n_bootstrap + 1)

    return dict(cvm_obs=CvM_obs, q025=q025, q95=q95, q975=q975,
                reject_5pct=reject_5pct, p_value=float(p_val),
                cvm_null=CvM_null)


# ─── COMBINED REFIT ───────────────────────────────────────────────────────────

def fit_marginals_and_copula(window_data, nu, refit_label, run_gof,
                              n_gof_bootstrap, rng_gof):
    """
    Fit AIC-selected marginals + t-copula on window_data (n, d).
    Optionally runs CvM GoF diagnostic.
    Returns dict: marginals, R, U_emp, marginal_records, gof (or None)
    """
    marginals        = []
    marginal_records = []

    for j, name in enumerate(INSTRUMENT_NAMES):
        x    = window_data[:, j]
        info = fit_best_marginal(x, instrument_name=name, refit_id=refit_label)
        marginals.append(info["frozen"])
        marginal_records.append({
            "refit_id":   refit_label,
            "instrument": name,
            "dist_name":  info["dist_name"],
            "aic":        info["aic"],
            "ks_pvalue":  info["ks_pvalue"],
            "params":     str(info["params"]),
            "aic_normal": next(
                (r["aic"] for r in info["aic_table"].to_dict("records")
                 if r["name"] == "norm"),
                np.nan,
            ),
        })

    U_emp = pseudo_observations(window_data)
    R     = fit_t_copula(U_emp, nu=nu, refit_id=refit_label)

    gof = None
    if run_gof:
        gof = cvm_gof_test(U_emp, R, nu, n_gof_bootstrap, rng_gof,
                           refit_id=refit_label)

    return dict(marginals=marginals, R=R, U_emp=U_emp,
                marginal_records=marginal_records, gof=gof)


# ─── MONTE CARLO VAR ──────────────────────────────────────────────────────────

def monte_carlo_var(fit_cache, n_sims, rng, nonlinear_window=None):
    """
    Draw n_sims copula scenarios, invert marginals, compute portfolio P&L.

    P&L mapping (lecture Section 4): simulated factors are log returns.
    exp(r) − 1 converts them to discrete returns.  Equal portfolio weights:
        weights      = [1/4, 1/4, 1/4, 1/4]
        PnL_linear_i = V0 * sum_j weights[j] * (exp(r_{j,i}) − 1)
    VaR is the negative (1−alpha) lower quantile of the simulated P&L
    distribution (lecture Section 6).

    Hybrid extension (project-specific, not the pure lecture model):
    IRS and straddle P&L are added by an independent historical bootstrap so
    that simulated P&L can be backtested against pnl_total.  This reduces the
    scope mismatch but does not model dependence between nonlinear P&L and the
    linear factors.

    Returns: (VaR, PnL_sim array)
    """
    R, marginals = fit_cache["R"], fit_cache["marginals"]
    U_sim = sample_t_copula(R, NU_T_COPULA, n_sims, rng)
    eps   = 1e-6
    r_sim = np.column_stack([
        marginals[j].ppf(np.clip(U_sim[:, j], eps, 1 - eps))
        for j in range(len(marginals))
    ])

    weights    = np.full(N_ASSETS, 1.0 / N_ASSETS)
    PnL_linear = V0 * ((np.exp(r_sim) - 1.0) @ weights)

    if nonlinear_window is not None and len(nonlinear_window) > 0:
        idx     = rng.integers(0, len(nonlinear_window), size=n_sims)
        PnL_sim = PnL_linear + nonlinear_window[idx]
    else:
        PnL_sim = PnL_linear

    VaR = max(-np.percentile(PnL_sim, (1 - ALPHA) * 100), 0.0)
    return VaR, PnL_sim


# ─── VAR BOOTSTRAP CI ─────────────────────────────────────────────────────────

def var_bootstrap_ci(PnL_sim, n_bootstrap, rng):
    """Bootstrap CI for VaR by resampling the MC P&L paths. Returns (ci_lo, ci_hi)."""
    n      = len(PnL_sim)
    VaR_bs = np.zeros(n_bootstrap)
    for b in range(n_bootstrap):
        idx       = rng.integers(0, n, size=n)
        VaR_bs[b] = -np.percentile(PnL_sim[idx], (1 - ALPHA) * 100)
    return float(np.percentile(VaR_bs, 2.5)), float(np.percentile(VaR_bs, 97.5))


# ─── RETURN DIAGNOSTICS ───────────────────────────────────────────────────────

def compute_return_diagnostics(returns_df):
    """
    Ljung-Box tests on raw returns and squared returns (lags 10 and 20).

    Low p-value on squared returns is consistent with volatility clustering
    (lecture Section 8), a known limitation of raw-return copula VaR that
    motivates GARCH+EVT as a conditional benchmark.
    KS p-values for marginal fits are diagnostic only (see module docstring).
    """
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox
    except ImportError:
        warnings.warn("statsmodels not available; skipping Ljung-Box diagnostics.")
        return None

    records = []
    for col in returns_df.columns:
        x  = returns_df[col].dropna().values
        x2 = x ** 2
        for lags in [10, 20]:
            for series, slabel in [(x, "returns"), (x2, "squared_returns")]:
                try:
                    res     = acorr_ljungbox(series, lags=[lags], return_df=True)
                    lb_stat = float(res["lb_stat"].iloc[-1])
                    lb_pval = float(res["lb_pvalue"].iloc[-1])
                except Exception as exc:
                    lb_stat = np.nan
                    lb_pval = np.nan
                    warnings.warn(
                        f"Ljung-Box [{col}/{slabel}/lag{lags}]: "
                        f"{type(exc).__name__}: {exc}"
                    )
                records.append({
                    "instrument": col,
                    "series":     slabel,
                    "lags":       lags,
                    "lb_stat":    lb_stat,
                    "lb_pvalue":  lb_pval,
                })

    df = pd.DataFrame(records)
    df.to_csv(TABS / "copula_return_diagnostics.csv", index=False)
    print(f"Return diagnostics -> {TABS / 'copula_return_diagnostics.csv'}")

    sq = df[df["series"] == "squared_returns"]
    n_cluster = int((sq["lb_pvalue"] < 0.05).sum())
    if n_cluster:
        print(
            f"  Ljung-Box (squared returns): {n_cluster}/{len(sq)} tests reject "
            f"i.i.d. at 5% — evidence of volatility clustering."
        )
        print(
            "  This is a limitation of raw-return copula VaR; "
            "GARCH+EVT is the conditional benchmark."
        )
    else:
        print("  Ljung-Box (squared returns): no strong evidence of volatility clustering.")

    return df


# ─── PLOTS ────────────────────────────────────────────────────────────────────

def _save(path):
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  Figure saved -> {path}")


def plot_marginals(window_data, marginals, U_emp, U_sim, refit_label):
    """
    var_copula_marginals_<date>.png  : histograms + QQ plots for all 4 instruments
    var_copula_dependence_<date>.png : pairwise pseudo-obs scatter for all 6 pairs
                                       with empirical and t-copula simulated overlay
    """
    # ── marginals: 4 instruments × (histogram + QQ) ──────────────────────────
    n_inst = len(INSTRUMENT_NAMES)
    fig, axes = plt.subplots(n_inst, 2, figsize=(10, 3 * n_inst))
    fig.suptitle(f"Copula: marginal diagnostics  [{refit_label}]", fontsize=11)

    for row, name in enumerate(INSTRUMENT_NAMES):
        x      = window_data[:, row]
        frozen = marginals[row]
        ax_h   = axes[row, 0]
        ax_q   = axes[row, 1]

        xg = np.linspace(x.min(), x.max(), 300)
        ax_h.hist(x, bins=40, density=True, alpha=0.5, color="steelblue",
                  label="empirical")
        ax_h.plot(xg, frozen.pdf(xg), "r-", lw=1.5,
                  label=f"{frozen.dist.name} fit")
        ax_h.set_title(name); ax_h.legend(fontsize=7)

        probs = (np.arange(1, len(x) + 1)) / (len(x) + 1)
        q_emp = np.sort(x)
        q_the = frozen.ppf(probs)
        ax_q.scatter(q_the, q_emp, s=4, alpha=0.5, color="steelblue")
        lim = (min(q_the.min(), q_emp.min()), max(q_the.max(), q_emp.max()))
        ax_q.plot(lim, lim, "r--", lw=1)
        ax_q.set_xlabel("theoretical"); ax_q.set_ylabel("sample")
        ax_q.set_title(f"QQ – {name}")

    _save(FIGS / f"var_copula_marginals_{refit_label}.png")

    # ── dependence: all 6 pairwise scatters ──────────────────────────────────
    pairs = [(i, j) for i in range(n_inst) for j in range(i + 1, n_inst)]
    n_pairs = len(pairs)  # 6 for 4 instruments
    ncols = 3
    nrows = (n_pairs + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes_flat = axes.flatten()
    fig.suptitle(f"Copula: pairwise pseudo-observations  [{refit_label}]",
                 fontsize=11)

    for k, (i, j) in enumerate(pairs):
        ax = axes_flat[k]
        ax.scatter(U_emp[:, i], U_emp[:, j], s=4, alpha=0.4,
                   color="steelblue", label="empirical")
        ax.scatter(U_sim[:, i], U_sim[:, j], s=2, alpha=0.2,
                   color="tomato", label="t-copula sim")
        ax.set_xlabel(INSTRUMENT_NAMES[i], fontsize=8)
        ax.set_ylabel(INSTRUMENT_NAMES[j], fontsize=8)
        ax.set_title(f"{INSTRUMENT_NAMES[i]} vs {INSTRUMENT_NAMES[j]}", fontsize=9)
        if k == 0:
            ax.legend(fontsize=6)

    for ax in axes_flat[n_pairs:]:
        ax.set_visible(False)

    _save(FIGS / f"var_copula_dependence_{refit_label}.png")


def plot_cvm_null(gof, refit_label):
    """var_copula_cvm_null_<date>.png – null distribution histogram."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(gof["cvm_null"], bins=50, density=True, alpha=0.6, color="steelblue",
            label="Bootstrap null")
    ax.axvline(gof["cvm_obs"], color="red", lw=2,
               label=f"Observed CvM = {gof['cvm_obs']:.4f}")
    ax.axvline(gof["q95"], color="orange", lw=1.5, ls="--",
               label=f"95% = {gof['q95']:.4f}")
    ax.axvline(gof["q975"], color="gold", lw=1.5, ls="--",
               label=f"97.5% = {gof['q975']:.4f}")
    reject_txt = "REJECT (5%)" if gof["reject_5pct"] else "fail to reject"
    ax.set_title(
        f"CvM GoF diagnostic  [{refit_label}]  p={gof['p_value']:.3f}  {reject_txt}"
    )
    ax.legend(fontsize=8)
    _save(FIGS / f"var_copula_cvm_null_{refit_label}.png")


def plot_var_results(dates, pnl, VaR_series, backtest_result):
    """var_copula_03_var_results.png – VaR vs loss + comparison overlay."""
    losses  = -pnl.values
    excepts = losses > VaR_series

    def _load_var(fname, col):
        try:
            df = pd.read_csv(DATA / fname, index_col=0, parse_dates=True)
            return df[col].reindex(dates)
        except Exception:
            return None

    var_evt   = _load_var("var_evt.csv",      "VaR_EVT")
    var_garch = _load_var("var_garch_evt.csv", "VaR_GARCH_EVT")

    n_panels = 2 if (var_evt is not None or var_garch is not None) else 1
    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 4 * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]

    ax = axes[0]
    ax.fill_between(dates, -pnl.values, alpha=0.35, color="steelblue",
                    label="Actual loss (−P&L)")
    ax.plot(dates, VaR_series, "r-", lw=1.2, label="Copula VaR (99%)")
    ax.scatter(dates[excepts], losses[excepts], color="crimson", s=20, zorder=5,
               label=f"Exceptions ({excepts.sum()})")
    ax.set_ylabel("USD"); ax.legend(fontsize=8)
    ax.set_title("Copula VaR vs realised loss")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    if n_panels == 2:
        ax2 = axes[1]
        ax2.plot(dates, VaR_series, "r-", lw=1.2, label="Copula")
        if var_garch is not None:
            ax2.plot(dates, var_garch.values, "g-", lw=1.0, alpha=0.8,
                     label="GARCH+EVT")
        if var_evt is not None:
            ax2.plot(dates, var_evt.values, "b--", lw=1.0, alpha=0.7, label="EVT")
        ax2.set_ylabel("USD"); ax2.legend(fontsize=8)
        ax2.set_title("VaR model comparison")
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.autofmt_xdate()
    _save(FIGS / "var_copula_03_var_results.png")


# ─── WALK-FORWARD ROLLING LOOP ────────────────────────────────────────────────

def compute_copula_var(returns_arr, dates, rng_gof, rng_plot, rng_ci,
                       nonlinear_series=None):
    """
    Walk-forward rolling VaR loop.

    The marginal + copula fit is updated every COPULA_REFIT days.  Monte Carlo
    VaR is redrawn each day with a deterministic per-date seed for
    reproducibility.  Separate RNG streams are used for GoF bootstrap
    (rng_gof), diagnostic plots (rng_plot), and bootstrap CI (rng_ci) so
    those routines do not consume from the daily VaR stream.

    pandas date-indexed Series and DataFrames used throughout are analogous to
    zoo/xts objects in R.

    Returns (VaR_arr, rho_mean_arr, gof_records, marginal_records,
             bsci_records, tail_dep_records)
    """
    T          = len(returns_arr)
    VaR        = np.full(T, np.nan)
    rho_mean   = np.full(T, np.nan)
    fit_cache  = None
    corr_cache = np.nan

    gof_records          = []
    marginal_records     = []
    bootstrap_ci_records = []
    tail_dep_records     = []

    first_plot = True
    mid_t      = WINDOW + (T - WINDOW) // 2

    n_refits = 0
    print(f"\nComputing Copula VaR for {T - WINDOW} days ...\n")
    print("MC VaR redrawn each day with deterministic per-date seed; "
          "fit cached between refits.")

    for t in range(WINDOW, T):
        need_refit = (fit_cache is None) or ((t - WINDOW) % COPULA_REFIT == 0)

        if need_refit:
            data_w      = returns_arr[t - WINDOW : t]
            refit_label = str(dates[t].date())
            n_refits   += 1
            print(f"  Refit {n_refits:3d}: t={t}/{T}  ({refit_label})")

            save_diag = first_plot or (abs(t - mid_t) < COPULA_REFIT // 2)

            fit_cache = fit_marginals_and_copula(
                data_w, NU_T_COPULA, refit_label,
                run_gof=True, n_gof_bootstrap=N_GOF_BOOTSTRAP, rng_gof=rng_gof,
            )

            marginal_records.extend(fit_cache["marginal_records"])

            gof = fit_cache["gof"]
            if gof is not None:
                gof_records.append({
                    "refit_id":    refit_label,
                    "cvm_obs":     gof["cvm_obs"],
                    "q025":        gof["q025"],
                    "q95":         gof["q95"],
                    "q975":        gof["q975"],
                    "reject_5pct": int(gof["reject_5pct"]),
                    "p_value":     gof["p_value"],
                })

            tail_dep_records.extend(
                copula_tail_dependence_table(fit_cache["R"], fit_cache["U_emp"],
                                             NU_T_COPULA, refit_label)
            )

            if save_diag:
                U_sim_diag = sample_t_copula(fit_cache["R"], NU_T_COPULA, 500, rng_plot)
                plot_marginals(data_w, fit_cache["marginals"],
                               fit_cache["U_emp"], U_sim_diag, refit_label)
                if gof is not None:
                    plot_cvm_null(gof, refit_label)
                first_plot = False

            R_c        = fit_cache["R"]
            d_c        = R_c.shape[0]
            corr_mask  = ~np.eye(d_c, dtype=bool)
            corr_cache = float(R_c[corr_mask].mean())

            try:
                rng_ci_local = np.random.default_rng(RANDOM_SEED + 20000 + t)
                _, PnL_ref   = monte_carlo_var(
                    fit_cache, N_SIMS, rng_ci_local,
                    nonlinear_window=(
                        nonlinear_series.iloc[t - WINDOW : t].values
                        if nonlinear_series is not None else None
                    ),
                )
                ci_lo, ci_hi = var_bootstrap_ci(PnL_ref, N_VAR_BOOTSTRAP, rng_ci)
            except Exception as exc:
                warnings.warn(
                    f"var_bootstrap_ci failed at refit t={t} ({refit_label}): "
                    f"{type(exc).__name__}: {exc}"
                )
                ci_lo, ci_hi = np.nan, np.nan
            bootstrap_ci_records.append({
                "refit_id": refit_label,
                "ci_lo":    ci_lo,
                "ci_hi":    ci_hi,
            })

        nonlinear_w = (
            nonlinear_series.iloc[t - WINDOW : t].values
            if nonlinear_series is not None else None
        )
        # Per-date seed: daily VaR is reproducible and independent of the
        # GoF, plot, and CI RNG streams.
        rng_var_t = np.random.default_rng(RANDOM_SEED + 10000 + t)
        try:
            VaR_t, _ = monte_carlo_var(fit_cache, N_SIMS, rng_var_t,
                                       nonlinear_window=nonlinear_w)
        except Exception as exc:
            warnings.warn(
                f"monte_carlo_var failed at t={t} ({dates[t].date()}): "
                f"{type(exc).__name__}: {exc}"
            )
            VaR_t = np.nan

        VaR[t]      = VaR_t
        rho_mean[t] = corr_cache

    print(f"\nTotal refits: {n_refits}")
    return (
        VaR[WINDOW:],
        rho_mean[WINDOW:],
        gof_records,
        marginal_records,
        bootstrap_ci_records,
        tail_dep_records,
    )


# ─── BACKTEST ─────────────────────────────────────────────────────────────────

def backtest_copula(pnl_backtest, VaR_arr, dates_bt):
    var_series = pd.Series(VaR_arr, index=dates_bt)
    return run_backtest(pnl_backtest, var_series, confidence=ALPHA,
                        method_name="Copula")


# ─── SAVE OUTPUTS ─────────────────────────────────────────────────────────────

def save_outputs(dates_bt, pnl_bt, VaR_series, rho_mean, backtest_result,
                 gof_records, marginal_records, bootstrap_ci_records,
                 tail_dep_records):
    # Build per-date cvm_pvalue Series aligned to each refit's active window.
    # Each date carries the p-value of the fit that produced that date's VaR.
    cvm_series = pd.Series(np.nan, index=dates_bt)
    if gof_records:
        sorted_gof = sorted(gof_records, key=lambda r: pd.Timestamp(r["refit_id"]))
        for i, rec in enumerate(sorted_gof):
            start  = pd.Timestamp(rec["refit_id"])
            end    = (pd.Timestamp(sorted_gof[i + 1]["refit_id"])
                      if i + 1 < len(sorted_gof)
                      else dates_bt[-1] + pd.Timedelta(days=1))
            active = (dates_bt >= start) & (dates_bt < end)
            cvm_series.loc[active] = rec["p_value"]

    df_var = pd.DataFrame({
        "VaR_COPULA":       VaR_series,
        "mean_correlation": rho_mean,
        "cvm_pvalue":       cvm_series.values,
    }, index=dates_bt)
    df_var.to_csv(DATA / "var_copula.csv")
    print(f"VaR saved     -> {DATA / 'var_copula.csv'}")

    exceptions = (-pnl_bt.values > VaR_series).astype(int)
    df_bt = pd.DataFrame({
        "VaR":         VaR_series,
        "actual_loss": -pnl_bt.values,
        "exception":   exceptions,
        "cvm_pvalue":  cvm_series.values,
    }, index=dates_bt)
    df_bt.to_csv(TABS / "backtest_copula.csv")
    print(f"Backtest      -> {TABS / 'backtest_copula.csv'}")

    if marginal_records:
        pd.DataFrame(marginal_records).to_csv(TABS / "marginal_selection.csv",
                                              index=False)
        print(f"Marginals     -> {TABS / 'marginal_selection.csv'}")

    if gof_records:
        pd.DataFrame(gof_records).to_csv(TABS / "copula_gof.csv", index=False)
        print(f"GoF diagnostic -> {TABS / 'copula_gof.csv'}")

    if bootstrap_ci_records:
        pd.DataFrame(bootstrap_ci_records).to_csv(TABS / "var_bootstrap_ci.csv",
                                                   index=False)
        print(f"Bootstrap CI  -> {TABS / 'var_bootstrap_ci.csv'}")

    if tail_dep_records:
        pd.DataFrame(tail_dep_records).to_csv(TABS / "copula_tail_dependence.csv",
                                              index=False)
        print(f"Tail dependence -> {TABS / 'copula_tail_dependence.csv'}")


# ─── RESULTS SUMMARY ──────────────────────────────────────────────────────────

def print_results_summary(backtest_result, gof_records, marginal_records,
                           VaR_series):
    print("\n" + "═" * 60)
    print("RESULTS SUMMARY")
    print("═" * 60)

    if gof_records:
        n_reject = sum(r["reject_5pct"] for r in gof_records)
        n_total  = len(gof_records)
        rej_rate = n_reject / n_total
        print(f"1. CvM rejection rate (5%)  : {n_reject}/{n_total} = {rej_rate:.1%}")
    else:
        rej_rate = 0.0
        print("1. CvM rejection rate (5%)  : no records")

    if marginal_records:
        df_m = pd.DataFrame(marginal_records)
        df_m["aic_adv"] = df_m["aic_normal"] - df_m["aic"]
        mean_adv = df_m["aic_adv"].mean()
        print(f"2. Mean AIC advantage (best vs Normal): {mean_adv:.1f}")
        print(f"   Most common marginal: "
              f"{df_m.groupby('dist_name').size().idxmax()}")

    print("3. Mean copula rho_mean : (see outputs; mean off-diagonal R)")

    exc = backtest_result.N
    kp  = backtest_result.pvalue_uc
    cp  = backtest_result.pvalue_cc
    print(f"4. Backtest – exceptions={exc}  Kupiec p={kp:.4f}  "
          f"Christoffersen p={cp:.4f}")

    print("\n5. Model comparison")
    print(f"{'Model':<18} | {'Exceptions':>10} | {'Kupiec p':>9} | "
          f"{'Christo p':>9} | {'Mean VaR':>10}")
    print("-" * 65)

    def _fmt_row(label, n_exc, kp_, cp_, mean_var):
        kp_s  = f"{kp_:>9.4f}" if kp_ is not None else f"{'see run':>9}"
        cp_s  = f"{cp_:>9.4f}" if cp_ is not None else f"{'see run':>9}"
        mv_s  = f"{mean_var:>10,.0f}" if mean_var is not None else f"{'n/a':>10}"
        ex_s  = f"{n_exc:>10}"  if n_exc is not None else f"{'n/a':>10}"
        print(f"{label:<18} | {ex_s} | {kp_s} | {cp_s} | {mv_s}")

    try:
        bt_evt  = pd.read_csv(TABS / "backtest_evt.csv")
        exc_evt = int(bt_evt["exception"].sum())
        var_evt = pd.read_csv(DATA / "var_evt.csv", index_col=0)["VaR_EVT"].mean()
        _fmt_row("EVT", exc_evt, None, None, var_evt)
    except Exception:
        _fmt_row("EVT", None, None, None, None)

    try:
        bt_ge  = pd.read_csv(TABS / "backtest_garch_evt.csv")
        exc_ge = int(bt_ge["exception"].sum())
        var_ge = pd.read_csv(DATA / "var_garch_evt.csv",
                             index_col=0)["VaR_GARCH_EVT"].mean()
        _fmt_row("GARCH+EVT", exc_ge, None, None, var_ge)
    except Exception:
        _fmt_row("GARCH+EVT", None, None, None, None)

    _fmt_row("Copula", exc, kp, cp, np.nanmean(VaR_series))
    print("-" * 65)

    T_bt    = len(VaR_series)
    exp_exc = T_bt * (1 - ALPHA)
    if exc > exp_exc * 1.5:
        print(
            f"\nNote: exception count ({exc}) exceeds 1.5× the expected value "
            f"({exp_exc:.1f}).  The raw-return t-copula may be under-conservative "
            f"for this data."
        )

    print(
        "\nPreferred production model: garch_evt.py, which applies a GARCH(1,1) "
        "filter before EVT so that the i.i.d. assumption for POT is more plausible. "
        "var_copula.py is retained as the multivariate dependence benchmark "
        "implementing the copula approach from lecture Section 7."
    )
    print("═" * 60)


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print(f"Copula VaR  |  alpha={ALPHA * 100:.0f}%  |  window={WINDOW} days")
    print(f"t-copula ν={NU_T_COPULA} (fixed) | MC paths={N_SIMS:,} | "
          f"refit every {COPULA_REFIT} days")
    print("=" * 60)

    rng_gof  = np.random.default_rng(RANDOM_SEED)
    rng_plot = np.random.default_rng(RANDOM_SEED + 1)
    rng_ci   = np.random.default_rng(RANDOM_SEED + 2)

    returns_df, pnl_series, nonlinear = load_data()

    if len(returns_df) <= WINDOW:
        raise ValueError(
            f"Insufficient data: len(returns_df)={len(returns_df)} <= WINDOW={WINDOW}. "
            f"Need at least WINDOW+1 observations to produce any VaR forecast."
        )
    if len(returns_df) < WINDOW + 50:
        warnings.warn(
            "Backtest period is short; statistical power of Kupiec/Christoffersen "
            "tests is low."
        )

    print("\nReturn i.i.d. diagnostics:")
    compute_return_diagnostics(returns_df)

    dates       = returns_df.index
    returns_arr = returns_df.values
    T           = len(dates)

    (VaR_arr, rho_arr, gof_records, marg_records,
     bsci_records, tail_dep_records) = compute_copula_var(
        returns_arr, dates, rng_gof, rng_plot, rng_ci,
        nonlinear_series=nonlinear,
    )

    n_nan = int(np.sum(np.isnan(VaR_arr)))
    if n_nan > 0:
        raise ValueError(f"compute_copula_var produced {n_nan} NaN VaR value(s).")
    n_neg = int(np.sum(VaR_arr < 0))
    if n_neg > 0:
        raise ValueError(f"compute_copula_var produced {n_neg} negative VaR value(s).")

    dates_bt = dates[WINDOW:]
    pnl_bt   = pnl_series.iloc[WINDOW:]

    print("\n" + "─" * 56)
    print("  Backtest Results  ·  Copula")
    print("─" * 56)
    bt = backtest_copula(pnl_bt, VaR_arr, dates_bt)
    print(bt)

    print("\nGenerating plots ...")
    plot_var_results(dates_bt, pnl_bt, VaR_arr, bt)

    save_outputs(dates_bt, pnl_bt, VaR_arr, rho_arr, bt,
                 gof_records, marg_records, bsci_records, tail_dep_records)

    print("\nRolling Copula summary:")
    print(f"  Mean VaR : ${np.nanmean(VaR_arr):>12,.0f}")
    print(f"  Min  VaR : ${np.nanmin(VaR_arr):>12,.0f}")
    print(f"  Max  VaR : ${np.nanmax(VaR_arr):>12,.0f}")
    print(f"  NaN days : {np.isnan(VaR_arr).sum()}")

    if nonlinear is not None:
        print("  Nonlinear P&L: included via independent historical bootstrap.")
    else:
        print("  Nonlinear P&L: not available; linear-only VaR.")

    print_results_summary(bt, gof_records, marg_records, VaR_arr)

    if gof_records:
        p_min = min(r["p_value"] for r in gof_records)
        p_max = max(r["p_value"] for r in gof_records)
        print(f"\nGoF p-value range across refits: [{p_min:.3f}, {p_max:.3f}]")

    print("\n" + "=" * 60)
    print("Copula VaR complete.")
    print(f"  Exceptions : {bt.N}  "
          f"(expected {(T - WINDOW) * (1 - ALPHA):.1f} at 99% VaR)")
    print(f"  Kupiec H0  : "
          f"{'not rejected' if bt.pvalue_uc >= 0.05 else 'rejected'}"
          f"  (p={bt.pvalue_uc:.4f})")
    print(f"  Copula fit : rank-based pseudo-likelihood (canonical MLE), fixed ν={NU_T_COPULA}; "
          f"plug-in sample correlation used where optimisation did not improve.")
    print("=" * 60)


if __name__ == "__main__":
    main()
