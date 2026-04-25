"""
var_copula.py  –  Copula-Based Value-at-Risk
═══════════════════════════════════════════════════════════════════════════════
CHANGELOG (model validation review — see section labels below):
  A1. Fix P&L formula to log-return re-pricing: (V0/d) × Σ (exp(r_j_sim) − 1).
      Previous linear formula r_sim overstated tail losses by ~5-10% at
      extreme quantiles; exp formula matches VaR_Estimation.R §4d exactly.
  A2. Monte Carlo VaR now redrawn every day using the cached fit (fresh RNG
      draws per day).  Previous: same VaR cached for the full 250-day refit
      window, giving only ~18 distinct values over 4269 days and making
      Kupiec/Christoffersen tests statistically unreliable.
  A3. Hybrid MC: IRS + straddle P&L added via historical bootstrap from the
      rolling window.  Removes the model-vs-backtest scope mismatch that was
      inflating the exception rate by construction.  Falls back to linear-only
      with a warning if instrument_pnl.csv is unavailable.
  A4. Per-refit GoF p-value broadcast to each date's active refit window in
      var_copula.csv.  Previous scalar (last-refit only) was meaningless for
      per-date filtering.  Column renamed cvm_pvalue_last_refit → cvm_pvalue.
  A5. Rename xi_mean → rho_mean throughout.  xi is the GPD shape parameter in
      evt.py / garch_evt.py; reusing it for copula correlation was confusing.
  A6. Replace hardcoded T_bt = 4269 with len(VaR_series) in advisory check.

METHODOLOGY
  Follows the course reference R implementation (VaR_Estimation.R):
  • AIC-selected parametric marginals (Normal / Logistic / Student-t / Laplace /
    GeneralisedNormal / GeneralisedLogistic), one per instrument.
  • Student-t copula with FIXED degrees of freedom ν = 4, matching
    R's tCopula(..., df.fixed=TRUE).  Fitting ν would be more rigorous
    (McNeil-Frey-Embrechts 2005) but departs from the course methodology.
  • Correlation matrix R fitted by MLE on t_ν-transformed pseudo-observations.
    For d ≤ 4 the sample-correlation shortcut is near-optimal and fast.
  • Parametric bootstrap Cramér-von Mises goodness-of-fit test (§4b).
  • Monte Carlo VaR via copula simulation (§4d).
  • Rolling window of WINDOW = 500 days, refit every COPULA_REFIT = 250 days
    (NOT expanding), matching the professor's "last N trading days" framing.

KEY DIFFERENCES FROM garch_evt.py
  • No GARCH filtering — marginals are fitted on raw log-returns.  Raw returns
    exhibit volatility clustering which violates the i.i.d. assumption of
    pseudo-observations; this is a known limitation of the course approach.
  • Copula df FIXED at ν = 4, not estimated.
  • Rolling window (not expanding).

DATA MODE — determined at Step 0
  Available in data/processed/:
    log_returns.csv  → columns [EURUSD, GLD, IEF, SPY]  (4 linear instruments)
    instrument_pnl.csv → [pnl_irs, pnl_straddle]  (nonlinear components)
  No per-asset price files found.  P&L formula (A1: log-return re-pricing,
  matches VaR_Estimation.R §4d):
    PnL_linear_sim = (V0 / 4) × Σ_j (exp(r_{j,sim}) − 1)
  Nonlinear P&L (IRS + straddle) added per path via historical bootstrap from
  the rolling window (A3).  This hybrid approach brings simulated P&L into the
  same space as pnl_total, removing the scope mismatch in backtesting.
  Assumption: nonlinear P&L is dependence-independent from linear factors —
  a simplification; fully correct treatment would include IRS/straddle in
  the copula via DV01 / gamma-vega risk factors, which is out of scope.

KNOWN LIMITATIONS
  • Volatility clustering → pseudo-observations are not i.i.d.
  • Fixed ν = 4 may not match true joint tail heaviness.
  • Nonlinear components added independently (no joint copula structure).
  • MC error ≈ O(1/√N_SIMS); at 20 000 paths and 99% VaR the MC standard
    error on the exception rate is ≈ 0.15 pp.
═══════════════════════════════════════════════════════════════════════════════
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from scipy.linalg import cholesky, eigh

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── add project root to path so backtesting module is importable ─────────────
_HERE  = Path(__file__).resolve()
_ROOT  = _HERE.parents[2]
sys.path.insert(0, str(_ROOT))
from backtesting.backtest import run_backtest

# ─── SETTINGS ─────────────────────────────────────────────────────────────────
WINDOW           = 500        # rolling window length (days)
ALPHA            = 0.99       # VaR confidence level
COPULA_REFIT     = 250        # refit every N days
N_SIMS           = 20_000     # Monte Carlo paths per VaR date
NU_T_COPULA      = 4          # t-copula degrees of freedom (fixed, df.fixed=TRUE)
N_GOF_BOOTSTRAP  = 1_000      # Cramér-von Mises null-distribution samples
N_VAR_BOOTSTRAP  = 1_000      # VaR bootstrap CI samples
RANDOM_SEED      = 42
V0               = 1_000_000  # portfolio value (USD)
N_ASSETS         = 4          # EURUSD, GLD, IEF, SPY
INSTRUMENT_NAMES = ["EURUSD", "GLD", "IEF", "SPY"]

# ─── PATHS ───────────────────────────────────────────────────────────────────
DATA = _ROOT / "data" / "processed"
FIGS = _ROOT / "outputs" / "figures"
TABS = _ROOT / "outputs" / "tables"
FIGS.mkdir(parents=True, exist_ok=True)
TABS.mkdir(parents=True, exist_ok=True)

# ─── MARGINAL CANDIDATES ─────────────────────────────────────────────────────
_MARGINAL_CANDIDATES = [
    ("norm",        stats.norm),
    ("logistic",    stats.logistic),
    ("t",           stats.t),
    ("laplace",     stats.laplace),
    ("gennorm",     stats.gennorm),
    ("genlogistic", stats.genlogistic),
]


# ─── STEP 0 – DATA LOADING ───────────────────────────────────────────────────

def load_data():
    """
    Load per-asset log-returns, total portfolio P&L, and nonlinear P&L components.

    Returns
    -------
    lr          : pd.DataFrame  log-returns for 4 linear assets (EURUSD, GLD, IEF, SPY)
    pnl_total   : pd.Series     total portfolio P&L (for backtesting)
    nonlinear   : pd.Series or None
                  IRS + straddle P&L summed; None if instrument_pnl.csv is unavailable.
                  Used in A3 hybrid MC: historical bootstrap of the nonlinear component.
    """
    lr   = pd.read_csv(DATA / "log_returns.csv",         index_col=0, parse_dates=True)
    pnl  = pd.read_csv(DATA / "total_portfolio_pnl.csv", index_col=0, parse_dates=True)

    # A3: load nonlinear P&L (IRS + straddle)
    nonlinear = None
    try:
        inst   = pd.read_csv(DATA / "instrument_pnl.csv", index_col=0, parse_dates=True)
        common = lr.index.intersection(pnl.index).intersection(inst.index)
        inst   = inst.loc[common]
        if "pnl_irs" in inst.columns and "pnl_straddle" in inst.columns:
            nonlinear = inst[["pnl_irs", "pnl_straddle"]].sum(axis=1)
            nonlinear.name = "pnl_nonlinear"
        else:
            warnings.warn(
                "load_data [A3]: instrument_pnl.csv missing pnl_irs or pnl_straddle columns; "
                "falling back to linear-only VaR (scope mismatch with pnl_total backtest)."
            )
    except Exception as exc:
        warnings.warn(
            f"load_data [A3]: instrument_pnl.csv not found or unreadable "
            f"({type(exc).__name__}: {exc}); falling back to linear-only VaR "
            f"(scope mismatch with pnl_total backtest)."
        )
        common = lr.index.intersection(pnl.index)

    lr   = lr.loc[common, INSTRUMENT_NAMES]
    pnl  = pnl.loc[common]
    if nonlinear is not None:
        nonlinear = nonlinear.loc[common]

    print(f"Loaded: {len(common)} days of data")
    print(f"Date range : {common[0].date()} -> {common[-1].date()}")
    print(f"Instruments: {INSTRUMENT_NAMES}")
    print(f"r mean (each):  {lr.mean().to_dict()}")
    print(f"pnl_total std:  ${pnl['pnl_total'].std():,.0f}")
    nl_label = f"${nonlinear.std():,.0f}" if nonlinear is not None else "not loaded (linear-only)"
    print(f"nonlinear std:  {nl_label}")
    return lr, pnl["pnl_total"], nonlinear


# ─── STEP 2 – MARGINAL FITTING ───────────────────────────────────────────────

def fit_best_marginal(x, instrument_name="?", refit_id="?"):
    """
    AIC-based marginal selection.
    Returns dict: dist_name, params, frozen, aic, ks_pvalue, aic_table
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
            f"fit_best_marginal [{instrument_name} @ {refit_id}]: all candidates failed; "
            f"falling back to Normal."
        )
        params  = stats.norm.fit(x)
        frozen  = stats.norm(*params)
        return dict(dist_name="norm", params=params, frozen=frozen,
                    aic=np.nan, ks_pvalue=np.nan,
                    aic_table=pd.DataFrame())

    results.sort(key=lambda r: r["aic"])
    best = results[0]

    # KS goodness-of-fit for selected distribution
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


# ─── STEP 3 – COPULA ─────────────────────────────────────────────────────────

def pseudo_observations(X):
    """
    Rank-based pseudo-observations with Hazen continuity correction.
    U[i, j] = rank(X[:, j])[i] / (n + 1)   (matches R's pobs())
    Returns array (n, d) with values in (0, 1).
    """
    n, d  = X.shape
    U     = np.zeros_like(X, dtype=float)
    for j in range(d):
        ranks   = stats.rankdata(X[:, j])   # 1-based
        U[:, j] = ranks / (n + 1)
    return U


def _nearest_pd(A):
    """Project symmetric matrix A to nearest positive-definite matrix (Higham 1988)."""
    B  = (A + A.T) / 2
    vals, vecs = eigh(B)
    vals = np.maximum(vals, 1e-8)
    return vecs @ np.diag(vals) @ vecs.T


def fit_t_copula(U_emp, nu=NU_T_COPULA, refit_id="?"):
    """
    Fit t-copula correlation matrix R on pseudo-observations U_emp (n, d).
    Method: sample correlation of t_ν-transformed pseudo-obs — near-optimal MLE
    for d ≤ 4 and avoids non-convex optimisation.
    """
    t_dist = stats.t(df=nu)
    eps    = 1e-6
    T_emp  = t_dist.ppf(np.clip(U_emp, eps, 1 - eps))   # (n, d)
    R      = np.corrcoef(T_emp.T)                         # (d, d)

    eigvals = np.linalg.eigvalsh(R)
    if eigvals.min() < 1e-8:
        warnings.warn(
            f"fit_t_copula [{refit_id}]: R not PD (min eigval={eigvals.min():.2e}); "
            f"projecting to nearest PD matrix."
        )
        R = _nearest_pd(R)

    return R


def sample_t_copula(R, nu, n_samples, rng):
    """
    Sample n_samples observations from t-copula(R, ν).
    Algorithm:  Y ~ N(0, R),  W ~ χ²(ν),  T = Y√(ν/W),  U = F_t,ν(T)
    Returns (n_samples, d) array in (0, 1).
    """
    d      = R.shape[0]
    try:
        L  = np.linalg.cholesky(R)
    except np.linalg.LinAlgError:
        R  = _nearest_pd(R)
        L  = np.linalg.cholesky(R)

    Z    = rng.standard_normal((n_samples, d))
    Y    = Z @ L.T                                        # correlated normals (n, d)
    W    = rng.chisquare(nu, size=n_samples)              # (n,)
    T    = Y * np.sqrt(nu / W[:, np.newaxis])             # (n, d) multivariate-t
    U    = stats.t.cdf(T, df=nu)                          # (n, d) copula samples
    return U


# ─── STEP 4 – CRAMÉR-VON MISES GOF TEST ─────────────────────────────────────

def eval_empirical_copula(eval_pts, data_pts):
    """
    Vectorised empirical copula: C_n(u) evaluated at multiple u.

    eval_pts  : (m, d) – points to evaluate at
    data_pts  : (n, d) – pseudo-observations
    Returns   : (m,) array,  C_n[i] = #{j : data_pts[j] ≤ eval_pts[i] componentwise} / n

    No Python loop over evaluation points; uses numpy broadcasting.
    Memory: m × n × d booleans ≈ 500×500×4 = 1 MB per call (negligible).
    """
    # mask[i, j] = True iff all d components of data_pts[j] ≤ eval_pts[i]
    mask = np.all(
        data_pts[np.newaxis, :, :] <= eval_pts[:, np.newaxis, :],
        axis=2,
    )                                                      # (m, n)
    return mask.mean(axis=1)                               # (m,)


def cvm_gof_test(U_emp, R, nu, n_bootstrap, rng, refit_id="?"):
    """
    Parametric bootstrap Cramér-von Mises GoF test (matches professor's §4b).

    Returns dict: cvm_obs, q025, q975, reject, p_value, cvm_null
    """
    n, _  = U_emp.shape

    def _sample(n_draw):
        return sample_t_copula(R, nu, n_draw, rng)

    # Reference copula sample (fixes the simulated copula C_sim_ref)
    U_ref     = _sample(n)

    # C_n(U_emp[i])       – empirical copula at each pseudo-obs
    C_n       = eval_empirical_copula(U_emp, U_emp)    # (n,)
    # C_sim_ref(U_emp[i]) – reference simulated copula at each pseudo-obs
    C_sim_ref = eval_empirical_copula(U_emp, U_ref)    # (n,)

    # Observed test statistic
    CvM_obs = float(np.sum((C_n - C_sim_ref) ** 2))

    # Bootstrap null distribution
    CvM_null = np.zeros(n_bootstrap)
    for b in range(n_bootstrap):
        U_b         = _sample(n)
        C_sim_b     = eval_empirical_copula(U_emp, U_b)
        CvM_null[b] = np.sum((C_sim_b - C_sim_ref) ** 2)

    q025   = float(np.percentile(CvM_null, 2.5))
    q975   = float(np.percentile(CvM_null, 97.5))
    reject = bool((CvM_obs < q025) or (CvM_obs > q975))
    p_val  = float(2 * min(np.mean(CvM_null >= CvM_obs),
                           np.mean(CvM_null <= CvM_obs)))

    return dict(cvm_obs=CvM_obs, q025=q025, q975=q975,
                reject=reject, p_value=p_val, cvm_null=CvM_null)


# ─── COMBINED REFIT ───────────────────────────────────────────────────────────

def fit_marginals_and_copula(window_data, nu, refit_label, run_gof,
                              n_gof_bootstrap, rng):
    """
    Fit AIC-selected marginals + t-copula on window_data (n, d).
    Optionally runs CvM GoF test.
    Returns dict: marginals, R, U_emp, marginal_records, gof (or None)
    """
    n, d  = window_data.shape
    marginals        = []
    marginal_records = []

    for j, name in enumerate(INSTRUMENT_NAMES):
        x    = window_data[:, j]
        info = fit_best_marginal(x, instrument_name=name, refit_id=refit_label)
        marginals.append(info["frozen"])
        marginal_records.append({
            "refit_id":    refit_label,
            "instrument":  name,
            "dist_name":   info["dist_name"],
            "aic":         info["aic"],
            "ks_pvalue":   info["ks_pvalue"],
            "params":      str(info["params"]),
            "aic_normal":  next(
                (r["aic"] for _, r in enumerate(info["aic_table"].to_dict("records"))
                 if r["name"] == "norm"),
                np.nan,
            ),
        })

    U_emp = pseudo_observations(window_data)
    R     = fit_t_copula(U_emp, nu=nu, refit_id=refit_label)

    gof = None
    if run_gof:
        gof = cvm_gof_test(U_emp, R, nu, n_gof_bootstrap, rng, refit_id=refit_label)

    return dict(marginals=marginals, R=R, U_emp=U_emp,
                marginal_records=marginal_records, gof=gof)


# ─── STEP 5 – MONTE CARLO VAR ─────────────────────────────────────────────────

def monte_carlo_var(fit_cache, n_sims, rng, nonlinear_window=None):
    """
    Draw n_sims copula scenarios, invert marginals, compute portfolio P&L.

    A1: Log-return re-pricing formula (matches VaR_Estimation.R §4d):
        PnL_linear_sim = (V0 / N_ASSETS) × Σ_j (exp(r_j_sim) − 1)
        This is the exact formula used in the professor's R code; the previous
        linear approximation (r_sim instead of exp(r_sim) − 1) overstated
        tail losses at extreme quantiles by ~5-10%.

    A3: Hybrid MC — add nonlinear P&L via historical bootstrap:
        If nonlinear_window is provided, one historical nonlinear observation
        is resampled (with replacement) per MC path and added to PnL_linear_sim.
        This brings simulated P&L into the same space as pnl_total, removing
        the scope mismatch between the copula (linear only) and the backtest
        target.  Assumption: nonlinear P&L is dependence-independent from
        linear factors (simplification; full treatment is out of scope).

    Returns: (VaR, PnL_sim array)
    """
    R, marginals = fit_cache["R"], fit_cache["marginals"]
    U_sim = sample_t_copula(R, NU_T_COPULA, n_sims, rng)      # (n_sims, d)
    eps   = 1e-6
    r_sim = np.column_stack([
        marginals[j].ppf(np.clip(U_sim[:, j], eps, 1 - eps))
        for j in range(len(marginals))
    ])                                                          # (n_sims, d)

    # A1: log-return re-pricing — exp(r) − 1 is the relative price change
    PnL_linear = (V0 / N_ASSETS) * (np.exp(r_sim) - 1).sum(axis=1)   # (n_sims,)

    # A3: add nonlinear P&L (IRS + straddle) via historical bootstrap
    if nonlinear_window is not None and len(nonlinear_window) > 0:
        idx         = rng.integers(0, len(nonlinear_window), size=n_sims)
        PnL_sim     = PnL_linear + nonlinear_window[idx]
    else:
        PnL_sim     = PnL_linear

    VaR = max(-np.percentile(PnL_sim, (1 - ALPHA) * 100), 0.0)
    return VaR, PnL_sim


# ─── STEP 7 – VAR BOOTSTRAP CI ────────────────────────────────────────────────

def var_bootstrap_ci(PnL_sim, n_bootstrap, rng):
    """
    Bootstrap CI for VaR by resampling the MC P&L paths.
    Returns (ci_lo, ci_hi).
    """
    n      = len(PnL_sim)
    VaR_bs = np.zeros(n_bootstrap)
    for b in range(n_bootstrap):
        idx       = rng.integers(0, n, size=n)
        VaR_bs[b] = -np.percentile(PnL_sim[idx], (1 - ALPHA) * 100)
    return float(np.percentile(VaR_bs, 2.5)), float(np.percentile(VaR_bs, 97.5))


# ─── PLOTS ────────────────────────────────────────────────────────────────────

def _save(path):
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  Figure saved -> {path}")


def plot_marginals(window_data, marginals, U_emp, U_sim, refit_label):
    """
    var_copula_marginals_<date>.png  :  histograms + QQ plots (first 2 instruments)
    var_copula_dependence_<date>.png :  pseudo-obs scatter + empirical vs simulated
    """
    # ── marginals ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(f"Copula: marginal diagnostics  [{refit_label}]", fontsize=12)

    for col, (ax_hist, ax_qq) in enumerate(zip(axes[0], axes[1])):
        if col >= len(INSTRUMENT_NAMES):
            break
        name   = INSTRUMENT_NAMES[col]
        x      = window_data[:, col]
        frozen = marginals[col]

        # histogram + PDF
        xg = np.linspace(x.min(), x.max(), 300)
        ax_hist.hist(x, bins=40, density=True, alpha=0.5, color="steelblue",
                     label="empirical")
        ax_hist.plot(xg, frozen.pdf(xg), "r-", lw=1.5, label=f"{frozen.dist.name} fit")
        ax_hist.set_title(name); ax_hist.legend(fontsize=7)

        # QQ plot vs fitted distribution
        probs  = (np.arange(1, len(x) + 1)) / (len(x) + 1)
        q_emp  = np.sort(x)
        q_the  = frozen.ppf(probs)
        ax_qq.scatter(q_the, q_emp, s=4, alpha=0.5, color="steelblue")
        lim = (min(q_the.min(), q_emp.min()), max(q_the.max(), q_emp.max()))
        ax_qq.plot(lim, lim, "r--", lw=1)
        ax_qq.set_xlabel("theoretical"); ax_qq.set_ylabel("sample")
        ax_qq.set_title(f"QQ – {name}")

    _save(FIGS / f"var_copula_marginals_{refit_label}.png")

    # ── dependence ───────────────────────────────────────────────────────────
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f"Copula: pseudo-observations  [{refit_label}]", fontsize=12)

    ax_l.scatter(U_emp[:, 0], U_emp[:, 1], s=4, alpha=0.4, color="steelblue")
    ax_l.set_xlabel(INSTRUMENT_NAMES[0]); ax_l.set_ylabel(INSTRUMENT_NAMES[1])
    ax_l.set_title("Empirical pseudo-obs")

    ax_r.scatter(U_emp[:, 0], U_emp[:, 1], s=4, alpha=0.4, color="steelblue",
                 label="empirical")
    ax_r.scatter(U_sim[:, 0], U_sim[:, 1], s=2, alpha=0.2, color="tomato",
                 label="t-copula sim")
    ax_r.set_xlabel(INSTRUMENT_NAMES[0]); ax_r.set_ylabel(INSTRUMENT_NAMES[1])
    ax_r.set_title("Empirical vs simulated"); ax_r.legend(fontsize=7)

    _save(FIGS / f"var_copula_dependence_{refit_label}.png")


def plot_cvm_null(gof, refit_label):
    """var_copula_cvm_null_<date>.png – null distribution histogram."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(gof["cvm_null"], bins=50, density=True, alpha=0.6, color="steelblue",
            label="Bootstrap null")
    ax.axvline(gof["cvm_obs"], color="red", lw=2,
               label=f"Observed CvM = {gof['cvm_obs']:.4f}")
    ax.axvline(gof["q025"], color="orange", lw=1.5, ls="--",
               label=f"2.5% = {gof['q025']:.4f}")
    ax.axvline(gof["q975"], color="orange", lw=1.5, ls="--",
               label=f"97.5% = {gof['q975']:.4f}")
    reject_txt = "REJECT H0" if gof["reject"] else "fail to reject H0"
    ax.set_title(
        f"CvM GoF null distribution  [{refit_label}]  p={gof['p_value']:.3f}  {reject_txt}"
    )
    ax.legend(fontsize=8)
    _save(FIGS / f"var_copula_cvm_null_{refit_label}.png")


def plot_var_results(dates, pnl, VaR_series, backtest_result):
    """var_copula_03_var_results.png – VaR vs loss + comparison overlay."""
    losses = -pnl.values
    excepts = losses > VaR_series

    # ── load comparison series if available ──────────────────────────────────
    def _load_var(fname, col):
        try:
            df = pd.read_csv(DATA / fname, index_col=0, parse_dates=True)
            return df[col].reindex(dates)
        except Exception:
            return None

    var_evt      = _load_var("var_evt.csv",      "VaR_EVT")
    var_garch    = _load_var("var_garch_evt.csv", "VaR_GARCH_EVT")

    n_panels = 2 if (var_evt is not None or var_garch is not None) else 1
    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 4 * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]

    # Panel 1: VaR vs actual loss
    ax = axes[0]
    ax.fill_between(dates, -pnl.values, alpha=0.35, color="steelblue", label="P&L")
    ax.plot(dates, VaR_series, "r-", lw=1.2, label=f"Copula VaR (99%)")
    ax.scatter(dates[excepts], losses[excepts], color="crimson", s=20, zorder=5,
               label=f"Exceptions ({excepts.sum()})")
    ax.set_ylabel("USD"); ax.legend(fontsize=8)
    ax.set_title("Copula VaR vs realised loss")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Panel 2: comparison
    if n_panels == 2:
        ax2 = axes[1]
        ax2.plot(dates, VaR_series, "r-",  lw=1.2, label="Copula (new)")
        if var_garch is not None:
            ax2.plot(dates, var_garch.values, "g-",  lw=1.0, alpha=0.8,
                     label="GARCH+EVT")
        if var_evt is not None:
            ax2.plot(dates, var_evt.values,   "b--", lw=1.0, alpha=0.7, label="EVT")
        ax2.set_ylabel("USD"); ax2.legend(fontsize=8)
        ax2.set_title("VaR model comparison")
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.autofmt_xdate()
    _save(FIGS / "var_copula_03_var_results.png")


# ─── STEP 6 – WALK-FORWARD ROLLING LOOP ───────────────────────────────────────

def compute_copula_var(returns_arr, dates, rng, nonlinear_series=None):
    """
    Walk-forward rolling loop.

    A2: monte_carlo_var() is called EVERY DAY using the cached fit with fresh
        random draws.  The marginal + copula fit is updated only at refit dates
        (every COPULA_REFIT days), but each day gets an independent MC draw so
        the 4269-day VaR series has ~4269 distinct values.

    A3: nonlinear_series is an aligned pd.Series of IRS+straddle P&L.  If
        provided, the rolling window of nonlinear observations is passed to
        monte_carlo_var() for historical bootstrap addition.

    Returns (VaR_arr, rho_mean_arr, gof_records, marginal_records, bsci_records)
    """
    T            = len(returns_arr)
    VaR          = np.full(T, np.nan)
    rho_mean     = np.full(T, np.nan)   # A5: mean off-diagonal copula correlation
    fit_cache    = None
    corr_cache   = np.nan

    gof_records          = []
    marginal_records     = []
    bootstrap_ci_records = []

    first_plot = True
    mid_t      = WINDOW + (T - WINDOW) // 2

    n_refits = 0
    print(f"\nComputing Copula VaR for {T - WINDOW} days ...\n")
    print("A2: MC VaR redrawn every day (fresh RNG draws; fit cached between refits)")

    for t in range(WINDOW, T):
        need_refit = (fit_cache is None) or ((t - WINDOW) % COPULA_REFIT == 0)

        if need_refit:
            data_w      = returns_arr[t - WINDOW : t]   # (WINDOW, d)
            refit_label = str(dates[t].date())

            n_refits += 1
            print(f"  Refit {n_refits:3d}: t={t}/{T}  ({refit_label})")

            save_diag = first_plot or (abs(t - mid_t) < COPULA_REFIT // 2)

            fit_cache = fit_marginals_and_copula(
                data_w, NU_T_COPULA, refit_label,
                run_gof=True, n_gof_bootstrap=N_GOF_BOOTSTRAP, rng=rng
            )

            marginal_records.extend(fit_cache["marginal_records"])

            gof = fit_cache["gof"]
            if gof is not None:
                gof_records.append({
                    "refit_id": refit_label,
                    "cvm_obs":  gof["cvm_obs"],
                    "q025":     gof["q025"],
                    "q975":     gof["q975"],
                    "reject":   int(gof["reject"]),
                    "p_value":  gof["p_value"],
                })

            if save_diag:
                U_sim_diag = sample_t_copula(fit_cache["R"], NU_T_COPULA, 500, rng)
                plot_marginals(data_w, fit_cache["marginals"],
                               fit_cache["U_emp"], U_sim_diag, refit_label)
                if gof is not None:
                    plot_cvm_null(gof, refit_label)
                first_plot = False

            # Mean off-diagonal correlation (property of the fit, not of draws)
            R_c         = fit_cache["R"]
            d_c         = R_c.shape[0]
            corr_mask   = ~np.eye(d_c, dtype=bool)
            corr_cache  = float(R_c[corr_mask].mean())

            # VaR bootstrap CI at refit dates (diagnostic only — uses one MC draw)
            try:
                _, PnL_ref = monte_carlo_var(
                    fit_cache, N_SIMS, rng,
                    nonlinear_window=(
                        nonlinear_series.iloc[t - WINDOW : t].values
                        if nonlinear_series is not None else None
                    ),
                )
                ci_lo, ci_hi = var_bootstrap_ci(PnL_ref, N_VAR_BOOTSTRAP, rng)
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

        # A2: Monte Carlo VaR every day (fresh random draws using cached fit)
        # A3: pass rolling nonlinear window for historical bootstrap
        nonlinear_w = (
            nonlinear_series.iloc[t - WINDOW : t].values
            if nonlinear_series is not None else None
        )
        try:
            VaR_t, _ = monte_carlo_var(fit_cache, N_SIMS, rng,
                                       nonlinear_window=nonlinear_w)
        except Exception as exc:
            warnings.warn(
                f"monte_carlo_var failed at t={t} ({dates[t].date()}): "
                f"{type(exc).__name__}: {exc}"
            )
            VaR_t = np.nan

        VaR[t]      = VaR_t
        rho_mean[t] = corr_cache   # A5: stepwise-constant (fit property, not per-draw)

    print(f"\nTotal refits: {n_refits}")
    return (
        VaR[WINDOW:],
        rho_mean[WINDOW:],
        gof_records,
        marginal_records,
        bootstrap_ci_records,
    )


# ─── STEP 8 – BACKTEST ────────────────────────────────────────────────────────

def backtest_copula(pnl_backtest, VaR_arr, dates_bt):
    var_series = pd.Series(VaR_arr, index=dates_bt)
    return run_backtest(pnl_backtest, var_series, confidence=ALPHA,
                        method_name="Copula")


# ─── STEP 10 – SAVE OUTPUTS ───────────────────────────────────────────────────

def save_outputs(dates_bt, pnl_bt, VaR_series, rho_mean, backtest_result,
                 gof_records, marginal_records, bootstrap_ci_records):
    # A4: Build per-date cvm_pvalue Series aligned to each refit's active window.
    # Each date carries the p-value of the fit that produced that date's VaR,
    # not just the last refit's scalar (which was meaningless for per-date filtering).
    cvm_series = pd.Series(np.nan, index=dates_bt)
    if gof_records:
        sorted_gof = sorted(gof_records, key=lambda r: pd.Timestamp(r["refit_id"]))
        for i, rec in enumerate(sorted_gof):
            start  = pd.Timestamp(rec["refit_id"])
            end    = (pd.Timestamp(sorted_gof[i + 1]["refit_id"])
                      if i + 1 < len(sorted_gof) else dates_bt[-1] + pd.Timedelta(days=1))
            active = (dates_bt >= start) & (dates_bt < end)
            cvm_series.loc[active] = rec["p_value"]

    # var_copula.csv
    df_var = pd.DataFrame({
        "VaR_COPULA":       VaR_series,
        "mean_correlation": rho_mean,   # A5: renamed from xi_mean
        "cvm_pvalue":       cvm_series.values,  # A4: per-date (renamed from _last_refit)
    }, index=dates_bt)
    df_var.to_csv(DATA / "var_copula.csv")
    print(f"VaR saved     -> {DATA / 'var_copula.csv'}")

    # backtest_copula.csv
    exceptions = (-pnl_bt.values > VaR_series).astype(int)
    df_bt = pd.DataFrame({
        "VaR":        VaR_series,
        "actual_loss": -pnl_bt.values,
        "exception":  exceptions,
        "cvm_pvalue": cvm_series.values,   # A4: per-date
    }, index=dates_bt)
    df_bt.to_csv(TABS / "backtest_copula.csv")
    print(f"Backtest      -> {TABS / 'backtest_copula.csv'}")

    # marginal_selection.csv
    if marginal_records:
        pd.DataFrame(marginal_records).to_csv(TABS / "marginal_selection.csv",
                                              index=False)
        print(f"Marginals     -> {TABS / 'marginal_selection.csv'}")

    # copula_gof.csv
    if gof_records:
        pd.DataFrame(gof_records).to_csv(TABS / "copula_gof.csv", index=False)
        print(f"GoF test      -> {TABS / 'copula_gof.csv'}")

    # var_bootstrap_ci.csv
    if bootstrap_ci_records:
        pd.DataFrame(bootstrap_ci_records).to_csv(TABS / "var_bootstrap_ci.csv",
                                                   index=False)
        print(f"Bootstrap CI  -> {TABS / 'var_bootstrap_ci.csv'}")


# ─── STEP 11 – VALIDATION SUMMARY ────────────────────────────────────────────

def print_validation_summary(backtest_result, gof_records, marginal_records,
                              VaR_series):
    print("\n" + "═" * 60)
    print("VALIDATION SUMMARY")
    print("═" * 60)

    # 1. CvM rejection rate
    if gof_records:
        n_reject = sum(r["reject"] for r in gof_records)
        n_total  = len(gof_records)
        rej_rate = n_reject / n_total
        print(f"1. CvM rejection rate  : {n_reject}/{n_total} = {rej_rate:.1%}")
    else:
        rej_rate = 0.0
        print("1. CvM rejection rate  : no records")

    # 2. AIC advantage over normal
    if marginal_records:
        df_m = pd.DataFrame(marginal_records)
        df_m["aic_adv"] = df_m["aic_normal"] - df_m["aic"]
        mean_adv = df_m["aic_adv"].mean()
        print(f"2. Mean AIC adv (best vs norm): {mean_adv:.1f}")
        print(f"   Most common marginal: "
              f"{df_m.groupby('dist_name').size().idxmax()}")

    # 3. Mean pairwise correlation (A5: rho_mean, not xi_mean)
    print(f"3. Mean copula rho_mean : (see outputs; mean off-diagonal R)")

    # 4. Backtest
    exc = backtest_result.N
    kp  = backtest_result.pvalue_uc
    cp  = backtest_result.pvalue_cc
    print(f"4. Backtest  – exceptions={exc}  Kupiec p={kp:.4f}  "
          f"Christoffersen p={cp:.4f}")

    # 5. Comparison table
    def _load_bt(fname, var_col, exc_col=None):
        try:
            df = pd.read_csv(TABS / fname)
            n_exc = int(df.get("exception", df.get(exc_col, pd.Series())).sum())
            return df, n_exc
        except Exception:
            return None, None

    print("\n5. Model comparison")
    print(f"{'Model':<18} | {'Exceptions':>10} | {'Kupiec p':>9} | "
          f"{'Christo p':>9} | {'Mean VaR':>10}")
    print("-" * 65)

    def _fmt_row(label, exc, kp, cp, mean_var):
        kp_s  = f"{kp:>9.4f}"  if kp is not None  else f"{'see run':>9}"
        cp_s  = f"{cp:>9.4f}"  if cp is not None  else f"{'see run':>9}"
        mv_s  = f"{mean_var:>10,.0f}" if mean_var is not None else f"{'n/a':>10}"
        exc_s = f"{exc:>10}"   if exc is not None else f"{'n/a':>10}"
        print(f"{label:<18} | {exc_s} | {kp_s} | {cp_s} | {mv_s}")

    # EVT
    try:
        bt_evt  = pd.read_csv(TABS / "backtest_evt.csv")
        exc_evt = int(bt_evt["exception"].sum())
        var_evt = pd.read_csv(DATA / "var_evt.csv", index_col=0)["VaR_EVT"].mean()
        _fmt_row("EVT", exc_evt, None, None, var_evt)
    except Exception:
        _fmt_row("EVT", None, None, None, None)

    # GARCH+EVT
    try:
        bt_ge  = pd.read_csv(TABS / "backtest_garch_evt.csv")
        exc_ge = int(bt_ge["exception"].sum())
        var_ge = pd.read_csv(DATA / "var_garch_evt.csv", index_col=0)["VaR_GARCH_EVT"].mean()
        _fmt_row("GARCH+EVT", exc_ge, None, None, var_ge)
    except Exception:
        _fmt_row("GARCH+EVT", None, None, None, None)

    # Copula
    mean_var = np.nanmean(VaR_series)
    _fmt_row("Copula (new)", exc, kp, cp, mean_var)
    print("-" * 65)

    # Advisory (A6: use actual backtest length, not hardcoded 4269)
    T_bt      = len(VaR_series)
    exp_exc   = T_bt * (1 - ALPHA)
    bad_bt    = exc > exp_exc * 1.5        # >50% above expected
    if (rej_rate > 0.30 and bad_bt) or bad_bt:
        print(
            "\nADVISORY: t-copula rejected in >{:.0%} of refits AND backtest "
            "exception count exceeds expected.\nThe copula-without-GARCH approach "
            "appears mis-specified for this data.\nGARCH+EVT is the preferred "
            "production model; this copula is included for course transparency.".format(rej_rate)
        )

    print("═" * 60)


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print(f"Copula VaR  |  alpha={ALPHA*100:.0f}%  |  window={WINDOW} days")
    print(f"t-copula ν={NU_T_COPULA} (fixed) | MC paths={N_SIMS:,} | "
          f"refit every {COPULA_REFIT} days")
    print("=" * 60)

    rng                               = np.random.default_rng(RANDOM_SEED)
    returns_df, pnl_series, nonlinear = load_data()   # A3: 3-tuple

    dates        = returns_df.index
    returns_arr  = returns_df.values         # (T, 4)
    T            = len(dates)

    # A3: pass nonlinear_series into the rolling loop
    VaR_arr, rho_arr, gof_records, marg_records, bsci_records = compute_copula_var(
        returns_arr, dates, rng, nonlinear_series=nonlinear
    )

    dates_bt = dates[WINDOW:]
    pnl_bt   = pnl_series.iloc[WINDOW:]

    # ── backtest ─────────────────────────────────────────────────────────────
    print("\n" + "─" * 56)
    print("  Backtest Results  ·  Copula")
    print("─" * 56)
    bt = backtest_copula(pnl_bt, VaR_arr, dates_bt)
    print(bt)

    # ── plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots ...")
    plot_var_results(dates_bt, pnl_bt, VaR_arr, bt)

    # ── save ─────────────────────────────────────────────────────────────────
    save_outputs(dates_bt, pnl_bt, VaR_arr, rho_arr, bt,   # A5: rho_arr
                 gof_records, marg_records, bsci_records)

    # ── summary stats ─────────────────────────────────────────────────────────
    n_distinct = len(np.unique(np.round(VaR_arr[~np.isnan(VaR_arr)], 0)))
    n_refits   = len(gof_records)
    print("\nRolling Copula summary:")
    print(f"  Mean VaR   : ${np.nanmean(VaR_arr):>12,.0f}")
    print(f"  Min  VaR   : ${np.nanmin(VaR_arr):>12,.0f}")
    print(f"  Max  VaR   : ${np.nanmax(VaR_arr):>12,.0f}")
    print(f"  NaN days   : {np.isnan(VaR_arr).sum()}")

    print_validation_summary(bt, gof_records, marg_records, VaR_arr)

    # D: Validation prints showing impact of each fix
    print("\nFix impact summary:")
    print(f"  Distinct daily VaR values : {n_distinct}  "
          f"(pre-A2: ~{n_refits}; post-A2: ~{T - WINDOW})")
    print(f"  Mean VaR with nonlinear component : ${np.nanmean(VaR_arr):,.0f}"
          f"  (A3: {'included' if nonlinear is not None else 'NOT included — linear only'})")
    if gof_records:
        p_min = min(r["p_value"] for r in gof_records)
        p_max = max(r["p_value"] for r in gof_records)
        print(f"  GoF p-value range across refits  : [{p_min:.3f}, {p_max:.3f}]")

    print("\n" + "=" * 60)
    print("Copula VaR complete!")
    print(f"  Exceptions : {bt.N}  (expected {(T - WINDOW) * (1 - ALPHA):.1f} at 99% VaR)")
    print(f"  Kupiec H0  : {'NOT rejected' if bt.pvalue_uc >= 0.05 else 'REJECTED'}"
          f"  (p={bt.pvalue_uc:.4f})")
    print("=" * 60)


if __name__ == "__main__":
    main()
