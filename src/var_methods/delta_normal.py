# =============================================================================
# delta_normal.py
# 1-day 99% Delta-Normal VaR — linear book only
#
# Theory references (Irle lecture notes):
#
# [Delta-Normal / Variance-Covariance (Irle p. 58-64)]:
#   Portfolio loss:  ΔV = -w^T Σ w  (quadratic form in factor returns)
#   Under normality: ΔV ~ N(μ_ΔV, σ²_ΔV)
#   VaR_α = z_α * σ_ΔV - μ_ΔV           (full, includes drift)
#   VaR_α = z_α * σ_ΔV                   (no-drift, textbook convention)
#
# [Expected Shortfall under normality]:
#   ES_α = σ_ΔV * φ(z_α) / (1 - α) - μ_ΔV
#   where φ is the standard normal PDF.  Closed-form; no simulation required.
#   ES answers "given VaR is breached, what is the average loss?"
#
# [Window sensitivity]:
#   Re-runs VaR at WINDOW ∈ {250, 500, 750} with all else fixed.
#   Addresses Irle project question: "Does the data history assumption
#   materially affect your VaR estimates?"
#
# [Crisis-period backtesting]:
#   Full-sample Kupiec / Christoffersen results supplemented by subperiod
#   diagnostics: GFC 2008, COVID 2020, Rate Hikes 2022, non-crisis.
#   Short windows mean p-values are descriptive; exception rates are the
#   primary diagnostic.
#
# SCOPE
#   VaR is estimated on the 4-asset linear book (SPY, IEF, GLD, EURUSD)
#   using equal weights w_j = 1/4, dollar exposure per asset = V0 / N_ASSETS.
#   Backtesting is performed against pnl_total (all instruments including IRS
#   and straddle) to produce a fair cross-method comparison.  The resulting
#   inflated exception rate reflects two compounding failures:
#     (a) normality assumption — fat tails in daily returns
#     (b) scope gap — non-linear instruments not modelled by Delta-Normal
#   Both are documentable and expected; together they motivate GARCH+EVT.
#
# WEIGHTS
#   Equal weights (1/4 per linear asset) consistent with the rest of the
#   project's portfolio convention: V0 = $1,000,000 equally allocated across
#   SPY, IEF, GLD, EURUSD.  Dollar exposure per asset = V0 * (1/N_ASSETS).
#   Assumption: daily rebalancing back to equal weight (constant exposure).
#
# DATA SOURCES
#   log_returns.csv     — processed daily log-returns for the 4 linear assets
#   total_portfolio_pnl.csv — pnl_total for backtesting (all instruments)
#
# =============================================================================

import sys
import warnings

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import norm, t as student_t

from pathlib import Path

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[2]

sys.path.insert(0, str(_ROOT))
from backtesting.backtest import run_backtest
from backtesting.plot_backtest import plot_all

# =============================================================================
# SETTINGS
# =============================================================================

WINDOW          = 500
ALPHA           = 0.99
Z_ALPHA         = norm.ppf(ALPHA)       # ≈ 2.326
PHI_ALPHA       = norm.pdf(Z_ALPHA)     # ≈ 0.0267 — used in closed-form ES
V0              = 1_000_000
LINEAR_ASSETS   = ["EURUSD", "GLD", "IEF", "SPY"]
N_ASSETS        = len(LINEAR_ASSETS)
WEIGHT          = 1.0 / N_ASSETS        # equal weight: 0.25 per asset

SENSITIVITY_WINDOWS = [250, 500, 750]   # window sensitivity sweep
EWMA_LAMBDAS        = [0.94, 0.97]      # RiskMetrics (0.94) and smoother variant

CRISIS_WINDOWS = [
    ("GFC 2008",        "2008-09-15", "2009-03-09"),
    ("COVID 2020",      "2020-02-19", "2020-03-23"),
    ("Rate Hikes 2022", "2022-01-01", "2022-10-01"),
]

PROCESSED_DIR = _ROOT / "data" / "processed"
OUTPUT_FIGS   = _ROOT / "outputs" / "figures"
OUTPUT_TABLES = _ROOT / "outputs" / "tables"

# =============================================================================
# GUARDS
# =============================================================================

def _assert_aligned(pnl: pd.Series, results: pd.DataFrame, context: str = "") -> None:
    """
    Raise ValueError if any results date is absent from pnl, or if the
    aligned loss series contains NaN.  Prevents silent misalignment bugs
    from propagating into backtests, plots, and save operations.
    """
    missing = results.index.difference(pnl.index)
    if len(missing):
        raise ValueError(
            f"{context}: {len(missing)} result dates not in pnl index. "
            f"First few: {missing[:5].tolist()}"
        )
    n_nan = int((-pnl.reindex(results.index)).isna().sum())
    if n_nan:
        raise ValueError(
            f"{context}: {n_nan} NaN(s) in aligned loss series. "
            "Check pnl for gaps on backtest dates."
        )


def _validate_results(results: pd.DataFrame) -> None:
    """
    Lightweight sanity checks on compute_delta_normal_var output.
    Raises ValueError on any violation.
    """
    errs = []
    for col in ("VaR_DN_full", "VaR_DN_nodrift", "ES_DN_full",
                "mu_deltaV", "sigma_deltaV"):
        n_bad = (~np.isfinite(results[col].values.astype(float))).sum()
        if n_bad:
            errs.append(f"{col}: {n_bad} non-finite values")

    if (results["sigma_deltaV"] < 0).any():
        errs.append("sigma_deltaV: negative values found")

    if (results["VaR_DN_full"] < 0).any():
        n = int((results["VaR_DN_full"] < 0).sum())
        errs.append(f"VaR_DN_full: {n} negative values")

    # ES >= VaR by construction (same σ_ΔV, ES adds positive term φ/[1-α])
    tol = 1e-8
    n_below = (results["ES_DN_full"] < results["VaR_DN_full"] - tol).sum()
    if n_below:
        errs.append(f"ES_DN_full < VaR_DN_full: {n_below} rows")

    if errs:
        raise ValueError("_validate_results failed:\n" +
                         "\n".join(f"  - {e}" for e in errs))


def _ewma_cov(window_r: np.ndarray, lam: float) -> np.ndarray:
    """
    Exponentially weighted covariance matrix (RiskMetrics convention).

    Weights decay geometrically: most recent observation receives weight
    proportional to λ^0 = 1, oldest to λ^(T-1). Weights are normalised
    to sum to 1. Demeans using the window sample mean before weighting.

    Parameters
    ----------
    window_r : (T, d) array of returns
    lam      : decay factor ∈ (0, 1); λ=0.94 is the RiskMetrics daily default

    Returns
    -------
    (d, d) EWMA covariance matrix
    """
    T, d     = window_r.shape
    mu       = window_r.mean(axis=0)
    demeaned = window_r - mu
    # w[0] = oldest weight ∝ λ^(T-1), w[T-1] = newest weight ∝ λ^0
    weights  = lam ** np.arange(T - 1, -1, -1)
    weights /= weights.sum()
    return np.einsum("i,ij,ik->jk", weights, demeaned, demeaned)

# =============================================================================
# STEP 1 — LOAD DATA
# =============================================================================

def load_data():
    """
    Load processed log-returns and total portfolio P&L.

    log_returns.csv    : daily log-returns for the 4 linear assets.
                         Columns must include EURUSD, GLD, IEF, SPY.
    total_portfolio_pnl.csv : pnl_total column — used as the backtest target.
                         Includes IRS and straddle P&L so the comparison
                         against other VaR methods is on equal footing.

    Returns
    -------
    log_ret  : pd.DataFrame  log-returns (T, 4), columns = LINEAR_ASSETS
    pnl      : pd.Series     total portfolio P&L (for backtesting)
    """
    log_ret = pd.read_csv(
        PROCESSED_DIR / "log_returns.csv",
        index_col=0, parse_dates=True
    )

    # Normalise FX column name if the pipeline emits "EURUSD=X"
    log_ret = log_ret.rename(columns={"EURUSD=X": "EURUSD"})

    missing = set(LINEAR_ASSETS) - set(log_ret.columns)
    if missing:
        raise ValueError(f"log_returns.csv missing columns: {sorted(missing)}")

    log_ret = log_ret[LINEAR_ASSETS].dropna()

    pnl = pd.read_csv(
        PROCESSED_DIR / "total_portfolio_pnl.csv",
        index_col=0, parse_dates=True
    )["pnl_total"]

    # Align on common dates
    common  = log_ret.index.intersection(pnl.index)
    log_ret = log_ret.loc[common]
    pnl     = pnl.loc[common]

    print(f"Loaded: {len(common)} days")
    print(f"Date range : {common[0].date()} -> {common[-1].date()}")
    print(f"Assets     : {LINEAR_ASSETS}")
    print(f"Weight     : {WEIGHT:.4f} per asset  (equal, V0=${V0:,})")
    print(f"pnl_total  : mean=${pnl.mean():,.0f}  std=${pnl.std():,.0f}")
    return log_ret, pnl

# =============================================================================
# STEP 2 — ROLLING DELTA-NORMAL VAR
# =============================================================================

def compute_delta_normal_var(log_ret: pd.DataFrame,
                             window: int = WINDOW) -> pd.DataFrame:
    """
    Rolling 500-day Delta-Normal VaR and ES (Irle p. 58-64).

    For each forecast date t:
      1. Estimate μ_r and Σ_r from log_ret[t-window : t].
      2. Dollar exposure: e_j = V0 / N_ASSETS (constant, equal weight).
      3. Portfolio P&L moments:
           μ_ΔV = e^T μ_r
           σ_ΔV = sqrt(e^T Σ_r e)
      4. VaR (Irle p. 61):
           VaR_full    = z_α * σ_ΔV - μ_ΔV
           VaR_nodrift = z_α * σ_ΔV
      5. ES under normality (closed-form):
           ES_full    = φ(z_α) / (1-α) * σ_ΔV - μ_ΔV
           ES_nodrift = φ(z_α) / (1-α) * σ_ΔV
         where φ = norm.pdf(z_α).  ES >= VaR by construction since
         φ(z_α)/(1-α) > z_α for all α ∈ (0, 1).

    Returns
    -------
    pd.DataFrame  index=forecast_date
      columns: VaR_DN_full, VaR_DN_nodrift, ES_DN_full, ES_DN_nodrift,
               mu_deltaV, sigma_deltaV
    """
    n           = len(log_ret)
    exposure    = np.full(N_ASSETS, V0 * WEIGHT)
    columns     = LINEAR_ASSETS
    r_vals      = log_ret[columns].values
    dates       = log_ret.index

    records = []
    print(f"\n{'='*60}")
    print(f"Delta-Normal VaR  |  alpha={ALPHA:.0%}  |  z={Z_ALPHA:.4f}")
    print(f"window={window} days  |  N_ASSETS={N_ASSETS}  |  weight={WEIGHT:.4f}")
    print(f"Computing VaR for {n - window} days ...")

    for t in range(window, n):
        window_r = r_vals[t - window : t]
        mu_r     = window_r.mean(axis=0)            # (4,)
        sigma_r  = np.cov(window_r.T)               # (4, 4)

        mu_dv    = float(exposure @ mu_r)
        var_dv   = float(exposure @ sigma_r @ exposure)
        sigma_dv = np.sqrt(max(var_dv, 0.0))

        var_full    = Z_ALPHA * sigma_dv - mu_dv
        var_nodrift = Z_ALPHA * sigma_dv

        # Closed-form ES under normality: φ(z_α) / (1-α) > z_α always,
        # so ES_full >= VaR_full by construction.
        es_factor   = PHI_ALPHA / (1.0 - ALPHA)
        es_full     = es_factor * sigma_dv - mu_dv
        es_nodrift  = es_factor * sigma_dv

        records.append({
            "VaR_DN_full"    : var_full,
            "VaR_DN_nodrift" : var_nodrift,
            "ES_DN_full"     : es_full,
            "ES_DN_nodrift"  : es_nodrift,
            "mu_deltaV"      : mu_dv,
            "sigma_deltaV"   : sigma_dv,
        })

    results = pd.DataFrame(records, index=dates[window:])

    print(f"\nDelta-Normal summary:")
    print(f"  Mean VaR (full)    : ${results['VaR_DN_full'].mean():>12,.0f}")
    print(f"  Mean VaR (nodrift) : ${results['VaR_DN_nodrift'].mean():>12,.0f}")
    print(f"  Mean ES  (full)    : ${results['ES_DN_full'].mean():>12,.0f}")
    print(f"  Min  VaR           : ${results['VaR_DN_full'].min():>12,.0f}")
    print(f"  Max  VaR           : ${results['VaR_DN_full'].max():>12,.0f}")

    _validate_results(results)
    return results

# =============================================================================
# STEP 3 — BACKTEST
# =============================================================================

def backtest_delta_normal(pnl: pd.Series, results: pd.DataFrame):
    """
    Kupiec + Christoffersen backtests via shared framework.

    Backtest target: pnl_total (all instruments).
    VaR estimate:    Delta-Normal on linear book only.

    Expected result: elevated exception rate driven by
      (a) normality failure — fat tails confirmed by exploratory analysis
      (b) scope gap — non-linear instruments not captured by Delta-Normal
    """
    _assert_aligned(pnl, results, context="backtest_delta_normal")
    bt = run_backtest(
        pnl=pnl,
        var=results["VaR_DN_full"],
        confidence=ALPHA,
        method_name="Delta-Normal",
    )
    print(bt)
    return bt

# =============================================================================
# STEP 4 — PLOTS
# =============================================================================

def plot_var_results(pnl: pd.Series, results: pd.DataFrame, bt) -> None:
    """
    Two-panel figure:
      Panel 1: VaR_DN_full and VaR_DN_nodrift vs actual loss, with exceptions.
      Panel 2: Rolling σ_ΔV (portfolio volatility in USD) over time.
               Volatility spikes during GFC and COVID confirm that normality
               is the binding assumption — not the portfolio structure.
    """
    OUTPUT_FIGS.mkdir(parents=True, exist_ok=True)

    crises = [
        ("2008-09-15", "2009-03-09", "GFC 2008"),
        ("2020-02-19", "2020-03-23", "COVID 2020"),
        ("2022-01-01", "2022-10-01", "Rate Hikes 2022"),
    ]

    _assert_aligned(pnl, results, context="plot_var_results")
    actual_loss = -pnl.reindex(results.index)
    exc_idx     = bt.exceptions_index

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Panel 1: VaR and ES vs actual loss
    axes[0].fill_between(results.index, 0, results["VaR_DN_full"],
                         alpha=0.10, color="#1565C0")
    axes[0].plot(results.index, results["VaR_DN_full"],
                 color="#1565C0", linewidth=1.3,
                 label="VaR_DN_full  (Irle p. 61)")
    axes[0].plot(results.index, results["ES_DN_full"],
                 color="#1565C0", linewidth=0.8, linestyle="--", alpha=0.7,
                 label="ES_DN_full  (closed-form normal)")
    axes[0].plot(results.index, results["VaR_DN_nodrift"],
                 color="#42A5F5", linewidth=0.9, linestyle=":",
                 label="VaR_DN_nodrift  (z·σ only)")
    axes[0].plot(actual_loss.index, actual_loss,
                 color="#90A4AE", linewidth=0.6, alpha=0.7,
                 label="Actual loss  (pnl_total)")
    if exc_idx is not None and len(exc_idx) > 0:
        axes[0].scatter(exc_idx, actual_loss.loc[exc_idx],
                        color="#D32F2F", s=22, zorder=5,
                        label=f"Exceptions  N={bt.N}  ({bt.exception_rate:.2%})")
    axes[0].axhline(0, color="black", linewidth=0.5, linestyle="--")
    axes[0].set_title(
        "Delta-Normal VaR 99% vs Actual Portfolio Loss (pnl_total)\n"
        "Linear book only — exceptions driven by fat tails + non-linear scope gap",
        fontsize=12, fontweight="bold",
    )
    axes[0].set_ylabel("USD")
    axes[0].legend(fontsize=9)

    # Panel 2: rolling σ_ΔV
    axes[1].plot(results.index, results["sigma_deltaV"],
                 color="#6A1B9A", linewidth=0.9,
                 label="σ_ΔV  (USD portfolio vol, linear book)")
    axes[1].set_title(
        "Rolling Portfolio Volatility σ_ΔV  |  500-day window  |  Irle p. 61",
        fontsize=12, fontweight="bold",
    )
    axes[1].set_ylabel("σ_ΔV  (USD)")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[1].legend(fontsize=9)

    for ax in axes:
        for s, e, lbl in crises:
            ax.axvspan(s, e, alpha=0.07, color="red")

    plt.tight_layout()
    path = OUTPUT_FIGS / "dn_01_var_results.png"
    plt.savefig(path, dpi=150)
    plt.close("all")
    print(f"Results plot saved -> {path}")

# =============================================================================
# STEP 5 — SAVE OUTPUTS
# =============================================================================

def save_results(pnl: pd.Series, results: pd.DataFrame, bt) -> None:
    """
    Save VaR time series and backtest detail table.
    Convention matches evt.py and garch_evt.py.
    """
    _assert_aligned(pnl, results, context="save_results")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)

    results.to_csv(PROCESSED_DIR / "var_delta_normal.csv")
    print(f"VaR saved     -> {PROCESSED_DIR / 'var_delta_normal.csv'}")

    common = pnl.index.intersection(results.index)
    loss   = -pnl.reindex(common)
    pd.DataFrame({
        "VaR"            : results.loc[common, "VaR_DN_full"].values,
        "VaR_nodrift"    : results.loc[common, "VaR_DN_nodrift"].values,
        "ES"             : results.loc[common, "ES_DN_full"].values,
        "ES_nodrift"     : results.loc[common, "ES_DN_nodrift"].values,
        "actual_loss"    : loss.values,
        "exception"      : (loss > results.loc[common, "VaR_DN_full"]).astype(int).values,
        "mu_deltaV"      : results.loc[common, "mu_deltaV"].values,
        "sigma_deltaV"   : results.loc[common, "sigma_deltaV"].values,
    }, index=common).to_csv(
        OUTPUT_TABLES / "backtest_delta_normal.csv"
    )
    print(f"Backtest      -> {OUTPUT_TABLES / 'backtest_delta_normal.csv'}")

# =============================================================================
# STEP 6 — CRISIS-PERIOD BACKTESTING
# =============================================================================

def run_crisis_backtests(pnl: pd.Series, results: pd.DataFrame,
                         alpha: float = ALPHA) -> pd.DataFrame:
    """
    Subperiod backtesting across crisis and non-crisis windows.

    Slices pnl and VaR_DN_full by date range and calls run_backtest for each
    period where at least 50 observations are available.  Short windows mean
    p-values are descriptive; exception rates and mean VaR are the primary output.

    Periods: GFC 2008, COVID 2020, Rate Hikes 2022, Non-crisis, Full sample.

    Saves -> outputs/tables/delta_normal_crisis_backtest.csv
    """
    _assert_aligned(pnl, results, context="run_crisis_backtests")
    actual_loss = -pnl.reindex(results.index)
    exception   = actual_loss > results["VaR_DN_full"]
    idx         = results.index

    crisis_mask = pd.Series(False, index=idx)
    for _, s, e in CRISIS_WINDOWS:
        crisis_mask |= (idx >= s) & (idx <= e)

    periods = [("Full sample", slice(None))]
    for name, s, e in CRISIS_WINDOWS:
        periods.append((name, (idx >= s) & (idx <= e)))
    periods.append(("Non-crisis", (~crisis_mask).values))

    rows = []
    for period_name, mask in periods:
        sub_res  = results.loc[mask]
        sub_loss = actual_loss.loc[mask]
        sub_exc  = exception.loc[mask]
        n_obs    = len(sub_res)

        kupiec_p = christoffersen_p = np.nan
        note = ""
        if n_obs >= 50:
            try:
                bt_sub = run_backtest(
                    pnl=pnl.reindex(sub_res.index),
                    var=sub_res["VaR_DN_full"],
                    confidence=alpha,
                    method_name=f"DN ({period_name})",
                )
                kupiec_p        = bt_sub.pvalue_uc
                christoffersen_p = getattr(bt_sub, "pvalue_cc", np.nan)
                if period_name == "Non-crisis":
                    note = "non-contiguous; Christoffersen p descriptive only"
            except Exception as exc_e:
                note = f"backtest error: {exc_e}"
        else:
            note = "< 50 obs; descriptive only"

        rows.append({
            "period"              : period_name,
            "start"               : str(sub_res.index[0].date()) if n_obs else "",
            "end"                 : str(sub_res.index[-1].date()) if n_obs else "",
            "n_obs"               : n_obs,
            "expected_exceptions" : n_obs * (1 - alpha),
            "exceptions"          : int(sub_exc.sum()),
            "exception_rate"      : sub_exc.mean() if n_obs else np.nan,
            "mean_VaR"            : sub_res["VaR_DN_full"].mean(),
            "mean_ES"             : sub_res["ES_DN_full"].mean(),
            "max_VaR"             : sub_res["VaR_DN_full"].max(),
            "max_actual_loss"     : sub_loss.max(),
            "avg_exception_loss"  : sub_loss[sub_exc].mean() if sub_exc.any() else np.nan,
            "kupiec_p"            : kupiec_p,
            "christoffersen_p"    : christoffersen_p,
            "note"                : note,
        })

    crisis_df = pd.DataFrame(rows)
    path = OUTPUT_TABLES / "delta_normal_crisis_backtest.csv"
    crisis_df.to_csv(path, index=False)
    print(f"\nCrisis backtest saved -> {path}")

    print(f"\n  Crisis-period summary (alpha={alpha:.0%}):")
    print(f"  {'period':<20}  {'n_obs':>6}  {'exc':>5}  {'exp':>6}  "
          f"{'rate':>7}  {'mean_VaR':>12}  {'mean_ES':>12}")
    print(f"  {'-'*75}")
    for row in rows:
        rate_s = f"{row['exception_rate']:.3f}" if np.isfinite(row['exception_rate']) else "  N/A"
        print(f"  {row['period']:<20}  {row['n_obs']:>6}  "
              f"{row['exceptions']:>5}  {row['expected_exceptions']:>6.1f}  "
              f"{rate_s:>7}  ${row['mean_VaR']:>11,.0f}  ${row['mean_ES']:>11,.0f}")
    return crisis_df


# =============================================================================
# STEP 7 — WINDOW SENSITIVITY
# =============================================================================

def window_sensitivity(log_ret: pd.DataFrame, pnl: pd.Series,
                       windows: list = SENSITIVITY_WINDOWS,
                       alpha: float = ALPHA) -> pd.DataFrame:
    """
    Re-run Delta-Normal VaR at multiple window lengths with all else fixed.

    Addresses Irle project question: "Does the data history assumption
    materially affect your VaR estimates?"

    For each window in windows:
      - compute_delta_normal_var() with that window
      - run_backtest() against pnl_total
      - collect mean VaR, mean ES, exceptions, Kupiec p

    Saves -> outputs/tables/delta_normal_window_sensitivity.csv
    """
    print(f"\n{'='*60}")
    print(f"Window sensitivity sweep  |  windows={windows}")

    rows = []
    for w in windows:
        res_w = compute_delta_normal_var(log_ret, window=w)
        _assert_aligned(pnl, res_w, context=f"window_sensitivity(w={w})")
        bt_w = run_backtest(
            pnl=pnl,
            var=res_w["VaR_DN_full"],
            confidence=alpha,
            method_name=f"DN (w={w})",
        )
        rows.append({
            "window"         : w,
            "n_backtest_days": len(res_w),
            "mean_VaR"       : res_w["VaR_DN_full"].mean(),
            "mean_ES"        : res_w["ES_DN_full"].mean(),
            "max_VaR"        : res_w["VaR_DN_full"].max(),
            "exceptions"     : bt_w.N,
            "exception_rate" : bt_w.exception_rate,
            "kupiec_p"       : bt_w.pvalue_uc,
            "christoffersen_p": getattr(bt_w, "pvalue_cc", np.nan),
        })

    sens_df = pd.DataFrame(rows)
    path = OUTPUT_TABLES / "delta_normal_window_sensitivity.csv"
    sens_df.to_csv(path, index=False)

    print(f"\n  {'window':>8}  {'mean_VaR':>12}  {'mean_ES':>12}  "
          f"{'exceptions':>10}  {'kupiec_p':>10}")
    print(f"  {'-'*58}")
    for row in rows:
        print(f"  {row['window']:>8}  ${row['mean_VaR']:>11,.0f}  "
              f"${row['mean_ES']:>11,.0f}  "
              f"{row['exceptions']:>10}  {row['kupiec_p']:>10.4f}")
    print(f"  Saved -> {path}")
    return sens_df






# =============================================================================
# STEP 8 — EWMA COVARIANCE SENSITIVITY
# =============================================================================

def ewma_sensitivity(log_ret: pd.DataFrame, pnl: pd.Series,
                     lambdas: list = EWMA_LAMBDAS,
                     window: int = WINDOW,
                     alpha: float = ALPHA) -> pd.DataFrame:
    """
    Replace the sample covariance with EWMA covariance at each λ in lambdas.

    EWMA assigns exponentially decaying weights to past observations so that
    recent returns dominate the covariance estimate.  This makes VaR more
    responsive to volatility regime changes — it should spike faster at
    GFC/COVID onset than the equal-weight sample covariance.

    λ = 0.94 is the RiskMetrics daily standard (half-life ≈ 11 days).
    λ = 0.97 decays more slowly (half-life ≈ 23 days), closer to sample cov.

    For each λ:
      - Roll through the backtest period using _ewma_cov(window_r, lam)
        instead of np.cov(window_r.T)
      - All other formula components (exposure, VaR, ES) are unchanged
      - run_backtest() against pnl_total

    Baseline (sample covariance, equal weights) is included as λ = 1.0.

    Saves -> outputs/tables/delta_normal_ewma_sensitivity.csv
    """
    print(f"\n{'='*60}")
    print(f"EWMA covariance sensitivity  |  λ={lambdas}  |  window={window}")

    exposure = np.full(N_ASSETS, V0 * WEIGHT)
    r_vals   = log_ret[LINEAR_ASSETS].values
    dates    = log_ret.index
    n        = len(log_ret)
    z        = norm.ppf(alpha)
    phi      = norm.pdf(z)
    es_fac   = phi / (1.0 - alpha)

    # Include sample covariance as baseline (λ → 1 limit)
    all_lambdas = [None] + list(lambdas)   # None = sample cov

    rows = []
    for lam in all_lambdas:
        label   = "sample (baseline)" if lam is None else f"EWMA λ={lam}"
        records = []
        for t in range(window, n):
            w_r      = r_vals[t - window : t]
            mu_r     = w_r.mean(axis=0)
            sigma_r  = np.cov(w_r.T) if lam is None else _ewma_cov(w_r, lam)
            mu_dv    = float(exposure @ mu_r)
            sigma_dv = np.sqrt(max(float(exposure @ sigma_r @ exposure), 0.0))
            records.append({
                "VaR_DN_full": z * sigma_dv - mu_dv,
                "ES_DN_full" : es_fac * sigma_dv - mu_dv,
            })

        res_lam = pd.DataFrame(records, index=dates[window:])
        _assert_aligned(pnl, res_lam, context=f"ewma_sensitivity(lam={lam})")
        bt_lam  = run_backtest(
            pnl=pnl,
            var=res_lam["VaR_DN_full"],
            confidence=alpha,
            method_name=f"DN ({label})",
        )
        rows.append({
            "lambda"           : lam if lam is not None else 1.0,
            "label"            : label,
            "mean_VaR"         : res_lam["VaR_DN_full"].mean(),
            "mean_ES"          : res_lam["ES_DN_full"].mean(),
            "max_VaR"          : res_lam["VaR_DN_full"].max(),
            "exceptions"       : bt_lam.N,
            "exception_rate"   : bt_lam.exception_rate,
            "kupiec_p"         : bt_lam.pvalue_uc,
            "christoffersen_p" : getattr(bt_lam, "pvalue_cc", np.nan),
        })
        print(f"  {label:<22}  mean_VaR=${rows[-1]['mean_VaR']:>10,.0f}  "
              f"exc={bt_lam.N:>4}  kupiec_p={bt_lam.pvalue_uc:.4f}")

    sens_df = pd.DataFrame(rows)
    path    = OUTPUT_TABLES / "delta_normal_ewma_sensitivity.csv"
    sens_df.to_csv(path, index=False)
    print(f"  Saved -> {path}")
    return sens_df


# =============================================================================
# STEP 9 — DELTA-t-NORMAL SENSITIVITY
# =============================================================================

def t_normal_sensitivity(log_ret: pd.DataFrame, pnl: pd.Series,
                         window: int = WINDOW,
                         alpha: float = ALPHA) -> pd.DataFrame:
    """
    Replace the Normal quantile with a Student-t quantile (Delta-t-Normal).

    The parametric fix for fat tails: same covariance structure as Delta-Normal
    but the tail quantile z_α comes from t_ν rather than N(0,1).  Since
    t_ν.ppf(0.99) > norm.ppf(0.99) for any finite ν, VaR will be uniformly
    higher and the exception rate will fall.

    The gap between the Normal and t exception rates isolates the normality
    assumption's contribution to the exception excess — separate from the
    non-linear scope gap.

    ν is estimated once from the full portfolio return series using MLE
    (scipy.stats.t.fit with location fixed at 0).  Using a fixed ν is the
    standard Delta-t-Normal approach; rolling ν estimation would add noise
    without improving the sensitivity comparison.

    ES under Student-t (closed-form, McNeil-Frey-Embrechts 2005, p. 46):
      ES_α = σ_ΔV * f_ν(z_ν) / (1-α) * (ν + z_ν²) / (ν - 1) - μ_ΔV
    where f_ν = student_t.pdf(z_ν, df=ν).  Requires ν > 1; clamped below at 2.1.

    Saves -> outputs/tables/delta_normal_t_sensitivity.csv
    """
    print(f"\n{'='*60}")
    print(f"Delta-t-Normal sensitivity  |  window={window}")

    # Estimate ν from portfolio returns (equal-weight mean of log-returns)
    port_ret = log_ret[LINEAR_ASSETS].mean(axis=1).dropna().values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nu_fit, _, _ = student_t.fit(port_ret, floc=0)
    nu = max(float(nu_fit), 2.1)   # ν > 2 ensures finite variance; > 1 for ES

    z_t     = float(student_t.ppf(alpha, df=nu))
    f_t     = float(student_t.pdf(z_t, df=nu))
    es_fac_t = f_t / (1.0 - alpha) * (nu + z_t ** 2) / (nu - 1.0)

    z_n     = norm.ppf(alpha)
    es_fac_n = norm.pdf(z_n) / (1.0 - alpha)

    print(f"  Full-sample ν estimate : {nu:.2f}")
    print(f"  Normal z_α={z_n:.4f}   t_ν z_α={z_t:.4f}  "
          f"(t quantile is {z_t - z_n:+.4f} higher)")

    exposure = np.full(N_ASSETS, V0 * WEIGHT)
    r_vals   = log_ret[LINEAR_ASSETS].values
    dates    = log_ret.index
    n        = len(log_ret)

    rows = []
    for label, use_t in [("Normal (baseline)", False), (f"Student-t ν={nu:.1f}", True)]:
        z_use      = z_t      if use_t else z_n
        es_fac_use = es_fac_t if use_t else es_fac_n
        records    = []
        for t in range(window, n):
            w_r      = r_vals[t - window : t]
            mu_r     = w_r.mean(axis=0)
            sigma_r  = np.cov(w_r.T)
            mu_dv    = float(exposure @ mu_r)
            sigma_dv = np.sqrt(max(float(exposure @ sigma_r @ exposure), 0.0))
            records.append({
                "VaR_DN_full": z_use * sigma_dv - mu_dv,
                "ES_DN_full" : es_fac_use * sigma_dv - mu_dv,
            })

        res_t = pd.DataFrame(records, index=dates[window:])
        _assert_aligned(pnl, res_t, context=f"t_normal_sensitivity({label})")
        bt_t  = run_backtest(
            pnl=pnl,
            var=res_t["VaR_DN_full"],
            confidence=alpha,
            method_name=f"DN ({label})",
        )
        rows.append({
            "distribution"     : label,
            "nu"               : nu if use_t else np.nan,
            "z_alpha"          : z_use,
            "mean_VaR"         : res_t["VaR_DN_full"].mean(),
            "mean_ES"          : res_t["ES_DN_full"].mean(),
            "max_VaR"          : res_t["VaR_DN_full"].max(),
            "exceptions"       : bt_t.N,
            "exception_rate"   : bt_t.exception_rate,
            "kupiec_p"         : bt_t.pvalue_uc,
            "christoffersen_p" : getattr(bt_t, "pvalue_cc", np.nan),
        })
        print(f"  {label:<30}  mean_VaR=${rows[-1]['mean_VaR']:>10,.0f}  "
              f"exc={bt_t.N:>4}  kupiec_p={bt_t.pvalue_uc:.4f}")

    # Exception reduction attributable to fat-tail correction
    exc_normal = rows[0]["exceptions"]
    exc_t      = rows[1]["exceptions"]
    print(f"\n  Exception reduction (Normal → t): {exc_normal} → {exc_t} "
          f"({exc_normal - exc_t:+d})")
    print(f"  Remaining excess above expected: "
          f"{exc_t - round(len(res_t) * (1 - alpha)):.0f}  "
          f"(attributable to scope gap + residual fat tails)")

    sens_df = pd.DataFrame(rows)
    path    = OUTPUT_TABLES / "delta_normal_t_sensitivity.csv"
    sens_df.to_csv(path, index=False)
    print(f"  Saved -> {path}")
    return sens_df, nu


# =============================================================================
# STEP 10 — CORRELATION STRUCTURE SENSITIVITY
# =============================================================================

def correlation_sensitivity(log_ret: pd.DataFrame, pnl: pd.Series,
                             window: int = WINDOW,
                             alpha: float = ALPHA) -> pd.DataFrame:
    """
    Compare full covariance vs diagonal covariance (no inter-asset correlations).

    Setting off-diagonal elements of Σ to zero removes all diversification
    benefit from the VaR estimate.  The gap between diagonal VaR and full VaR
    is the diversification benefit the model claims.

    Diagonal VaR > Full VaR by construction when any correlations are positive
    (which they are for a portfolio containing equities, bonds, gold, FX).

    Diversification benefit per day = VaR_diagonal - VaR_full
    Mean diversification benefit over the backtest period is the headline figure.

    Saves -> outputs/tables/delta_normal_correlation_sensitivity.csv
    """
    print(f"\n{'='*60}")
    print(f"Correlation structure sensitivity  |  window={window}")

    exposure = np.full(N_ASSETS, V0 * WEIGHT)
    r_vals   = log_ret[LINEAR_ASSETS].values
    dates    = log_ret.index
    n        = len(log_ret)
    z        = norm.ppf(alpha)
    phi      = norm.pdf(z)
    es_fac   = phi / (1.0 - alpha)

    records = []
    for t in range(window, n):
        w_r      = r_vals[t - window : t]
        mu_r     = w_r.mean(axis=0)
        sigma_r  = np.cov(w_r.T)
        sigma_diag = np.diag(np.diag(sigma_r))     # zero off-diagonal

        mu_dv       = float(exposure @ mu_r)
        sigma_full  = np.sqrt(max(float(exposure @ sigma_r    @ exposure), 0.0))
        sigma_diag_ = np.sqrt(max(float(exposure @ sigma_diag @ exposure), 0.0))

        var_full  = z * sigma_full  - mu_dv
        var_diag  = z * sigma_diag_ - mu_dv
        es_full   = es_fac * sigma_full  - mu_dv
        es_diag   = es_fac * sigma_diag_ - mu_dv

        records.append({
            "VaR_full"          : var_full,
            "VaR_diagonal"      : var_diag,
            "ES_full"           : es_full,
            "ES_diagonal"       : es_diag,
            "diversif_benefit"  : var_diag - var_full,   # always >= 0
            "sigma_full"        : sigma_full,
            "sigma_diagonal"    : sigma_diag_,
        })

    res_corr = pd.DataFrame(records, index=dates[window:])
    _assert_aligned(pnl, res_corr, context="correlation_sensitivity(full)")

    bt_full = run_backtest(
        pnl=pnl,
        var=res_corr["VaR_full"],
        confidence=alpha,
        method_name="DN (full cov)",
    )
    bt_diag = run_backtest(
        pnl=pnl,
        var=res_corr["VaR_diagonal"],
        confidence=alpha,
        method_name="DN (diagonal cov)",
    )

    mean_benefit = res_corr["diversif_benefit"].mean()
    max_benefit  = res_corr["diversif_benefit"].max()
    mean_pct     = (res_corr["diversif_benefit"] / res_corr["VaR_full"]).mean()

    print(f"  Mean diversification benefit : ${mean_benefit:>10,.0f}  "
          f"({mean_pct:.1%} of full VaR)")
    print(f"  Max  diversification benefit : ${max_benefit:>10,.0f}")
    print(f"  Full cov   exc={bt_full.N}  kupiec_p={bt_full.pvalue_uc:.4f}")
    print(f"  Diagonal   exc={bt_diag.N}  kupiec_p={bt_diag.pvalue_uc:.4f}")

    summary = pd.DataFrame([
        {
            "cov_structure"    : "full",
            "mean_VaR"         : res_corr["VaR_full"].mean(),
            "mean_ES"          : res_corr["ES_full"].mean(),
            "exceptions"       : bt_full.N,
            "exception_rate"   : bt_full.exception_rate,
            "kupiec_p"         : bt_full.pvalue_uc,
            "christoffersen_p" : getattr(bt_full, "pvalue_cc", np.nan),
            "mean_diversif_benefit" : np.nan,
            "mean_diversif_pct"     : np.nan,
        },
        {
            "cov_structure"    : "diagonal",
            "mean_VaR"         : res_corr["VaR_diagonal"].mean(),
            "mean_ES"          : res_corr["ES_diagonal"].mean(),
            "exceptions"       : bt_diag.N,
            "exception_rate"   : bt_diag.exception_rate,
            "kupiec_p"         : bt_diag.pvalue_uc,
            "christoffersen_p" : getattr(bt_diag, "pvalue_cc", np.nan),
            "mean_diversif_benefit" : mean_benefit,
            "mean_diversif_pct"     : mean_pct,
        },
    ])

    path = OUTPUT_TABLES / "delta_normal_correlation_sensitivity.csv"
    summary.to_csv(path, index=False)

    # Also save the full daily series for plotting / further analysis
    res_corr.to_csv(
        OUTPUT_TABLES / "delta_normal_correlation_sensitivity_daily.csv"
    )
    print(f"  Saved -> {path}")
    return summary, res_corr


if __name__ == "__main__":
    OUTPUT_FIGS.mkdir(parents=True, exist_ok=True)
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Delta-Normal VaR  |  alpha={ALPHA:.0%}  |  window={WINDOW} days")
    print(f"Scope : linear book (SPY, IEF, GLD, EURUSD)")
    print(f"Backtest target : pnl_total (all instruments)")
    print(f"{'='*60}")

    log_ret, pnl = load_data()

    # Step 2: Rolling VaR + ES
    results = compute_delta_normal_var(log_ret)

    # Step 3: Backtest
    bt = backtest_delta_normal(pnl, results)

    # Step 4: Plots
    plot_var_results(pnl, results, bt)
    plot_all(bt, pnl=pnl, var=results["VaR_DN_full"], save=True)

    # Step 5: Save
    save_results(pnl, results, bt)

    # Step 6: Crisis-period backtesting
    crisis_df = run_crisis_backtests(pnl, results)

    # Step 7: Window sensitivity
    sens_df = window_sensitivity(log_ret, pnl)

    # Step 8: EWMA covariance sensitivity
    ewma_df = ewma_sensitivity(log_ret, pnl)

    # Step 9: Delta-t-Normal sensitivity
    t_df, nu_fit = t_normal_sensitivity(log_ret, pnl)

    # Step 10: Correlation structure sensitivity
    corr_df, corr_daily = correlation_sensitivity(log_ret, pnl)

    print(f"\n{'='*60}")
    print(f"Delta-Normal VaR complete!")
    print(f"  Exceptions  : {bt.N}  (expected {bt.expected_N:.1f}  at 99% VaR)")
    print(f"  Kupiec H0   : {'NOT rejected' if not bt.reject_uc else 'REJECTED'}  "
          f"(p={bt.pvalue_uc:.4f})")
    print(f"  Mean ES     : ${results['ES_DN_full'].mean():,.0f}")
    print(f"  t-dist ν    : {nu_fit:.2f}  (used in Delta-t-Normal sensitivity)")
    print(f"  NOTE: exception excess expected — normality failure + linear scope only")
    print(f"  Outputs:")
    print(f"    data/processed/var_delta_normal.csv")
    print(f"    outputs/tables/backtest_delta_normal.csv")
    print(f"    outputs/tables/delta_normal_crisis_backtest.csv")
    print(f"    outputs/tables/delta_normal_window_sensitivity.csv")
    print(f"    outputs/tables/delta_normal_ewma_sensitivity.csv")
    print(f"    outputs/tables/delta_normal_t_sensitivity.csv")
    print(f"    outputs/tables/delta_normal_correlation_sensitivity.csv")
    print(f"    outputs/tables/delta_normal_correlation_sensitivity_daily.csv")
    print(f"    outputs/figures/dn_01_var_results.png")
    print(f"    outputs/figures/11_exceptions_timeline_Delta-Normal.png")
    print(f"    outputs/figures/14_transition_matrix_Delta-Normal.png")
    print(f"{'='*60}")