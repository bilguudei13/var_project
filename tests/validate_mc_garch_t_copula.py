"""
validate_mc_garch_copula_v2_audit.py
=====================================
Model assumption validation for monte_carlo_garch_copula_v2_audit.py.

Covers nine blocks that mirror the model pipeline and lecture requirements:

  Block 1  P&L / risk mapping      — simulated P&L distribution vs actual P&L
  Block 2  GARCH assumptions       — standardised residual diagnostics per factor
  Block 3  Copula assumptions      — correlation structure, tail dependence, GoF
  Block 4  Simulation quality      — simulated vs historical factor distributions
  Block 5  GARCH parameter stability — ω, α, β, ν evolution across rolling refits
  Block 6  Asymmetry / sign-bias   — Engle-Ng (1993) test per factor
  Block 7  Component P&L           — linear / IRS / straddle decomposition
  Block 8  Stress scenarios        — worst historical losses vs MC VaR
  Block 9  Tail-specific           — extreme quantiles, GPD ξ, Hill plot

All functions imported directly from v2_audit (no duplication):
  fit_garch_marginals, compute_pseudo_observations, fit_t_copula,
  simulate_t_copula, copula_gof_cvm, scenarios_to_pnl, load_data,
  price_straddle_vec, price_irs_vec.

Outputs
-------
Figures  → outputs/figures/validate_mc_garch_t_copula_NN_*.png  (NN = 01 … 11)
Tables   → outputs/tables/validate_mc_garch_t_copula_*.csv
Console  → printed summary tables for each block
"""

import sys
import warnings
import textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from pathlib import Path
from scipy import stats
from scipy.stats import t as scipy_t

# ── Paths ────────────────────────────────────────────────────────────────────
_ROOT       = Path(__file__).resolve().parent.parent   # var_project/
_HERE       = Path(__file__).resolve().parent
FIG_DIR     = _ROOT / "outputs" / "figures"
TAB_DIR     = _ROOT / "outputs" / "tables"
DATA_DIR    = _ROOT / "data" / "processed"
RAW_DIR     = _ROOT / "data" / "raw"

for p in (_ROOT / "src" / "var_methods", _ROOT / "src" / "data", _ROOT, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# ── Import v2 (reuses all its fitted functions without duplication) ─────────
import mc_garch_t_copula as v2

# Statsmodels — optional; degrade gracefully if missing
try:
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.stattools import acf as sm_acf
    _HAS_SM = True
except ImportError:  # pragma: no cover
    _HAS_SM = False
    warnings.warn("statsmodels not found — Ljung-Box tests will be skipped.")

# ── Constants ────────────────────────────────────────────────────────────────
LB_LAGS    = [5, 10, 20]       # Ljung-Box lags to report
TAIL_ALPHA = 0.05              # joint-tail threshold for copula validation
N_SIM_VAL  = 10_000            # MC scenarios for validation simulation
SEED_VAL   = 99                # separate seed so it differs from v2's SEED=42

# ── Plot style ────────────────────────────────────────────────────────────────
_STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.edgecolor":   "#cccccc",
    "axes.linewidth":   0.8,
    "axes.grid":        True,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "grid.color":       "#ebebeb",
    "grid.linewidth":   0.6,
    "grid.linestyle":   "--",
    "font.size":        9,
    "axes.titlesize":   10,
    "axes.titleweight": "bold",
    "axes.titlepad":    8,
    "axes.labelsize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  8,
    "legend.framealpha":0.9,
    "figure.dpi":       150,
}

_C = {
    "hist":    "#2E86AB",
    "sim":     "#E07B39",
    "actual":  "#2E86AB",
    "ref":     "#C0392B",
    "pass":    "#27AE60",
    "fail":    "#C0392B",
    "neutral": "#7F8C8D",
}


def _apply_style():
    plt.rcParams.update(_STYLE)


def _savefig(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure → {path.relative_to(_ROOT)}")


def _savetab(df, name):
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    path = TAB_DIR / name
    df.to_csv(path)
    print(f"  Table  → {path.relative_to(_ROOT)}")


# =============================================================================
# DATA SETUP  — load, fit final window, run one MC simulation
# =============================================================================

def setup(n_sim=N_SIM_VAL, seed=SEED_VAL):
    """
    Load all data, fit GARCH + copula on the final WINDOW, run one MC
    simulation for the final day, and return a dict of everything needed
    by the validation blocks.
    """
    from config import V0, WEIGHTS_DICT, IRS_NOTIONAL, IRS_FIXED_RATE, RF_RATE

    print("=" * 65)
    print("SETUP — loading data and fitting final-window model")
    print("=" * 65)

    factors, pnl_series, prices, vix_series, dgs10_series = v2.load_data()
    returns_arr = factors.values
    dates       = factors.index
    N           = len(returns_arr)

    # ── Final window ─────────────────────────────────────────────────────────
    W      = returns_arr[N - v2.WINDOW : N]
    W_date = dates[N - v2.WINDOW : N]

    print(f"\nFitting GARCH(1,1)-t on final {v2.WINDOW}-day window …")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        std_resids, garch_params = v2.fit_garch_marginals(W)

    nu_marginals  = np.array([p["nu"]         for p in garch_params])
    sigma_last    = np.array([p["sigma_last"]  for p in garch_params])
    mu_arr        = np.array([p["mu"]          for p in garch_params])
    omega_arr     = np.array([p["omega"]       for p in garch_params])
    alpha_arr     = np.array([p["alpha"]       for p in garch_params])
    beta_arr      = np.array([p["beta"]        for p in garch_params])

    # One-step-ahead sigma forecast using the last return in W
    x_last        = W[-1] - mu_arr
    sigma_sq_next = omega_arr + alpha_arr * x_last**2 + beta_arr * sigma_last**2
    sigma_forecast = np.sqrt(np.maximum(sigma_sq_next, 1e-12))

    print("Fitting t-copula on GARCH residuals (v2 convention) …")
    U_resid = v2.compute_pseudo_observations(std_resids)
    R_static, nu_copula, _ = v2.fit_t_copula(U_resid)

    # ── Final-day instrument state ────────────────────────────────────────────
    LINEAR_TICKERS = ["EURUSD", "GLD", "IEF", "SPY"]
    prices_linear  = prices.rename(columns={"EURUSD=X": "EURUSD"})[LINEAR_TICKERS]
    prices_linear  = prices_linear.reindex(dates, method="ffill")
    weights        = pd.Series(
        {("EURUSD" if k == "EURUSD=X" else k): v_ for k, v_ in WEIGHTS_DICT.items()}
    )[LINEAR_TICKERS]
    linear_shares  = (V0 * weights / prices_linear.iloc[0]).values
    linear_prices_now = prices_linear.iloc[-1].values

    S_now     = float(prices["SPY"].iloc[-1])
    sigma_now = float(vix_series.iloc[-1]) / 100.0
    rate_now  = float(dgs10_series.iloc[-1]) / 100.0
    K_atm     = S_now                            # ATM strike for validation
    T_now     = 30.0 / 252.0                     # ~30 trading days remaining

    # ── MC simulation ─────────────────────────────────────────────────────────
    print(f"Running MC simulation (M={n_sim:,} scenarios, seed={seed}) …")
    rng = np.random.default_rng(seed)
    pnl_sim = v2.scenarios_to_pnl(
        nu_marginals, sigma_forecast, mu_arr,
        R_static, nu_copula, rng,
        linear_prices_now, linear_shares,
        S_now, sigma_now, rate_now,
        K_atm, T_now,
    )

    # Also collect raw simulated factor returns (for Block 4)
    rng2     = np.random.default_rng(seed + 1)
    U_sim    = v2.simulate_t_copula(R_static, nu_copula, n_sim, rng2)
    eps      = 1e-6
    std_sc   = np.sqrt((nu_marginals - 2.0) / nu_marginals)
    r_sim    = mu_arr[np.newaxis, :] + sigma_forecast[np.newaxis, :] * np.column_stack([
        scipy_t.ppf(np.clip(U_sim[:, j], eps, 1 - eps), df=nu_marginals[j]) * std_sc[j]
        for j in range(v2.N_ASSETS)
    ])

    # ── Load saved VaR and realised P&L ──────────────────────────────────────
    var_path = DATA_DIR / "var_mc_garch_t_copula.csv"
    var_series = None
    if var_path.exists():
        var_df     = pd.read_csv(var_path, index_col=0, parse_dates=True).squeeze()
        var_series = var_df
    else:
        print(f"  WARNING: {var_path.name} not found — run v2 first to generate it.")

    pnl_bt    = pnl_series.iloc[v2.WINDOW:]
    dates_bt  = dates[v2.WINDOW:]

    print("Setup complete.\n")

    return {
        "W"               : W,
        "W_dates"         : W_date,
        "returns_arr"     : returns_arr,
        "dates"           : dates,
        "std_resids"      : std_resids,
        "garch_params"    : garch_params,
        "nu_marginals"    : nu_marginals,
        "sigma_forecast"  : sigma_forecast,
        "R_static"        : R_static,
        "nu_copula"       : nu_copula,
        "pnl_sim"         : pnl_sim,
        "r_sim"           : r_sim,
        "pnl_bt"          : pnl_bt,
        "dates_bt"        : dates_bt,
        "var_series"      : var_series,
        "linear_shares"   : linear_shares,
        "linear_prices_now": linear_prices_now,
        "S_now"           : S_now,
        "sigma_now"       : sigma_now,
        "rate_now"        : rate_now,
        "mu_arr"          : mu_arr,
        "K_atm"           : K_atm,
        "T_now"           : T_now,
        "prices"          : prices,
        "vix_series"      : vix_series,
        "dgs10_series"    : dgs10_series,
    }


# =============================================================================
# BLOCK 1 — P&L / RISK MAPPING VALIDATION
# =============================================================================

def validate_pnl_mapping(d):
    """
    Compare simulated P&L distribution (one MC run, final window) to realised
    P&L over the backtest period.

    Tests whether the model's assumed P&L distribution is consistent with
    the empirical distribution of daily portfolio outcomes.

    Produces:
      validate_mc_garch_t_copula_01_pnl_distribution.png  —  3-panel: histogram, CDF, tail detail
      validate_mc_garch_t_copula_pnl_stats.csv            —  moment + quantile comparison table
    """
    print("=" * 65)
    print("BLOCK 1 — P&L / Risk Mapping Validation")
    print("=" * 65)

    pnl_sim   = d["pnl_sim"]
    pnl_bt    = d["pnl_bt"]
    var_series = d["var_series"]

    # ── Stats comparison ─────────────────────────────────────────────────────
    def _moments(x, label):
        q = [0.005, 0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 0.995]
        row = {
            "source"   : label,
            "mean"     : float(np.mean(x)),
            "std"      : float(np.std(x)),
            "skew"     : float(stats.skew(x)),
            "kurtosis" : float(stats.kurtosis(x)),
        }
        for qi in q:
            row[f"q{qi:.1%}"] = float(np.percentile(x, qi * 100))
        return row

    rows = [
        _moments(pnl_sim,        "Simulated (MC, final window)"),
        _moments(pnl_bt.values,  "Realised P&L (full backtest)"),
        _moments(pnl_bt.values[-v2.WINDOW:], "Realised P&L (last 500 days)"),
    ]
    stats_df = pd.DataFrame(rows).set_index("source")

    print("\n  P&L Summary Statistics")
    print(f"  {'─'*60}")
    print(stats_df[["mean", "std", "skew", "kurtosis",
                     "q1.0%", "q5.0%"]].to_string())

    # VaR coverage check using saved series
    if var_series is not None:
        common    = pnl_bt.index.intersection(var_series.index)
        losses    = -pnl_bt.loc[common]
        var_aligned = var_series.loc[common]
        exc_rate  = float((losses > var_aligned).mean())
        print(f"\n  VaR coverage check (saved series)")
        print(f"    Expected breach rate : {1 - v2.ALPHA:.2%}")
        print(f"    Observed breach rate : {exc_rate:.4%}")

    _savetab(stats_df, "validate_mc_garch_t_copula_pnl_stats.csv")

    # ── Figure ───────────────────────────────────────────────────────────────
    _apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(
        "Block 1 — P&L Distribution: Simulated vs Realised",
        fontsize=11, fontweight="bold", x=0.02, ha="left"
    )

    actual = pnl_bt.values[-v2.WINDOW:]   # match window length
    lims   = np.percentile(np.concatenate([pnl_sim, actual]), [0.5, 99.5])
    bins   = np.linspace(lims[0], lims[1], 60)

    # Panel A: histogram
    ax = axes[0]
    ax.hist(actual,   bins=bins, density=True, alpha=0.55,
            color=_C["actual"],  label="Realised (last 500d)")
    ax.hist(pnl_sim,  bins=bins, density=True, alpha=0.55,
            color=_C["sim"],     label="Simulated (MC)")
    ax.axvline(np.percentile(pnl_sim,  1), color=_C["sim"],
               linestyle="--", linewidth=1.2, label="VaR sim (1%)")
    ax.axvline(np.percentile(actual,   1), color=_C["actual"],
               linestyle=":",  linewidth=1.2, label="VaR actual (1%)")
    ax.axvline(0, color="#aaaaaa", linewidth=0.7)
    ax.set_xlabel("Daily P&L (USD)")
    ax.set_ylabel("Density")
    ax.set_title("A — Histogram overlay")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1e"))
    ax.legend(fontsize=7.5)

    # Panel B: empirical CDF
    ax = axes[1]
    for arr, lbl, col in [(actual, "Realised", _C["actual"]),
                           (pnl_sim, "Simulated", _C["sim"])]:
        xs   = np.sort(arr)
        probs = np.arange(1, len(xs) + 1) / len(xs)
        ax.plot(xs, probs, color=col, linewidth=1.2, label=lbl)
    ax.axvline(0, color="#aaaaaa", linewidth=0.7)
    ax.set_xlabel("Daily P&L (USD)")
    ax.set_ylabel("CDF")
    ax.set_title("B — Empirical CDF")
    ax.legend()

    # Panel C: left-tail detail (losses)
    ax = axes[2]
    tail_qs = np.linspace(0.001, 0.10, 50)
    q_sim  = np.percentile(pnl_sim,  tail_qs * 100)
    q_act  = np.percentile(actual,   tail_qs * 100)
    ax.plot(tail_qs * 100, q_sim,  color=_C["sim"],    linewidth=1.5,
            label="Simulated")
    ax.plot(tail_qs * 100, q_act,  color=_C["actual"], linewidth=1.5,
            label="Realised")
    ax.plot(tail_qs * 100, tail_qs * 100 / tail_qs * q_sim,
            color="#cccccc", linewidth=0, alpha=0)   # placeholder for alignment
    ax.fill_between(tail_qs * 100, q_sim, q_act,
                    alpha=0.15, color=_C["neutral"], label="Gap")
    ax.axhline(0, color="#aaaaaa", linewidth=0.7)
    ax.set_xlabel("Quantile (%)")
    ax.set_ylabel("P&L (USD)")
    ax.set_title("C — Left-tail quantile comparison")
    ax.legend()

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _savefig(fig, "validate_mc_garch_t_copula_01_pnl_distribution.png")


# =============================================================================
# BLOCK 2 — GARCH RESIDUAL VALIDATION
# =============================================================================

def validate_garch_residuals(d):
    """
    Validate GARCH(1,1)-t model assumptions on the final WINDOW residuals.

    Tests:
    - Standardised residuals should have mean≈0, std≈1, t(ν)-distributed
    - Ljung-Box on residuals: no remaining autocorrelation (H0 should hold)
    - Ljung-Box on squared residuals: no remaining ARCH effects (H0 should hold)
    - QQ plots: residuals vs t(ν_j)

    Produces:
      validate_mc_garch_t_copula_02_garch_qq.png         — 6-panel QQ plot
      validate_mc_garch_t_copula_03_garch_acf_sq.png     — 6-panel ACF of squared residuals
      validate_mc_garch_t_copula_garch_stats.csv         — summary statistics per factor
      validate_mc_garch_t_copula_garch_ljungbox.csv      — Ljung-Box p-values at lags 5, 10, 20
    """
    print("=" * 65)
    print("BLOCK 2 — GARCH Residual Validation")
    print("=" * 65)

    std_resids   = d["std_resids"]
    garch_params = d["garch_params"]
    nu_marginals = d["nu_marginals"]
    instruments  = v2.INSTRUMENTS

    # ── Summary statistics ────────────────────────────────────────────────────
    rows = []
    for j, instr in enumerate(instruments):
        z = std_resids[:, j]
        rows.append({
            "factor"   : instr,
            "mean"     : float(np.mean(z)),
            "std"      : float(np.std(z, ddof=1)),
            "skewness" : float(stats.skew(z)),
            "excess_kurt": float(stats.kurtosis(z)),
            "min"      : float(np.min(z)),
            "max"      : float(np.max(z)),
            "ν_GARCH"  : float(nu_marginals[j]),
        })
    stat_df = pd.DataFrame(rows).set_index("factor")

    print("\n  GARCH Standardised Residuals — Summary Statistics")
    print(f"  (mean≈0, std≈1 expected;  excess kurtosis > 0 for t-distributed)")
    print(f"  {'─'*60}")
    print(stat_df.to_string(float_format="{:.4f}".format))
    _savetab(stat_df, "validate_mc_garch_t_copula_garch_stats.csv")

    # ── Ljung-Box test ────────────────────────────────────────────────────────
    lb_rows = []
    if _HAS_SM:
        for j, instr in enumerate(instruments):
            z  = std_resids[:, j]
            z2 = z ** 2
            row = {"factor": instr}
            for lag in LB_LAGS:
                lb_z  = acorr_ljungbox(z,  lags=[lag], return_df=True)
                lb_z2 = acorr_ljungbox(z2, lags=[lag], return_df=True)
                row[f"p_resid_lag{lag}"]    = float(lb_z["lb_pvalue"].iloc[0])
                row[f"p_sq_resid_lag{lag}"] = float(lb_z2["lb_pvalue"].iloc[0])
            lb_rows.append(row)

        lb_df = pd.DataFrame(lb_rows).set_index("factor")
        print(f"\n  Ljung-Box p-values  (H0: no autocorrelation; p > 0.05 = PASS)")
        print(f"  {'─'*60}")
        print(lb_df.to_string(float_format="{:.4f}".format))
        print()
        for j, instr in enumerate(instruments):
            for lag in LB_LAGS:
                pv_z  = lb_df.loc[instr, f"p_resid_lag{lag}"]
                pv_z2 = lb_df.loc[instr, f"p_sq_resid_lag{lag}"]
                r_z   = "FAIL" if pv_z  < 0.05 else "pass"
                r_z2  = "FAIL" if pv_z2 < 0.05 else "pass"
                print(f"  {instr:<12} lag={lag:>2}  "
                      f"resid p={pv_z:.3f} [{r_z}]   "
                      f"sq_resid p={pv_z2:.3f} [{r_z2}]")
        _savetab(lb_df, "validate_mc_garch_t_copula_garch_ljungbox.csv")
    else:
        print("  Ljung-Box skipped (statsmodels not installed).")

    # ── Figure: QQ plots ─────────────────────────────────────────────────────
    _apply_style()
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    fig.suptitle(
        "Block 2 — GARCH QQ Plots: Standardised Residuals vs t(ν)",
        fontsize=11, fontweight="bold", x=0.02, ha="left"
    )
    for j, (ax, instr) in enumerate(zip(axes.flat, instruments)):
        z  = std_resids[:, j]
        nu = nu_marginals[j]
        n  = len(z)

        probs = (np.arange(1, n + 1) - 0.5) / n
        theor = scipy_t.ppf(probs, df=nu)
        empir = np.sort(z)

        ax.scatter(theor, empir, s=4, alpha=0.45, color=_C["hist"])
        qlim = max(abs(theor[2]), abs(theor[-3]))
        ax.plot([-qlim, qlim], [-qlim, qlim],
                color=_C["ref"], linewidth=1.3, linestyle="--", label="y = x")
        ax.set_title(f"{instr}  (ν={nu:.1f})")
        ax.set_xlabel("t(ν) quantiles")
        ax.set_ylabel("Empirical quantiles")
        ax.legend(fontsize=7)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _savefig(fig, "validate_mc_garch_t_copula_02_garch_qq.png")

    # ── Figure: ACF of squared residuals ─────────────────────────────────────
    _apply_style()
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    fig.suptitle(
        "Block 2 — ACF of Squared GARCH Residuals  "
        "(bars should lie inside confidence band if ARCH effects removed)",
        fontsize=10, fontweight="bold", x=0.02, ha="left"
    )
    max_lags = 30
    conf_lev = 1.96 / np.sqrt(len(std_resids))
    for j, (ax, instr) in enumerate(zip(axes.flat, instruments)):
        z2 = std_resids[:, j] ** 2
        if _HAS_SM:
            acf_vals = sm_acf(z2, nlags=max_lags, fft=True)
        else:
            acf_vals = np.array([
                float(np.corrcoef(z2[:-k], z2[k:])[0, 1]) if k > 0 else 1.0
                for k in range(max_lags + 1)
            ])
        lags = np.arange(1, max_lags + 1)
        colors = [_C["fail"] if abs(v) > conf_lev else _C["pass"]
                  for v in acf_vals[1:]]
        ax.bar(lags, acf_vals[1:], color=colors, width=0.7, alpha=0.8)
        ax.axhline( conf_lev, color="#555555", linestyle="--",
                    linewidth=0.9, label=f"±{conf_lev:.3f} (95%)")
        ax.axhline(-conf_lev, color="#555555", linestyle="--", linewidth=0.9)
        ax.axhline(0, color="#aaaaaa", linewidth=0.5)
        ax.set_title(f"{instr}")
        ax.set_xlabel("Lag")
        ax.set_ylabel("ACF(z²)")
        ax.set_xlim(0, max_lags + 1)
        ax.legend(fontsize=7)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _savefig(fig, "validate_mc_garch_t_copula_03_garch_acf_sq.png")


# =============================================================================
# BLOCK 3 — COPULA VALIDATION
# =============================================================================

def validate_copula(d):
    """
    Validate the fitted t-copula.

    Tests:
    - Empirical vs model correlation matrix (Pearson on t^{-1}(U))
    - Joint lower-tail probabilities P(U_i ≤ α AND U_j ≤ α) at α=5%
    - Cramér-von Mises GoF test (reuses v2.copula_gof_cvm)

    Produces:
      validate_mc_garch_t_copula_04_copula_corr.png     — 3-panel: empirical / model / difference
      validate_mc_garch_t_copula_05_copula_tail.png     — 2-panel heatmap: empirical vs model tail probs
      validate_mc_garch_t_copula_copula_corr.csv        — correlation comparison per pair
      validate_mc_garch_t_copula_copula_tail.csv        — joint tail probability matrices
    """
    print("=" * 65)
    print("BLOCK 3 — Copula Validation")
    print("=" * 65)

    std_resids = d["std_resids"]
    R_static   = d["R_static"]
    nu_copula  = d["nu_copula"]
    instruments = v2.INSTRUMENTS
    d_n        = v2.N_ASSETS

    # ── Empirical correlation from pseudo-observations ─────────────────────
    U_resid  = v2.compute_pseudo_observations(std_resids)
    eps      = 1e-6
    T_emp    = scipy_t.ppf(np.clip(U_resid, eps, 1 - eps), df=nu_copula)
    R_emp    = np.corrcoef(T_emp.T)

    # ── Correlation comparison table ───────────────────────────────────────
    pair_rows = []
    for i in range(d_n):
        for j in range(i + 1, d_n):
            pair_rows.append({
                "pair"      : f"{instruments[i]} / {instruments[j]}",
                "empirical" : float(R_emp[i, j]),
                "model"     : float(R_static[i, j]),
                "diff"      : float(R_emp[i, j] - R_static[i, j]),
            })
    corr_df = pd.DataFrame(pair_rows).set_index("pair")
    print(f"\n  Copula Correlation Comparison  (ν_copula = {nu_copula})")
    print(f"  {'─'*55}")
    print(corr_df.to_string(float_format="{:.4f}".format))
    _savetab(corr_df, "validate_mc_garch_t_copula_copula_corr.csv")

    # ── Joint tail probability ─────────────────────────────────────────────
    # Empirical: P(U_i ≤ α AND U_j ≤ α) from pseudo-observations
    def _joint_tail_empirical(U, alpha):
        m = np.zeros((d_n, d_n))
        for i in range(d_n):
            for j in range(d_n):
                m[i, j] = np.mean((U[:, i] <= alpha) & (U[:, j] <= alpha))
        return m

    # Model: simulate from fitted t-copula
    rng_tail = np.random.default_rng(SEED_VAL + 10)
    U_model  = v2.simulate_t_copula(R_static, nu_copula, 50_000, rng_tail)

    joint_emp   = _joint_tail_empirical(U_resid, TAIL_ALPHA)
    joint_model = _joint_tail_empirical(U_model, TAIL_ALPHA)
    joint_diff  = joint_emp - joint_model

    # Diagonal = marginal exceedance rate (should be ≈ TAIL_ALPHA)
    print(f"\n  Joint Lower-Tail Probabilities  P(U_i ≤ {TAIL_ALPHA:.0%}, U_j ≤ {TAIL_ALPHA:.0%})")
    print(f"  (diagonal = marginal rate ≈ {TAIL_ALPHA:.0%}; "
          f"off-diagonal shows co-crash probability)")
    print(f"  {'─'*55}")
    header = f"  {'':>12}" + "".join(f"  {s:>8}" for s in instruments)
    print(header)
    for i, si in enumerate(instruments):
        row_str = f"  {si:>12}"
        for j in range(d_n):
            row_str += f"  {joint_emp[i,j]:8.4f}"
        print(row_str + f"  (emp)")
    print()
    for i, si in enumerate(instruments):
        row_str = f"  {si:>12}"
        for j in range(d_n):
            row_str += f"  {joint_model[i,j]:8.4f}"
        print(row_str + f"  (model)")

    # Save flat tables
    tail_flat = []
    for i in range(d_n):
        for j in range(d_n):
            tail_flat.append({
                "factor_i": instruments[i],
                "factor_j": instruments[j],
                "emp"     : float(joint_emp[i, j]),
                "model"   : float(joint_model[i, j]),
                "diff"    : float(joint_diff[i, j]),
            })
    _savetab(pd.DataFrame(tail_flat), "validate_mc_garch_t_copula_copula_tail.csv")

    # ── CvM GoF test ──────────────────────────────────────────────────────
    print(f"\n  Cramér-von Mises GoF test (n_mc=200 bootstrap)")
    print(f"  H0: data consistent with fitted t-copula(R, ν={nu_copula})")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cvm_stat, cvm_p = v2.copula_gof_cvm(
            U_resid, R_static, nu_copula, n_mc=200, rng_seed=SEED_VAL
        )
    verdict = "FAIL TO REJECT H0 ✓" if cvm_p >= 0.05 else "REJECT H0 ✗"
    print(f"  CvM S = {cvm_stat:.5f}   p-value = {cvm_p:.4f}   {verdict}")

    # ── Figure: correlation matrices ───────────────────────────────────────
    _apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle(
        f"Block 3 — Copula Correlation Matrices  (ν={nu_copula})",
        fontsize=11, fontweight="bold", x=0.02, ha="left"
    )
    panels = [
        (R_emp,         "Empirical (t-space)", "Blues",    -1, 1),
        (R_static,      "Model (fitted)",       "Blues",    -1, 1),
        (R_emp - R_static, "Difference",        "RdBu_r", -0.2, 0.2),
    ]
    for ax, (mat, title, cmap, vmin, vmax) in zip(axes, panels):
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(d_n))
        ax.set_yticks(range(d_n))
        ax.set_xticklabels(instruments, rotation=40, ha="right", fontsize=7.5)
        ax.set_yticklabels(instruments, fontsize=7.5)
        ax.set_title(title)
        for i in range(d_n):
            for j in range(d_n):
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                        fontsize=6.5,
                        color="white" if abs(mat[i,j]) > 0.6 * vmax else "#333")

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _savefig(fig, "validate_mc_garch_t_copula_04_copula_corr.png")

    # ── Figure: joint tail heatmaps ────────────────────────────────────────
    _apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle(
        f"Block 3 — Joint Lower-Tail Probabilities  "
        f"P(U_i ≤ {TAIL_ALPHA:.0%}, U_j ≤ {TAIL_ALPHA:.0%})",
        fontsize=11, fontweight="bold", x=0.02, ha="left"
    )
    vmax_tail = max(joint_emp.max(), joint_model.max()) * 1.05
    diff_bound = max(abs(joint_diff).max(), 0.005)
    tail_panels = [
        (joint_emp,   "Empirical",       "Blues",  0, vmax_tail),
        (joint_model, "t-Copula model",  "Blues",  0, vmax_tail),
        (joint_diff,  "Diff (emp−model)","RdBu_r", -diff_bound, diff_bound),
    ]
    for ax, (mat, title, cmap, vmin, vmax) in zip(axes, tail_panels):
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(d_n))
        ax.set_yticks(range(d_n))
        ax.set_xticklabels(instruments, rotation=40, ha="right", fontsize=7.5)
        ax.set_yticklabels(instruments, fontsize=7.5)
        ax.set_title(title)
        for i in range(d_n):
            for j in range(d_n):
                ax.text(j, i, f"{mat[i,j]:.3f}", ha="center", va="center",
                        fontsize=6.5,
                        color="white" if mat[i,j] > 0.6 * vmax else "#333")

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _savefig(fig, "validate_mc_garch_t_copula_05_copula_tail.png")


# =============================================================================
# BLOCK 4 — SIMULATION VALIDATION
# =============================================================================

def validate_simulation(d):
    """
    Compare simulated factor return distributions to the historical distributions
    in the final WINDOW.

    Tests:
    - Visual: histogram overlay (simulated vs historical) per factor
    - KS two-sample test: H0 same distribution
    - Moment comparison: mean, std, skew, kurtosis, tail quantiles

    Produces:
      validate_mc_garch_t_copula_06_sim_distributions.png  — 6-panel histogram overlay
      validate_mc_garch_t_copula_sim_stats.csv             — moment + quantile + KS comparison
    """
    print("=" * 65)
    print("BLOCK 4 — Simulation Distribution Validation")
    print("=" * 65)

    r_sim    = d["r_sim"]            # (N_SIM_VAL, N_ASSETS) simulated returns
    W        = d["W"]                # (WINDOW, N_ASSETS) historical returns
    instruments = v2.INSTRUMENTS
    nu_marg  = d["nu_marginals"]
    sig_fc   = d["sigma_forecast"]

    # ── Moment + KS comparison ────────────────────────────────────────────
    sim_rows = []
    hist_rows = []
    for j, instr in enumerate(instruments):
        rs  = r_sim[:, j]
        rh  = W[:, j]
        ks_stat, ks_p = stats.ks_2samp(rs, rh)

        def _row(x, src, j):
            return {
                "factor"        : instr,
                "source"        : src,
                "mean"          : float(np.mean(x)),
                "std"           : float(np.std(x)),
                "skewness"      : float(stats.skew(x)),
                "excess_kurt"   : float(stats.kurtosis(x)),
                "q0.5%"         : float(np.percentile(x, 0.5)),
                "q1%"           : float(np.percentile(x, 1.0)),
                "q5%"           : float(np.percentile(x, 5.0)),
                "q95%"          : float(np.percentile(x, 95.0)),
                "q99%"          : float(np.percentile(x, 99.0)),
                "ks_stat"       : ks_stat,
                "ks_pvalue"     : ks_p,
                "ks_verdict"    : "FAIL" if ks_p < 0.05 else "pass",
            }
        sim_rows.append(_row(rs, "simulated", j))
        hist_rows.append(_row(rh, "historical", j))

    sim_df  = pd.DataFrame(sim_rows).set_index(["factor", "source"])
    hist_df = pd.DataFrame(hist_rows).set_index(["factor", "source"])
    all_df  = pd.concat([sim_df, hist_df]).sort_index()

    print("\n  Simulated vs Historical Factor Returns — Key Moments")
    print(f"  (KS test H0: same distribution)")
    print(f"  {'─'*65}")
    display_cols = ["mean", "std", "skewness", "excess_kurt",
                    "q1%", "q99%", "ks_stat", "ks_pvalue", "ks_verdict"]
    print(all_df[display_cols].to_string(float_format="{:.5f}".format))
    _savetab(all_df, "validate_mc_garch_t_copula_sim_stats.csv")

    # ── Figure: histogram overlays ────────────────────────────────────────
    _apply_style()
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(
        "Block 4 — Simulated vs Historical Factor Return Distributions",
        fontsize=11, fontweight="bold", x=0.02, ha="left"
    )
    for j, (ax, instr) in enumerate(zip(axes.flat, instruments)):
        rs   = r_sim[:, j]
        rh   = W[:, j]
        nu   = nu_marg[j]
        sig  = sig_fc[j]
        ks_stat, ks_p = stats.ks_2samp(rs, rh)

        # Use shared bin range
        lo, hi = np.percentile(np.concatenate([rs, rh]), [0.5, 99.5])
        bins   = np.linspace(lo, hi, 55)

        ax.hist(rh, bins=bins, density=True, alpha=0.55,
                color=_C["hist"], label="Historical")
        ax.hist(rs, bins=bins, density=True, alpha=0.45,
                color=_C["sim"],  label="Simulated")

        # Theoretical t(ν) density scaled by sigma_forecast
        xr   = np.linspace(lo, hi, 300)
        pdf  = scipy_t.pdf(xr / sig, df=nu) / sig
        ax.plot(xr, pdf, color=_C["ref"], linewidth=1.3, linestyle="--",
                label=f"t(ν={nu:.1f})×σ")

        ks_col = _C["fail"] if ks_p < 0.05 else _C["pass"]
        ax.set_title(f"{instr}   KS p={ks_p:.3f}", color=ks_col)
        ax.set_xlabel("Log-return")
        ax.set_ylabel("Density")
        if j == 0:
            ax.legend(fontsize=7)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _savefig(fig, "validate_mc_garch_t_copula_06_sim_distributions.png")


# =============================================================================
# HELPERS FOR BLOCKS 5-9
# =============================================================================

def _sign_bias_ols(z):
    """
    Engle-Ng (1993) sign-bias test on standardised residuals z.

    Regresses z²_t on [1, S⁻_{t-1}, S⁻_{t-1}·z_{t-1}, S⁺_{t-1}·z_{t-1}]
    where S⁻ = I(z < 0).  Returns dict with beta, t-stats, p-values, joint F.
    """
    T   = len(z)
    y   = z[1:] ** 2
    s_n = (z[:-1] < 0).astype(float)
    X   = np.column_stack([
        np.ones(T - 1),
        s_n,
        s_n * z[:-1],
        (1.0 - s_n) * z[:-1],
    ])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    e      = y - X @ beta
    n, k   = X.shape
    s2     = np.sum(e ** 2) / (n - k)
    XtXinv = np.linalg.pinv(X.T @ X)
    se     = np.sqrt(np.maximum(np.diag(s2 * XtXinv), 0.0))
    t_st   = beta / np.where(se > 1e-12, se, 1e-12)
    p_t    = 2.0 * stats.t.sf(np.abs(t_st), df=n - k)
    # Joint F-test on columns 1, 2, 3 (the three sign / size terms)
    R_mat  = np.eye(k)[1:, :]
    Rb     = R_mat @ beta
    cov_R  = s2 * R_mat @ XtXinv @ R_mat.T
    try:
        F_stat = float(Rb @ np.linalg.solve(cov_R, Rb) / 3.0)
    except np.linalg.LinAlgError:
        F_stat = np.nan
    p_F = (float(stats.f.sf(F_stat, dfn=3, dfd=n - k))
           if np.isfinite(F_stat) else np.nan)
    return {"beta": beta, "t_stats": t_st, "p_vals": p_t,
            "F_stat": F_stat, "p_F": p_F, "n": n}


def _hill_estimator(losses, k_max=None):
    """
    Hill estimator: α_H(k) = k / Σ_{i=1}^{k} log(x_{(i)}/x_{(k+1)}).
    losses: positive loss values (right tail).
    Returns (k_vals, alpha_H_vals).
    """
    x = np.sort(losses)[::-1]
    n = len(x)
    if k_max is None:
        k_max = min(n // 2, 150)
    k_vals  = np.arange(2, k_max + 1)
    alpha_H = np.array([
        k / np.sum(np.log(x[:k] / (x[k] + 1e-12)))
        for k in k_vals
    ])
    return k_vals, alpha_H


def _fit_gpd_tail(losses, threshold_pct=0.90):
    """
    Fit GPD to exceedances above the threshold_pct quantile of losses.
    Returns dict {xi, scale, threshold, n_exceed} or None if too few points.
    """
    try:
        from scipy.stats import genpareto
    except ImportError:
        return None
    u  = np.percentile(losses, threshold_pct * 100.0)
    ex = losses[losses > u] - u
    if len(ex) < 15:
        return None
    xi, _, scale = genpareto.fit(ex, floc=0.0)
    return {"xi": float(xi), "scale": float(scale),
            "threshold": float(u), "n_exceed": int(len(ex))}


def _pnl_components(mu_arr, nu_marginals, sigma_forecast, R, nu_copula, rng_seed,
                    linear_prices_now, linear_shares,
                    S_now, sigma_now, rate_now, K, T_now, n_sim):
    """Return (pnl_linear, pnl_irs, pnl_straddle) arrays of length n_sim."""
    eps       = 1e-6
    rng       = np.random.default_rng(rng_seed)
    U         = v2.simulate_t_copula(R, nu_copula, n_sim, rng)
    std_scale = np.sqrt((nu_marginals - 2.0) / nu_marginals)
    z = np.column_stack([
        scipy_t.ppf(np.clip(U[:, j], eps, 1 - eps), df=nu_marginals[j]) * std_scale[j]
        for j in range(v2.N_ASSETS)
    ])
    r_sim = mu_arr[np.newaxis, :] + sigma_forecast[np.newaxis, :] * z

    dollar_pos  = linear_shares * linear_prices_now
    pnl_linear  = (dollar_pos[np.newaxis, :] *
                   (np.exp(r_sim[:, v2.IDX_LINEAR]) - 1.0)).sum(axis=1)

    S_sim     = S_now    * np.exp(r_sim[:, v2.IDX_SPY])
    sigma_sim = sigma_now * np.exp(r_sim[:, v2.IDX_VIX])
    rate_sim  = rate_now  + r_sim[:, v2.IDX_DGS10]

    T_next      = max(T_now - 1.0 / 252.0, 1.0 / 252.0)
    v_irs_now   = v2.price_irs_vec(v2.IRS_NOTIONAL, v2.IRS_FIXED_RATE, rate_now)
    v_strad_now = v2.price_straddle_vec(S_now, K, T_now, rate_now, sigma_now)
    pnl_irs     = v2.price_irs_vec(v2.IRS_NOTIONAL, v2.IRS_FIXED_RATE, rate_sim) - v_irs_now
    pnl_strad   = (v2.price_straddle_vec(S_sim, K, T_next, rate_sim, sigma_sim)
                   - v_strad_now) * v2.STRADDLE_SHARES
    return pnl_linear, pnl_irs, pnl_strad


# =============================================================================
# EXTENDED SETUP — GARCH parameter history for Block 5
# =============================================================================

def setup_extended(d):
    """
    Enrich d in-place with GARCH parameter history by re-running rolling fits
    at every REFIT_EVERY step across the full dataset.

    Adds key 'param_history': DataFrame indexed by refit date with columns
    omega_*, alpha_*, beta_*, nu_*, persist_* for each of the 6 factors.
    """
    if "param_history" in d:
        return d

    returns_arr = d["returns_arr"]
    dates       = d["dates"]
    N           = len(returns_arr)

    print("=" * 65)
    print("SETUP EXTENDED — collecting GARCH parameter history (Block 5)")
    print("=" * 65)

    history     = []
    refit_times = list(range(v2.WINDOW, N, v2.REFIT_EVERY))
    n_refits    = len(refit_times)
    tick        = max(1, n_refits // 8)

    for i, t in enumerate(refit_times):
        W = returns_arr[t - v2.WINDOW : t]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, gp = v2.fit_garch_marginals(W)

        rec = {"date": dates[t]}
        for j, instr in enumerate(v2.INSTRUMENTS):
            rec[f"omega_{instr}"]   = gp[j]["omega"]
            rec[f"alpha_{instr}"]   = gp[j]["alpha"]
            rec[f"beta_{instr}"]    = gp[j]["beta"]
            rec[f"nu_{instr}"]      = gp[j]["nu"]
            rec[f"persist_{instr}"] = gp[j]["alpha"] + gp[j]["beta"]
        history.append(rec)

        if (i + 1) % tick == 0 or i == n_refits - 1:
            print(f"  Refit {i+1:3d}/{n_refits}  {dates[t].date()}")

    d["param_history"] = pd.DataFrame(history).set_index("date")
    print(f"  Parameter history ready ({n_refits} refit windows).\n")
    return d


# =============================================================================
# BLOCK 5 — GARCH PARAMETER STABILITY
# =============================================================================

def validate_garch_stability(d):
    """
    Track and plot GARCH(1,1)-t parameter evolution across all rolling refits.

    Tests whether ω, α, β, ν and persistence (α+β) are stable over time
    or show regime shifts.  A model with consistently α+β < 1 and stable ν
    is well-specified; large swings in ω signal unmodelled structural breaks.

    Produces:
      validate_mc_garch_t_copula_07_garch_stability.png  — 3-panel: persistence, ν, ω over refits
      validate_mc_garch_t_copula_garch_stability.csv     — full parameter history
    """
    print("=" * 65)
    print("BLOCK 5 — GARCH Parameter Stability")
    print("=" * 65)

    if "param_history" not in d:
        print("  Skipped: call setup_extended(d) first.")
        return

    df    = d["param_history"]
    instr = v2.INSTRUMENTS

    # Persistence summary table
    rows_tab = []
    for s in instr:
        col = f"persist_{s}"
        if col not in df.columns:
            continue
        vals = df[col].values
        rows_tab.append({
            "factor"      : s,
            "mean α+β"    : float(np.mean(vals)),
            "std α+β"     : float(np.std(vals)),
            "min α+β"     : float(np.min(vals)),
            "max α+β"     : float(np.max(vals)),
            "ever ≥ 1"    : bool(np.any(vals >= 1.0)),
        })
    tab_df = pd.DataFrame(rows_tab).set_index("factor")

    print("\n  GARCH Persistence  (α + β)  statistics across all refits")
    print(f"  {'─'*55}")
    print(tab_df.to_string(float_format="{:.4f}".format))
    _savetab(df.reset_index(), "validate_mc_garch_t_copula_garch_stability.csv")

    # ── Figure ────────────────────────────────────────────────────────────────
    _apply_style()
    fig   = plt.figure(figsize=(14, 10))
    gs    = gridspec.GridSpec(3, 1, hspace=0.45)
    ax_p  = fig.add_subplot(gs[0])
    ax_nu = fig.add_subplot(gs[1])
    ax_om = fig.add_subplot(gs[2])

    colors  = plt.cm.tab10(np.linspace(0, 1, len(instr)))
    dates_x = df.index

    # Panel A: α+β persistence
    for j, s in enumerate(instr):
        col = f"persist_{s}"
        if col in df.columns:
            ax_p.plot(dates_x, df[col], color=colors[j], linewidth=1.4,
                      marker="o", markersize=3, label=s)
    ax_p.axhline(1.0,  color=_C["ref"],    linewidth=1.2, linestyle="--",
                 label="Unit root (1.0)")
    ax_p.axhline(0.95, color="#888888",     linewidth=0.8, linestyle=":",
                 label="0.95 reference")
    ax_p.set_ylabel("α + β")
    ax_p.set_title("A — GARCH Persistence  (α + β)  by Refit Date")
    ax_p.legend(fontsize=7, ncol=3)
    ax_p.set_ylim(0.75, 1.03)

    # Panel B: ν (marginal tail degrees of freedom)
    for j, s in enumerate(instr):
        col = f"nu_{s}"
        if col in df.columns:
            ax_nu.plot(dates_x, df[col], color=colors[j], linewidth=1.4,
                       marker="o", markersize=3, label=s)
    ax_nu.set_ylabel("ν (d.o.f.)")
    ax_nu.set_title("B — Marginal Tail Thickness  (ν)  by Refit Date")
    ax_nu.legend(fontsize=7, ncol=3)

    # Panel C: ω × 10⁴ (unconditional variance intercept)
    for j, s in enumerate(instr):
        col = f"omega_{s}"
        if col in df.columns:
            ax_om.plot(dates_x, df[col] * 1e4, color=colors[j], linewidth=1.4,
                       marker="o", markersize=3, label=s)
    ax_om.set_ylabel("ω × 10⁴")
    ax_om.set_xlabel("Refit date")
    ax_om.set_title("C — GARCH Intercept ω  (unconditional variance target)")
    ax_om.legend(fontsize=7, ncol=3)

    fig.suptitle(
        "Block 5 — GARCH Parameter Stability Across Rolling Refits",
        fontsize=11, fontweight="bold", x=0.02, ha="left"
    )
    _savefig(fig, "validate_mc_garch_t_copula_07_garch_stability.png")


# =============================================================================
# BLOCK 6 — ASYMMETRY / SIGN-BIAS TEST  (Engle-Ng 1993)
# =============================================================================

def validate_sign_bias(d):
    """
    Engle-Ng (1993) sign-bias test on GARCH standardised residuals.

    Tests whether negative vs positive innovations have asymmetric effects on
    subsequent conditional variance — a violation of the symmetric GARCH(1,1)
    assumption.  Under GJR-GARCH or EGARCH the test would be expected to pass;
    for plain GARCH(1,1) failing factors indicate a misspecification.

    Produces:
      validate_mc_garch_t_copula_08_sign_bias.png  — 6-panel scatter z_{t-1} vs z²_t with OLS fits
      validate_mc_garch_t_copula_sign_bias.csv     — regression output per factor
    """
    print("=" * 65)
    print("BLOCK 6 — Asymmetry / Sign-Bias Test  (Engle-Ng 1993)")
    print("=" * 65)

    std_resids  = d["std_resids"]
    instruments = v2.INSTRUMENTS

    rows = []
    for j, instr in enumerate(instruments):
        z   = std_resids[:, j]
        res = _sign_bias_ols(z)
        rows.append({
            "factor"       : instr,
            "β_sign"       : float(res["beta"][1]),
            "t_sign"       : float(res["t_stats"][1]),
            "p_sign"       : float(res["p_vals"][1]),
            "β_neg_size"   : float(res["beta"][2]),
            "t_neg_size"   : float(res["t_stats"][2]),
            "p_neg_size"   : float(res["p_vals"][2]),
            "β_pos_size"   : float(res["beta"][3]),
            "t_pos_size"   : float(res["t_stats"][3]),
            "p_pos_size"   : float(res["p_vals"][3]),
            "F_joint"      : float(res["F_stat"]),
            "p_F_joint"    : float(res["p_F"]),
            "verdict"      : "FAIL" if res["p_F"] < 0.05 else "pass",
        })

    sb_df = pd.DataFrame(rows).set_index("factor")
    print(f"\n  Engle-Ng Sign-Bias  (H0: symmetric vol, no asymmetric effects)")
    print(f"  Joint F-test p ≥ 0.05 = PASS  (GARCH(1,1) symmetry holds)")
    print(f"  {'─'*60}")
    disp_cols = ["β_sign", "t_sign", "p_sign",
                 "β_neg_size", "t_neg_size",
                 "β_pos_size", "t_pos_size",
                 "F_joint", "p_F_joint", "verdict"]
    print(sb_df[disp_cols].to_string(float_format="{:.4f}".format))
    _savetab(sb_df.reset_index(), "validate_mc_garch_t_copula_sign_bias.csv")

    d["_sign_bias_df"] = sb_df

    # ── Figure ────────────────────────────────────────────────────────────────
    _apply_style()
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(
        "Block 6 — Sign-Bias: z²_t vs z_{t-1}  "
        "(red = negative prior shock; blue = positive prior shock)",
        fontsize=10, fontweight="bold", x=0.02, ha="left"
    )
    for j, (ax, instr) in enumerate(zip(axes.flat, instruments)):
        z     = std_resids[:, j]
        z_lag = z[:-1]
        z2_t  = z[1:] ** 2
        s_neg = z_lag < 0

        ax.scatter(z_lag[ s_neg], z2_t[ s_neg], s=5, alpha=0.30,
                   color=_C["ref"],  label="Neg prior")
        ax.scatter(z_lag[~s_neg], z2_t[~s_neg], s=5, alpha=0.30,
                   color=_C["hist"], label="Pos prior")

        # OLS fit lines per regime
        for mask, col in [(s_neg, _C["ref"]), (~s_neg, _C["hist"])]:
            xm, ym = z_lag[mask], z2_t[mask]
            if len(xm) > 4:
                m_fit, b_fit = np.polyfit(xm, ym, 1)
                xs = np.linspace(xm.min(), xm.max(), 80)
                ax.plot(xs, m_fit * xs + b_fit, color=col, linewidth=1.6)

        r    = sb_df.loc[instr]
        col_v = _C["fail"] if r["p_F_joint"] < 0.05 else _C["pass"]
        ax.set_title(f"{instr}  F={r['F_joint']:.2f}  p={r['p_F_joint']:.3f}",
                     color=col_v)
        ax.set_xlabel("z_{t-1}")
        ax.set_ylabel("z²_t")
        ax.set_ylim(0, np.percentile(z2_t, 98))
        if j == 0:
            ax.legend(fontsize=7)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    _savefig(fig, "validate_mc_garch_t_copula_08_sign_bias.png")


# =============================================================================
# BLOCK 7 — COMPONENT P&L VALIDATION
# =============================================================================

def validate_component_pnl(d):
    """
    Decompose portfolio P&L into linear, IRS, and straddle components for both
    the MC simulation and the historical final window (current-state repricing).

    Current-state repricing applies the WINDOW historical factor shocks to the
    current portfolio positions — the same method used in v3's stress test —
    so the comparison is like-for-like at today's instrument state.

    Produces:
      validate_mc_garch_t_copula_09_component_pnl.png  — 3 histogram panels + volatility pie chart
      validate_mc_garch_t_copula_component_pnl.csv     — moment table per component per source
    """
    print("=" * 65)
    print("BLOCK 7 — Component P&L Validation")
    print("=" * 65)

    mu_arr    = d["mu_arr"]
    nu_marg   = d["nu_marginals"]
    sig_fc    = d["sigma_forecast"]
    R         = d["R_static"]
    nu_cop    = d["nu_copula"]
    lp_now    = d["linear_prices_now"]
    lshares   = d["linear_shares"]
    S_now     = d["S_now"]
    sigma_now = d["sigma_now"]
    rate_now  = d["rate_now"]
    K         = d["K_atm"]
    T_now     = d["T_now"]
    W         = d["W"]

    # ── Simulated components ──────────────────────────────────────────────────
    pnl_lin_s, pnl_irs_s, pnl_str_s = _pnl_components(
        mu_arr, nu_marg, sig_fc, R, nu_cop, SEED_VAL + 200,
        lp_now, lshares, S_now, sigma_now, rate_now, K, T_now, N_SIM_VAL
    )

    # ── Historical components (current-state repricing of W) ─────────────────
    S_hist      = S_now    * np.exp(W[:, v2.IDX_SPY])
    sigma_hist  = sigma_now * np.exp(W[:, v2.IDX_VIX])
    rate_hist   = rate_now  + W[:, v2.IDX_DGS10]
    dollar_pos  = lshares * lp_now
    pnl_lin_h   = (dollar_pos[np.newaxis, :] *
                   (np.exp(W[:, v2.IDX_LINEAR]) - 1.0)).sum(axis=1)
    T_next      = max(T_now - 1.0 / 252.0, 1.0 / 252.0)
    v_irs_now   = v2.price_irs_vec(v2.IRS_NOTIONAL, v2.IRS_FIXED_RATE, rate_now)
    v_strad_now = v2.price_straddle_vec(S_now, K, T_now, rate_now, sigma_now)
    pnl_irs_h   = v2.price_irs_vec(v2.IRS_NOTIONAL, v2.IRS_FIXED_RATE, rate_hist) - v_irs_now
    pnl_str_h   = (v2.price_straddle_vec(S_hist, K, T_next, rate_hist, sigma_hist)
                   - v_strad_now) * v2.STRADDLE_SHARES

    # ── Summary table ─────────────────────────────────────────────────────────
    total_std_s = float(np.std(pnl_lin_s + pnl_irs_s + pnl_str_s)) + 1e-12
    total_std_h = float(np.std(pnl_lin_h + pnl_irs_h + pnl_str_h)) + 1e-12

    tab_rows = []
    for comp, xs, xh in [
        ("Linear",   pnl_lin_s, pnl_lin_h),
        ("IRS",      pnl_irs_s, pnl_irs_h),
        ("Straddle", pnl_str_s, pnl_str_h),
    ]:
        for x, src, ts in [(xs, "Simulated",  total_std_s),
                           (xh, "Historical", total_std_h)]:
            tab_rows.append({
                "component"       : comp,
                "source"          : src,
                "mean"            : float(np.mean(x)),
                "std"             : float(np.std(x)),
                "q1%"             : float(np.percentile(x, 1.0)),
                "q99%"            : float(np.percentile(x, 99.0)),
                "vol_share"       : float(np.std(x)) / ts,
            })
    comp_df = pd.DataFrame(tab_rows)

    print("\n  Component P&L Summary  (current-state repricing for historical)")
    print(f"  {'─'*65}")
    print(comp_df.set_index(["component", "source"]).to_string(
        float_format="{:,.0f}".format
    ))
    _savetab(comp_df, "validate_mc_garch_t_copula_component_pnl.csv")

    # ── Figure ────────────────────────────────────────────────────────────────
    _apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        "Block 7 — Component P&L: Simulated vs Historical (current-state repricing)",
        fontsize=11, fontweight="bold", x=0.02, ha="left"
    )
    comp_panels = [
        ("Linear",   pnl_lin_s, pnl_lin_h,  axes[0, 0]),
        ("IRS",      pnl_irs_s, pnl_irs_h,  axes[0, 1]),
        ("Straddle", pnl_str_s, pnl_str_h,  axes[1, 0]),
    ]
    for comp, xs, xh, ax in comp_panels:
        all_v = np.concatenate([xs, xh])
        lo, hi = np.percentile(all_v, [0.5, 99.5])
        bins = np.linspace(lo, hi, 55)
        ax.hist(xh, bins=bins, density=True, alpha=0.55,
                color=_C["hist"], label="Historical")
        ax.hist(xs, bins=bins, density=True, alpha=0.45,
                color=_C["sim"],  label="Simulated")
        ax.axvline(np.percentile(xs, 1.0), color=_C["sim"],    linewidth=1.2,
                   linestyle="--", label="1% sim")
        ax.axvline(np.percentile(xh, 1.0), color=_C["actual"], linewidth=1.2,
                   linestyle=":",  label="1% hist")
        ax.axvline(0, color="#aaaaaa", linewidth=0.6)
        ax.set_title(comp)
        ax.set_xlabel("Component P&L (USD)")
        ax.set_ylabel("Density")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1e"))
        if comp == "Linear":
            ax.legend(fontsize=7)

    # Volatility share pie (simulated)
    ax_pie = axes[1, 1]
    std_vals = [float(np.std(pnl_lin_s)),
                float(np.std(pnl_irs_s)),
                float(np.std(pnl_str_s))]
    ax_pie.pie(std_vals, labels=["Linear", "IRS", "Straddle"],
               colors=[_C["hist"], _C["sim"], _C["ref"]],
               autopct="%1.1f%%", startangle=90,
               textprops={"fontsize": 9})
    ax_pie.set_title("Simulated Volatility Share by Component")

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _savefig(fig, "validate_mc_garch_t_copula_09_component_pnl.png")


# =============================================================================
# BLOCK 8 — STRESS / EXTREME SCENARIO VALIDATION
# =============================================================================

def validate_stress_scenarios(d, n_worst=10):
    """
    Apply every scenario from the final WINDOW to the current portfolio state
    (current-state repricing) and identify the worst-N historical losses.

    Tests whether these historical extremes are captured by the MC tail:
    if many stress scenarios lie far outside the simulated distribution the
    model is under-estimating tail risk.

    Produces:
      validate_mc_garch_t_copula_10_stress_scenarios.png  — density + stress rug, ranked bar chart
      validate_mc_garch_t_copula_stress_scenarios.csv     — top-N worst scenarios with components
    """
    print("=" * 65)
    print("BLOCK 8 — Stress / Extreme Scenario Validation")
    print("=" * 65)

    W         = d["W"]
    W_dates   = d["W_dates"]
    pnl_sim   = d["pnl_sim"]
    S_now     = d["S_now"]
    sigma_now = d["sigma_now"]
    rate_now  = d["rate_now"]
    K         = d["K_atm"]
    T_now     = d["T_now"]
    lp_now    = d["linear_prices_now"]
    lshares   = d["linear_shares"]

    # Current-state repricing of all WINDOW historical scenarios
    S_hist      = S_now    * np.exp(W[:, v2.IDX_SPY])
    sigma_hist  = sigma_now * np.exp(W[:, v2.IDX_VIX])
    rate_hist   = rate_now  + W[:, v2.IDX_DGS10]
    dollar_pos  = lshares * lp_now
    pnl_lin_h   = (dollar_pos[np.newaxis, :] *
                   (np.exp(W[:, v2.IDX_LINEAR]) - 1.0)).sum(axis=1)
    T_next      = max(T_now - 1.0 / 252.0, 1.0 / 252.0)
    v_irs_now   = v2.price_irs_vec(v2.IRS_NOTIONAL, v2.IRS_FIXED_RATE, rate_now)
    v_strad_now = v2.price_straddle_vec(S_now, K, T_now, rate_now, sigma_now)
    pnl_irs_h   = v2.price_irs_vec(v2.IRS_NOTIONAL, v2.IRS_FIXED_RATE, rate_hist) - v_irs_now
    pnl_str_h   = (v2.price_straddle_vec(S_hist, K, T_next, rate_hist, sigma_hist)
                   - v_strad_now) * v2.STRADDLE_SHARES
    pnl_all_h   = pnl_lin_h + pnl_irs_h + pnl_str_h

    rank_order = np.argsort(pnl_all_h)
    var_mc_01  = float(-np.percentile(pnl_sim, 1.0))
    n_breach   = int(np.sum(pnl_all_h < -var_mc_01))

    print(f"\n  MC 1% VaR (current-window simulation): ${var_mc_01:,.0f}")
    print(f"  Hist scenarios breaching MC VaR: {n_breach} / {len(pnl_all_h)}")
    print(f"\n  Top {n_worst} worst historical scenarios (current-state repricing):")
    print(f"  {'─'*72}")
    print(f"  {'Rank':>4}  {'Date':<11}  {'Total P&L':>12}  "
          f"{'Linear':>10}  {'IRS':>10}  {'Straddle':>10}  Flag")

    stress_rows = []
    for rank, idx in enumerate(rank_order[:n_worst], start=1):
        dt  = W_dates[idx]
        dt_s = dt.date().isoformat() if hasattr(dt, "date") else str(dt)
        tp   = float(pnl_all_h[idx])
        lin  = float(pnl_lin_h[idx])
        irs  = float(pnl_irs_h[idx])
        strd = float(pnl_str_h[idx])
        flag = "✗ BREACH" if tp < -var_mc_01 else ""
        print(f"  {rank:>4}  {dt_s:<11}  {tp:>12,.0f}  "
              f"{lin:>10,.0f}  {irs:>10,.0f}  {strd:>10,.0f}  {flag}")
        stress_rows.append({
            "rank": rank, "date": dt_s, "pnl_total": tp,
            "pnl_linear": lin, "pnl_irs": irs, "pnl_straddle": strd,
            "breach_mc_var": tp < -var_mc_01,
        })

    stress_df = pd.DataFrame(stress_rows)
    _savetab(stress_df, "validate_mc_garch_t_copula_stress_scenarios.csv")
    d["_stress_n_breach"] = n_breach
    d["_stress_n_total"]  = len(pnl_all_h)
    d["_var_mc_01"]       = var_mc_01

    # ── Figure ────────────────────────────────────────────────────────────────
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Block 8 — Stress Validation: Historical Worst Scenarios vs MC Distribution",
        fontsize=11, fontweight="bold", x=0.02, ha="left"
    )

    # Panel A: MC density with stress scenario rug
    ax = axes[0]
    lo_p, hi_p = np.percentile(pnl_sim, [0.1, 99.9])
    bins = np.linspace(lo_p, hi_p, 70)
    ax.hist(pnl_sim, bins=bins, density=True, alpha=0.55,
            color=_C["sim"], label="MC simulated (final window)")
    ax.axvline(-var_mc_01, color=_C["ref"], linewidth=1.5, linestyle="--",
               label=f"MC 1% VaR = ${var_mc_01:,.0f}")
    worst_pnl = pnl_all_h[rank_order[:n_worst]]
    for val in worst_pnl:
        color = _C["fail"] if val < -var_mc_01 else _C["neutral"]
        ax.axvline(val, color=color, linewidth=1.0, alpha=0.85,
                   ymin=0.0, ymax=0.20)
    ax.set_xlabel("Portfolio P&L (USD)")
    ax.set_ylabel("Density")
    ax.set_title(f"A — MC Distribution + Top-{n_worst} Hist Stress Losses")
    ax.legend(fontsize=8)

    # Panel B: ranked bar chart of losses
    ax = axes[1]
    ranks   = [r["rank"] for r in stress_rows]
    losses  = [-r["pnl_total"] for r in stress_rows]
    labels  = [f"#{r['rank']}  {r['date'][:7]}" for r in stress_rows]
    colors  = [_C["fail"] if r["breach_mc_var"] else _C["neutral"]
               for r in stress_rows]
    ax.barh(ranks, losses, color=colors, alpha=0.85)
    ax.axvline(var_mc_01, color=_C["ref"], linewidth=1.5, linestyle="--",
               label="MC 1% VaR")
    ax.set_yticks(ranks)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("Loss (USD)")
    ax.set_title(f"B — Top-{n_worst} Historical Losses  (red = exceed VaR)")
    ax.legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _savefig(fig, "validate_mc_garch_t_copula_10_stress_scenarios.png")


# =============================================================================
# BLOCK 9 — TAIL-SPECIFIC VALIDATION
# =============================================================================

def validate_tail_behavior(d):
    """
    Tail-specific validation: extreme quantile accuracy, GPD fit comparison,
    Hill estimator for tail index stability.

    Tests:
    - Extreme quantiles (0.1 %, 0.5 %, 1 %, 2.5 %, 5 %) sim vs hist
    - GPD shape parameter ξ: consistent between simulated and historical tails
    - Hill plot: tail index α_H = 1/ξ; stable plateau indicates power-law tail

    Produces:
      validate_mc_garch_t_copula_11_tail_behavior.png  — Q-Q tail, GPD densities, Hill plot
      validate_mc_garch_t_copula_tail_behavior.csv     — quantile table + GPD parameters
    """
    print("=" * 65)
    print("BLOCK 9 — Tail-Specific Validation")
    print("=" * 65)

    pnl_sim = d["pnl_sim"]
    pnl_bt  = d["pnl_bt"]

    losses_sim  = -pnl_sim
    losses_hist = -pnl_bt.values[-v2.WINDOW:]

    # ── Extreme quantile comparison ───────────────────────────────────────────
    tail_pcts = [0.1, 0.5, 1.0, 2.5, 5.0]
    q_rows = []
    for pct in tail_pcts:
        q_s = float(np.percentile(losses_sim,  pct))
        q_h = float(np.percentile(losses_hist, pct))
        ratio = q_s / (q_h + 1e-8) if abs(q_h) > 1e-8 else np.nan
        q_rows.append({
            "loss_pct" : pct,
            "q_sim"    : q_s,
            "q_hist"   : q_h,
            "ratio"    : ratio,
            "verdict"  : "FAIL" if (np.isfinite(ratio) and abs(ratio - 1.0) > 0.50) else "pass",
        })
    quant_df = pd.DataFrame(q_rows)

    print(f"\n  Extreme Quantile Comparison  (losses = -P&L)")
    print(f"  {'─'*55}")
    print(quant_df.to_string(index=False, float_format="{:,.0f}".format))

    # ── GPD fits ─────────────────────────────────────────────────────────────
    gpd_s = _fit_gpd_tail(losses_sim,  threshold_pct=0.90)
    gpd_h = _fit_gpd_tail(losses_hist, threshold_pct=0.90)

    print(f"\n  GPD Fit  (losses > 90th-pct threshold)")
    print(f"  {'─'*50}")
    for label, g in [("Simulated",  gpd_s),
                     ("Historical", gpd_h)]:
        if g is not None:
            print(f"  {label:<12}  ξ={g['xi']:+.4f}  "
                  f"scale={g['scale']:,.1f}  "
                  f"threshold=${g['threshold']:,.0f}  "
                  f"n_exceed={g['n_exceed']}")
        else:
            print(f"  {label:<12}  GPD fit unavailable (too few exceedances)")

    gpd_xi_ok = False
    if gpd_s is not None and gpd_h is not None:
        xi_diff   = abs(gpd_s["xi"] - gpd_h["xi"])
        gpd_xi_ok = xi_diff < 0.50
        print(f"\n  |ξ_sim − ξ_hist| = {xi_diff:.4f}   "
              f"{'PASS ✓ (< 0.5)' if gpd_xi_ok else 'FAIL ✗ (≥ 0.5)'}")

    d["_gpd_xi_ok"]    = gpd_xi_ok
    d["_tail_quant_df"] = quant_df

    # Save tables
    gpd_rows = []
    for label, g in [("simulated", gpd_s), ("historical", gpd_h)]:
        if g:
            gpd_rows.append({"source": label, **g})
    gpd_tab = pd.DataFrame(gpd_rows) if gpd_rows else pd.DataFrame()
    _savetab(quant_df, "validate_mc_garch_t_copula_tail_behavior.csv")
    if not gpd_tab.empty:
        _savetab(gpd_tab, "validate_mc_garch_t_copula_gpd_params.csv")

    # ── Figure ────────────────────────────────────────────────────────────────
    _apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "Block 9 — Tail-Specific Validation",
        fontsize=11, fontweight="bold", x=0.02, ha="left"
    )

    # Panel A: tail Q-Q (losses bottom 10%)
    ax = axes[0]
    pcts = np.linspace(0.1, 10.0, 80)
    q_s_arr = np.percentile(losses_sim,  pcts)
    q_h_arr = np.percentile(losses_hist, pcts)
    ax.plot(q_h_arr, q_s_arr, color=_C["sim"], linewidth=1.5, label="Q-Q")
    diag_lo = min(q_s_arr.min(), q_h_arr.min())
    diag_hi = max(q_s_arr.max(), q_h_arr.max())
    ax.plot([diag_lo, diag_hi], [diag_lo, diag_hi],
            color=_C["ref"], linewidth=1.2, linestyle="--", label="y = x")
    ax.fill_between(q_h_arr, q_s_arr * 0.50, q_s_arr * 1.50,
                    alpha=0.10, color=_C["neutral"], label="±50% band")
    ax.set_xlabel("Historical loss quantiles (USD)")
    ax.set_ylabel("Simulated loss quantiles (USD)")
    ax.set_title("A — Tail Q-Q  (bottom 10 %)")
    ax.legend(fontsize=7)

    # Panel B: GPD density overlay
    ax = axes[1]
    if gpd_s is not None and gpd_h is not None:
        from scipy.stats import genpareto
        x_max = max(gpd_s["threshold"] + 5.0 * gpd_s["scale"],
                    gpd_h["threshold"] + 5.0 * gpd_h["scale"])
        for label, g, col in [("Simulated",  gpd_s, _C["sim"]),
                               ("Historical", gpd_h, _C["hist"])]:
            xs  = np.linspace(0, x_max - g["threshold"], 300)
            pdf = genpareto.pdf(xs, c=g["xi"], scale=g["scale"])
            ax.plot(xs + g["threshold"], pdf, color=col, linewidth=1.5,
                    label=f"{label}  ξ={g['xi']:.3f}")
        ax.set_xlabel("Loss (USD)")
        ax.set_ylabel("GPD density")
        ax.set_title("B — GPD Fit to Left Tail  (90th-pct threshold)")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "GPD fit unavailable\n(too few exceedances)",
                ha="center", va="center", transform=ax.transAxes, fontsize=9)
        ax.set_title("B — GPD Fit (unavailable)")

    # Panel C: Hill plot
    ax = axes[2]
    k_s, hill_s = _hill_estimator(losses_sim,  k_max=200)
    k_h, hill_h = _hill_estimator(losses_hist,
                                   k_max=min(150, len(losses_hist) // 2))
    ax.plot(k_s, hill_s, color=_C["sim"],  linewidth=1.3, alpha=0.85,
            label="Simulated")
    ax.plot(k_h, hill_h, color=_C["hist"], linewidth=1.3, alpha=0.85,
            label="Historical")
    ax.axhline(2.0, color="#aaaaaa", linewidth=0.8, linestyle=":",
               label="α_H = 2  (finite-variance threshold)")
    ax.set_xlabel("k  (order statistics used)")
    ax.set_ylabel("Hill tail index  α_H")
    ax.set_title("C — Hill Plot  (larger α_H = lighter tail)")
    ax.set_ylim(0, 12)
    ax.legend(fontsize=7)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _savefig(fig, "validate_mc_garch_t_copula_11_tail_behavior.png")


# =============================================================================
# SUMMARY
# =============================================================================

def print_summary(d):
    """Print a one-page pass/fail summary of all validation checks."""
    std_resids     = d["std_resids"]
    nu_marginals   = d["nu_marginals"]
    sigma_forecast = d["sigma_forecast"]
    R_static       = d["R_static"]
    nu_copula      = d["nu_copula"]
    r_sim          = d["r_sim"]
    W              = d["W"]
    pnl_sim        = d["pnl_sim"]
    pnl_bt         = d["pnl_bt"]

    print("\n" + "=" * 65)
    print("VALIDATION SUMMARY  —  mc_garch_t_copula.py")
    print("=" * 65)

    # GARCH residuals: check mean ≈ 0, std ≈ 1
    means = np.abs([np.mean(std_resids[:, j]) for j in range(v2.N_ASSETS)])
    stds  = np.abs([np.std(std_resids[:, j], ddof=1) - 1.0
                    for j in range(v2.N_ASSETS)])
    garch_mean_ok = all(m < 0.1 for m in means)
    garch_std_ok  = all(s < 0.15 for s in stds)

    # Copula: CvM GoF
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, cvm_p = v2.copula_gof_cvm(
            v2.compute_pseudo_observations(std_resids),
            R_static, nu_copula, n_mc=100, rng_seed=SEED_VAL
        )
    copula_ok = cvm_p >= 0.05

    # Simulation: KS test per factor
    ks_pass = []
    for j in range(v2.N_ASSETS):
        _, p = stats.ks_2samp(r_sim[:, j] / sigma_forecast[j],
                      std_resids[:, j])
        ks_pass.append(p >= 0.05)

    # P&L coverage
    actual_q01 = float(np.percentile(pnl_bt.values[-v2.WINDOW:], 1.0))
    sim_q01    = float(np.percentile(pnl_sim, 1.0))
    pnl_ok     = abs(actual_q01 - sim_q01) / (abs(actual_q01) + 1e-8) < 0.30

    def _tick(ok):
        return "PASS ✓" if ok else "FAIL ✗"

    # Block 5: persistence in final-window GARCH params (already in d)
    garch_params  = d["garch_params"]
    persist_vals  = np.array([garch_params[j]["alpha"] + garch_params[j]["beta"]
                               for j in range(v2.N_ASSETS)])
    persist_ok    = bool(np.all(persist_vals < 1.0))

    # Block 6: sign-bias — use cached result if block was run, else compute now
    if "_sign_bias_df" in d:
        sb_p = d["_sign_bias_df"]["p_F_joint"].values
    else:
        sb_p = np.array([_sign_bias_ols(std_resids[:, j])["p_F"]
                         for j in range(v2.N_ASSETS)])
    n_sb_fail   = int(np.sum(np.array(sb_p) < 0.05))
    sign_bias_ok = n_sb_fail == 0

    # Blocks 8-9: use flags cached by the block functions if they were run
    stress_n_breach = d.get("_stress_n_breach", None)
    stress_n_total  = d.get("_stress_n_total",  None)
    gpd_xi_ok       = d.get("_gpd_xi_ok",       None)
    tail_quant_df   = d.get("_tail_quant_df",   None)
    tail_ok = (bool(tail_quant_df["verdict"].eq("pass").all())
               if tail_quant_df is not None else None)

    rows_summary = [
        ("GARCH: residual mean ≈ 0  (max|mean| < 0.10)",
         _tick(garch_mean_ok), f"max|mean| = {means.max():.4f}"),
        ("GARCH: residual std ≈ 1   (max|std-1| < 0.15)",
         _tick(garch_std_ok),  f"max|std-1| = {stds.max():.4f}"),
        ("Copula: CvM GoF test  (p ≥ 0.05)",
         _tick(copula_ok),     f"p = {cvm_p:.4f}"),
        ("Simulation: KS test all factors",
         _tick(all(ks_pass)),  f"{sum(ks_pass)}/{len(ks_pass)} factors pass"),
        ("P&L: simulated 1% ≈ actual 1%  (within 30%)",
         _tick(pnl_ok),
         f"sim={sim_q01:,.0f}  actual={actual_q01:,.0f}"),
        ("GARCH: α+β < 1 all factors  (final window)",
         _tick(persist_ok),
         f"max(α+β) = {persist_vals.max():.4f}"),
        ("Sign-bias: GARCH(1,1) symmetry  (F-test p ≥ 0.05)",
         _tick(sign_bias_ok),
         f"{v2.N_ASSETS - n_sb_fail}/{v2.N_ASSETS} factors pass"),
        ("Stress: worst hist scenarios vs VaR  (Block 8)",
         (_tick(stress_n_breach == 0) if stress_n_breach is not None
          else "not run"),
         (f"{stress_n_breach} breaches / {stress_n_total} scenarios"
          if stress_n_breach is not None else "run validate_stress_scenarios(d)")),
        ("Tail: quantile ratios within ±50%  (Block 9)",
         (_tick(tail_ok) if tail_ok is not None else "not run"),
         ("all pass" if tail_ok else
          ("some FAIL" if tail_ok is False else "run validate_tail_behavior(d)"))),
        ("Tail: GPD ξ consistent  |ξ_sim−ξ_hist| < 0.5  (Block 9)",
         (_tick(gpd_xi_ok) if gpd_xi_ok is not None else "not run"),
         ("" if gpd_xi_ok is None else "")),
    ]

    print(f"\n  {'Check':<55} {'Result':<12} {'Detail'}")
    print(f"  {'─'*70}")
    for check, result, detail in rows_summary:
        print(f"  {check:<55} {result:<12} {detail}")

    core_pass = (garch_mean_ok and garch_std_ok and copula_ok
                 and all(ks_pass) and pnl_ok)
    ext_pass  = persist_ok and sign_bias_ok
    all_pass  = core_pass and ext_pass
    print(f"\n  Core (Blocks 1-4): {'PASS ✓' if core_pass else 'FAIL ✗'}")
    print(f"  Extended (Blocks 5-9): {'PASS ✓' if ext_pass else 'FAIL / not run — review figures'}")
    print(f"\n  Overall: {'ALL CHECKS PASS ✓' if all_pass else 'SOME CHECKS FAILED — review figures'}")
    print("=" * 65)


# =============================================================================
# MAIN
# =============================================================================

def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    d = setup()

    validate_pnl_mapping(d)
    print()

    validate_garch_residuals(d)
    print()

    validate_copula(d)
    print()

    validate_simulation(d)
    print()

    # Blocks 5-9 — extended validation
    d = setup_extended(d)

    validate_garch_stability(d)
    print()

    validate_sign_bias(d)
    print()

    validate_component_pnl(d)
    print()

    validate_stress_scenarios(d)
    print()

    validate_tail_behavior(d)
    print()

    print_summary(d)


if __name__ == "__main__":
    main()
