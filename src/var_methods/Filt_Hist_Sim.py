"""
Filtered Historical Simulation (FHS) Value-at-Risk

Method:
    Step 1 — Fit a GARCH(1,1)-t model on an expanding window (same spec as
              garch_evt.py: arch library, returns scaled ×100, refit every 50 days).
    Step 2 — Collect the last WINDOW=500 standardised residuals  ẑ_t as the
              empirical shock distribution.
    Step 3 — Scale each historical shock by today's one-step-ahead conditional
              volatility forecast:  r̃_i = ẑ_i × σ̂_t
    Step 4 — Build the simulated P&L distribution and read off the 1st percentile
              (99% VaR, 1-day horizon).

Why FHS?
    Plain HS ignores volatility clustering → Christoffersen p_IND rejected (EVT:
    0.0064, Copula: 0.0003).  FHS inherits the GARCH volatility signal while
    keeping a non-parametric tail — no GPD shape assumption needed.  Expected to
    land between plain HS and GARCH+EVT in terms of exception clustering.

Assumptions documented (mirroring Assumption_Overview_Var_Project format):
    FHS-A  GARCH(1,1) with Student-t innovations is sufficient  [same as B.1/B.2]
    FHS-B  Standardised residuals ẑ_t are approximately i.i.d.  [same as B.4]
    FHS-C  500 historical residuals form a representative empirical distribution
    FHS-D  One-step-ahead σ̂_t is the correct scaling variable   [same as B.7]
    FHS-E  Equal-weight portfolio P&L uses log-return approx.    [same as D.1]
"""

# Imports
from gettext import install
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from arch import arch_model
from scipy import stats

# Config - mirrors garch_evt.py and config.py
WINDOW        = 500      # burn-in / residual look-back (trading days)
REFIT_EVERY   = 50       # re-estimate GARCH every N days (expanding window)
CONFIDENCE    = 0.99     # VaR confidence level (99%)
ALPHA         = 1 - CONFIDENCE   # exception probability = 0.01
V0            = 1_000_000        # portfolio notional ($)
RANDOM_SEED   = 42               # for reproducibility of any stochastic steps

#Paths
PNL_PATH      = "data/processed/total_portfolio_pnl.csv"
RETURNS_PATH  = "data/processed/log_returns.csv"
OUT_VAR       = "data/processed/var_fhs.csv"
OUT_BT        = "outputs/tables/backtest_fhs.csv"
FIG_DIAG      = "outputs/figures/fhs_01_diagnostics.png"
FIG_VAR       = "outputs/figures/fhs_02_var_and_exceptions.png"

# Weights (config.py convention)
WEIGHTS = np.array([0.25, 0.25, 0.25, 0.25])

#1. Load data
def load_data(returns_path: str, pnl_path: str):
    """
    Load log-returns and total portfolio P&L.
 
    Returns
    -------
    r_port : pd.Series  — equal-weighted portfolio log-return (4 linear assets)
    pnl    : pd.Series  — total portfolio P&L including IRS + straddle
    """
    log_returns = pd.read_csv(returns_path, index_col=0, parse_dates=True)
    pnl         = pd.read_csv(pnl_path,     index_col=0, parse_dates=True)
 
    # Equal-weighted portfolio return (linear assets only; same as garch_evt.py)
    r_port = log_returns.dot(WEIGHTS)
    r_port.name = "portfolio_return"
 
    # Align dates
    pnl = pnl["pnl_total"] if "pnl_total" in pnl.columns else pnl.iloc[:, 0]
    common = r_port.index.intersection(pnl.index)
    r_port = r_port.loc[common]
    pnl    = pnl.loc[common]
 
    print(f"[Data]  {len(common)} aligned trading days  "
          f"({common[0].date()} → {common[-1].date()})")
    return r_port, pnl

#2. Fit GARCH 
def fit_garch(train: pd.Series):
    """
    Fit GARCH(1,1) with Student-t innovations on a training series.
 
    Parameters
    ----------
    train : pd.Series of portfolio log-returns (decimal, e.g. 0.012)
 
    Returns
    -------
    res        : arch ModelResult
    mu_hat     : float  — conditional mean (decimal)
    alpha1     : float  — ARCH parameter
    beta       : float  — GARCH parameter
    nu         : float  — degrees of freedom of t-innovation
    """
    # arch library requires percentage returns for numerical stability (F.2)
    train_pct = train * 100.0
 
    model = arch_model(
        train_pct,
        vol="Garch", p=1, q=1,
        dist="t",
        mean="Constant",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = model.fit(disp="off", show_warning=False)
 
    mu_hat = res.params["mu"]    / 100.0   # back to decimal
    alpha1 = res.params.get("alpha[1]", res.params.get("alpha", np.nan))
    beta   = res.params.get("beta[1]",  res.params.get("beta",  np.nan))
    nu     = res.params.get("nu", 5.0)     # degrees of freedom
 
    ab = alpha1 + beta
    status = "OK" if ab < 1.0 else "!! UNIT ROOT"
    print(f"   GARCH refit | μ={mu_hat*100:.4f}%  α+β={ab:.4f} {status}  ν={nu:.2f}")
 
    return res, mu_hat, alpha1, beta, nu

# 3.  MAIN FHS LOOP
def compute_fhs_var(r_port: pd.Series) -> pd.DataFrame:
    """
    Rolling FHS VaR over the full sample.
 
    Algorithm (per day t  after burn-in):
      (a) If refit due: re-estimate GARCH(1,1)-t on expanding window r[0:t].
      (b) Compute one-step-ahead conditional volatility σ̂_t.
      (c) Collect the last WINDOW standardised residuals  ẑ_{t-WINDOW:t}.
      (d) Scale: r̃_i = ẑ_i × σ̂_t   (FHS shock scenarios).
      (e) Build simulated P&L: PnL_i = V0 × r̃_i  (linear P&L only; same
          simplification as plain HS — nonlinear P&L added via pnl_total in BT).
      (f) VaR_t = −quantile(PnL_i, ALPHA).
 
    Returns
    -------
    DataFrame with columns: date, sigma_hat, var_fhs, z_mean, z_std
    """
    n   = len(r_port)
    dates_all = r_port.index
    r_vals    = r_port.values
 
    # Output containers
    var_arr   = np.full(n, np.nan)
    sigma_arr = np.full(n, np.nan)
 
    # Standardised residual store (filled incrementally)
    z_arr     = np.full(n, np.nan)
 
    # Track last fitted result
    res_current   = None
    mu_hat_prev   = 0.0
    last_refit_t  = WINDOW   # first refit at burn-in boundary
 
    print(f"\n[FHS]  Running over {n - WINDOW} backtest days "
          f"(burn-in = {WINDOW}, refit every {REFIT_EVERY} days) ...\n")
 
    for t in range(WINDOW, n):
 
        # ------------------------------------------------------------------
        # (a) Refit GARCH on expanding window when due
        # ------------------------------------------------------------------
        if res_current is None or (t - last_refit_t) >= REFIT_EVERY:
            train = r_port.iloc[:t]
            res_current, mu_hat_prev, _, _, _ = fit_garch(train)
            last_refit_t = t
 
        # ------------------------------------------------------------------
        # (b) One-step-ahead conditional volatility  σ̂_t
        #     arch forecast on data through t-1 (same as garch_evt.py B.7)
        # ------------------------------------------------------------------
        train_pct = r_port.iloc[:t] * 100.0
        model_t = arch_model(train_pct, vol="Garch", p=1, q=1,
                             dist="t", mean="Constant")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res_t = model_t.fit(disp="off", show_warning=False,
                                starting_values=res_current.params.values)
 
        fcast         = res_t.forecast(horizon=1, reindex=False)
        sigma_sq_curr = fcast.variance.iloc[-1, 0] / 10_000.0   # pct² → decimal²
        sigma_hat_t   = np.sqrt(max(sigma_sq_curr, 1e-10))
        sigma_arr[t]  = sigma_hat_t
 
        # ------------------------------------------------------------------
        # (c) Standardised residual for day t-1 (for residual history)
        #     z_{t-1} = (r_{t-1} - μ̂) / σ̂_{t-1}
        # ------------------------------------------------------------------
        if t > WINDOW:
            # Use the in-sample conditional vols from the current fit
            resid_pct = res_t.resid            # in percentage units
            cvol_pct  = res_t.conditional_volatility
            z_series  = (resid_pct / cvol_pct).values  # standardised residuals
            # Store only the last value (t-1 position in our expanding series)
            z_arr[t - 1] = z_series[-1]
 
        # ------------------------------------------------------------------
        # (d) Collect WINDOW most-recent standardised residuals
        # ------------------------------------------------------------------
        z_window = z_arr[max(0, t - WINDOW): t]
        z_window = z_window[~np.isnan(z_window)]   # drop leading NaNs
 
        if len(z_window) < 50:
            # Not enough residual history yet — skip
            continue
 
        # ------------------------------------------------------------------
        # (e) FHS scenarios: scale historical shocks by today's σ̂_t
        # ------------------------------------------------------------------
        simulated_returns = z_window * sigma_hat_t   # FHS core equation
 
        # Linear P&L scenarios  (V0 already scaled to total portfolio return)
        simulated_pnl = V0 * simulated_returns
 
        # ------------------------------------------------------------------
        # (f) VaR = negative of the ALPHA-quantile of simulated P&L
        # ------------------------------------------------------------------
        var_arr[t] = -np.quantile(simulated_pnl, ALPHA)
 
    # Assemble results DataFrame
    result = pd.DataFrame({
        "date":      dates_all,
        "sigma_hat": sigma_arr,
        "var_fhs":   var_arr,
        "z_mean":    [np.nanmean(z_arr[max(0, t-WINDOW):t]) for t in range(n)],
        "z_std":     [np.nanstd( z_arr[max(0, t-WINDOW):t]) for t in range(n)],
    }).set_index("date")
 
    valid = result["var_fhs"].notna().sum()
    print(f"\n[FHS]  Done — {valid} valid VaR estimates produced.")
    return result

# 4.  BACKTESTING  (Kupiec + Christoffersen — mirrors backtest.py)
def kupiec_test(n_exceptions: int, n_obs: int, alpha: float = 0.01):
    """
    Likelihood-ratio test for unconditional coverage (Kupiec 1995).
    H0: exception rate = alpha
    """
    if n_exceptions == 0:
        return np.nan, np.nan
    p_hat = n_exceptions / n_obs
    if p_hat >= 1.0:
        return np.nan, np.nan
    lr = -2 * (
        n_exceptions * np.log(alpha / p_hat)
        + (n_obs - n_exceptions) * np.log((1 - alpha) / (1 - p_hat))
    )
    p_val = 1 - stats.chi2.cdf(lr, df=1)
    return lr, p_val

def christoffersen_test(exceptions: np.ndarray):
    """
    Independence LR test (Christoffersen 1998).
    H0: exception indicator is serially independent (first-order Markov).
    """
    it = exceptions.astype(int)
    # Transition counts
    n00 = np.sum((it[:-1] == 0) & (it[1:] == 0))
    n01 = np.sum((it[:-1] == 0) & (it[1:] == 1))
    n10 = np.sum((it[:-1] == 1) & (it[1:] == 0))
    n11 = np.sum((it[:-1] == 1) & (it[1:] == 1))
 
    # Avoid division by zero
    if (n00 + n01) == 0 or (n10 + n11) == 0:
        return np.nan, np.nan
 
    pi_01 = n01 / (n00 + n01)
    pi_11 = n11 / (n10 + n11)
    pi    = (n01 + n11) / (n00 + n01 + n10 + n11)
 
    def safe_log(x):
        return np.log(x) if x > 0 else 0.0
 
    lr = -2 * (
        (n00 + n10) * safe_log(1 - pi)
        + (n01 + n11) * safe_log(pi)
        - n00 * safe_log(1 - pi_01)
        - n01 * safe_log(pi_01 if pi_01 > 0 else 1e-10)
        - n10 * safe_log(1 - pi_11)
        - n11 * safe_log(pi_11 if pi_11 > 0 else 1e-10)
    )
    p_val = 1 - stats.chi2.cdf(lr, df=1)
    return lr, p_val

def run_backtest(var_series: pd.Series, pnl: pd.Series,
                 alpha: float = 0.01, label: str = "FHS") -> dict:
    """
    Backtest VaR against realised total portfolio P&L.
    Mirrors run_backtest() in backtesting/backtest.py.
 
    An exception occurs when pnl_t < -VaR_t  (loss exceeds predicted VaR).
    """
    common = var_series.dropna().index.intersection(pnl.index)
    var_bt  = var_series.loc[common]
    pnl_bt  = pnl.loc[common]
    T       = len(common)
 
    exceptions  = (pnl_bt < -var_bt).astype(int)
    n_exc       = int(exceptions.sum())
    n_exp       = round(T * alpha, 1)
    exc_rate    = n_exc / T
 
    lr_uc, p_uc = kupiec_test(n_exc, T, alpha)
    lr_ind, p_ind = christoffersen_test(exceptions.values)
 
    # Combined CC statistic
    if not np.isnan(lr_uc) and not np.isnan(lr_ind):
        lr_cc = lr_uc + lr_ind
        p_cc  = 1 - stats.chi2.cdf(lr_cc, df=2)
    else:
        lr_cc, p_cc = np.nan, np.nan
 
    results = {
        "model":    label,
        "T":        T,
        "N_exc":    n_exc,
        "N_exp":    n_exp,
        "exc_rate": round(exc_rate * 100, 3),
        "LR_UC":    round(lr_uc,  4) if not np.isnan(lr_uc)  else np.nan,
        "p_UC":     round(p_uc,   4) if not np.isnan(p_uc)   else np.nan,
        "LR_IND":   round(lr_ind, 4) if not np.isnan(lr_ind) else np.nan,
        "p_IND":    round(p_ind,  4) if not np.isnan(p_ind)  else np.nan,
        "LR_CC":    round(lr_cc,  4) if not np.isnan(lr_cc)  else np.nan,
        "p_CC":     round(p_cc,   4) if not np.isnan(p_cc)   else np.nan,
    }
 
    print(f"\n{'='*60}")
    print(f"  Backtest: {label}")
    print(f"  T={T}  |  Exceptions: {n_exc} obs, {n_exp} exp  |  Rate={exc_rate*100:.2f}%")
    print(f"  Kupiec    LR={lr_uc:.4f}  p={p_uc:.4f}")
    print(f"  Indep.    LR={lr_ind:.4f}  p={p_ind:.4f}")
    print(f"  CC (joint)LR={lr_cc:.4f}  p={p_cc:.4f}")
    print(f"{'='*60}\n")
 
    return results, exceptions, pnl_bt, var_bt


# 5.  DIAGNOSTICS PLOT  (standardised residuals)
def plot_diagnostics(r_port: pd.Series, fhs_result: pd.DataFrame,
                     save_path: str = None):
    """
    Replicate the residual diagnostics from garch_evt_01_diagnostics.png
    but for the FHS GARCH fit.
    """
    # Re-fit on full sample to get residual series for plotting
    print("[Diagnostics]  Fitting GARCH on full sample for diagnostic plots ...")
    model = arch_model(r_port * 100, vol="Garch", p=1, q=1, dist="t",
                       mean="Constant")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = model.fit(disp="off")
 
    z = res.resid / res.conditional_volatility   # standardised residuals
 
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("FHS — GARCH(1,1)-t Diagnostic Plots", fontsize=14, fontweight="bold")
 
    # --- (1) Standardised residuals time series ---
    ax = axes[0, 0]
    ax.plot(r_port.index[-len(z):], z, lw=0.5, color="steelblue", alpha=0.8)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_title("Standardised Residuals  ẑ_t")
    ax.set_ylabel("ẑ_t")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
 
    # --- (2) ACF of |ẑ_t|² (should be flat if GARCH captures clustering) ---
    ax = axes[0, 1]
    z2 = z ** 2
    lags = range(1, 31)
    acf_vals = [pd.Series(z2).autocorr(lag=l) for l in lags]
    ax.bar(lags, acf_vals, color="steelblue", alpha=0.7)
    ax.axhline(0, color="black", lw=0.8)
    ci = 1.96 / np.sqrt(len(z))
    ax.axhline( ci, color="red", ls="--", lw=1, label="95% CI")
    ax.axhline(-ci, color="red", ls="--", lw=1)
    ax.set_title("ACF of  ẑ_t²  (should be near zero)")
    ax.set_xlabel("Lag")
    ax.legend(fontsize=8)
 
    # --- (3) QQ-plot vs Student-t ---
    ax = axes[1, 0]
    nu_hat = res.params.get("nu", 5.0)
    (osm, osr), (slope, intercept, r) = stats.probplot(z, dist=stats.t,
                                                        sparams=(nu_hat,))
    ax.scatter(osm, osr, s=4, alpha=0.4, color="steelblue")
    ax.plot(osm, slope * np.array(osm) + intercept,
            color="red", lw=1.5, label=f"t(ν={nu_hat:.1f}) reference")
    ax.set_title("QQ-Plot: ẑ_t vs Student-t")
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Sample Quantiles")
    ax.legend(fontsize=8)
 
    # --- (4) Conditional volatility σ̂_t ---
    ax = axes[1, 1]
    cv = res.conditional_volatility / 100   # back to decimal
    ax.plot(r_port.index[-len(cv):], cv * 100, lw=0.7,
            color="darkorange", label="σ̂_t (GARCH)")
    ax.set_title("Conditional Volatility  σ̂_t  (%)")
    ax.set_ylabel("σ̂_t (%/day)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(fontsize=8)
 
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Plot]  Saved diagnostics → {save_path}")
    plt.show()
 
    # Summary stats
    print(f"\n[GARCH residuals]  mean={z.mean():.4f}  std={z.std():.4f}  "
          f"skew={stats.skew(z):.4f}  kurt={stats.kurtosis(z):.4f}")
    print(f"  (expected: mean≈0, std≈1)\n")


# 6.  VaR TIME SERIES + EXCEPTIONS PLOT

def plot_var_and_exceptions(fhs_result: pd.DataFrame,
                             pnl: pd.Series,
                             exceptions: pd.Series,
                             var_series: pd.Series,
                             backtest_results: dict,
                             save_path: str = None):
    """
    Two-panel plot:
      Top    — Portfolio P&L vs. −VaR_FHS (compare with your other methods)
      Bottom — VaR time series with conditional volatility overlay
    """
    common = var_series.dropna().index.intersection(pnl.index)
 
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    fig.suptitle(
        f"FHS VaR (99%, 1-day)  |  Exceptions: "
        f"{backtest_results['N_exc']} / {backtest_results['T']} "
        f"(expected {backtest_results['N_exp']})  |  "
        f"Kupiec p={backtest_results['p_UC']:.4f}  "
        f"Christo. p={backtest_results['p_IND']:.4f}",
        fontsize=11, fontweight="bold"
    )
 
    # --- Top panel: P&L vs. -VaR ---
    ax = axes[0]
    ax.plot(common, pnl.loc[common] / 1000, lw=0.6,
            color="steelblue", alpha=0.8, label="Total P&L ($k)")
    ax.plot(common, -var_series.loc[common] / 1000, lw=1.2,
            color="firebrick", label="-VaR FHS ($k)")
 
    # Mark exceptions
    exc_dates = exceptions[exceptions == 1].index
    ax.scatter(exc_dates,
               pnl.loc[exc_dates] / 1000,
               color="red", s=18, zorder=5, label=f"Exceptions (n={len(exc_dates)})")
 
    ax.axhline(0, color="black", lw=0.5, ls="--")
    ax.set_ylabel("P&L ($k)")
    ax.legend(fontsize=9, loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
 
    # Shade GFC and COVID
    for (start, end, label) in [
        ("2008-09-01", "2009-06-01", "GFC"),
        ("2020-02-15", "2020-05-01", "COVID"),
    ]:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                   alpha=0.08, color="grey")
        ax.text(pd.Timestamp(start), ax.get_ylim()[0] * 0.85,
                label, fontsize=7, color="grey")
 
    # --- Bottom panel: VaR over time ---
    ax = axes[1]
    ax.plot(common, var_series.loc[common] / 1000, lw=1.0,
            color="firebrick", label="VaR FHS ($k)")
 
    # Overlay conditional volatility (right axis)
    ax2 = ax.twinx()
    sigma_plot = fhs_result["sigma_hat"].loc[common] * 100   # in %
    ax2.fill_between(common, sigma_plot, alpha=0.15, color="darkorange")
    ax2.plot(common, sigma_plot, lw=0.5, color="darkorange",
             label="σ̂_t GARCH (%)")
    ax2.set_ylabel("σ̂_t (%/day)", color="darkorange")
    ax2.tick_params(axis="y", labelcolor="darkorange")
 
    ax.set_ylabel("VaR ($k)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(fontsize=9, loc="upper left")
    ax2.legend(fontsize=9, loc="upper right")
 
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Plot]  Saved VaR chart → {save_path}")
    plt.show()


# 7.  MAIN

def main():
    print("=" * 65)
    print("  Filtered Historical Simulation (FHS) — VaR Pipeline")
    print("=" * 65)
 
    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------
    r_port, pnl = load_data(RETURNS_PATH, PNL_PATH)
 
    # -----------------------------------------------------------------------
    # Run FHS
    # -----------------------------------------------------------------------
    fhs_result = compute_fhs_var(r_port)
 
    # Save VaR series
    fhs_result.to_csv(OUT_VAR)
    print(f"[Output]  VaR series saved → {OUT_VAR}")
 
    # -----------------------------------------------------------------------
    # Backtest
    # -----------------------------------------------------------------------
    var_series = fhs_result["var_fhs"]
    bt_results, exceptions, pnl_bt, var_bt = run_backtest(
        var_series, pnl, alpha=ALPHA, label="FHS"
    )
 
    # Save backtest table
    pd.DataFrame([bt_results]).to_csv(OUT_BT, index=False)
    print(f"[Output]  Backtest results saved → {OUT_BT}")
 
    # -----------------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------------
    plot_diagnostics(r_port, fhs_result, save_path=FIG_DIAG)
    plot_var_and_exceptions(fhs_result, pnl, exceptions,
                            var_series, bt_results, save_path=FIG_VAR)
 
    # -----------------------------------------------------------------------
    # Print summary table comparable to your existing methods
    # -----------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("  BACKTEST SUMMARY TABLE  (add to your comparison)")
    print("=" * 65)
    print(f"  {'Model':<20} {'T':>6} {'Exc':>5} {'Exp':>5} "
          f"{'Rate%':>7} {'p_UC':>8} {'p_IND':>8} {'p_CC':>8}")
    print("-" * 65)
 
    # Reference results from your project (from Assumption_Overview doc)
    reference = [
        ("EVT",        4269, 56, 42.7, 1.31, 0.0507, 0.0064, 0.0036),
        ("GARCH+EVT",  4269, 41, 42.7, 0.96, 0.7936, 0.4132, 0.6914),
        ("Copula",     4269, 51, 42.7, 1.19, 0.2149, 0.0003, 0.0007),
    ]
    for row in reference:
        print(f"  {row[0]:<20} {row[1]:>6} {row[2]:>5} {row[3]:>5} "
              f"{row[4]:>7.2f} {row[5]:>8.4f} {row[6]:>8.4f} {row[7]:>8.4f}")
 
    r = bt_results
    print(f"  {'FHS (new)':<20} {r['T']:>6} {r['N_exc']:>5} {r['N_exp']:>5} "
          f"{r['exc_rate']:>7.2f} {r['p_UC']:>8.4f} {r['p_IND']:>8.4f} {r['p_CC']:>8.4f}")
    print("=" * 65)
    print("\nDone.\n")
 
 
if __name__ == "__main__":
    main()
