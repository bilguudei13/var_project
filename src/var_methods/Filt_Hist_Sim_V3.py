"""
Filtered Historical Simulation (FHS) Value-at-Risk  —  v3 (daily recursion)
============================================================================
Frankfurt School of Finance & Management
Market Risk Modelling Project — VaR Calculation and Evolution

Fixes vs. v1 and v2:
    v1 — GARCH fitted on linear log-returns only → 439 exceptions (10.41%).
         Fixed in v2 by fitting on r_total = pnl_total / V0.

    v2 — σ̂_t was constant for 50 days between refits because
         res_current.forecast() always returns the forecast from the
         last refit date. VaR appeared as a staircase (visible in plots).
         Also z_insample was only updated at refit, not daily.
         Fixed in v3 by the daily GARCH variance recursion.

Method (v3):
    Step 1 — Fit GARCH(1,1)-t on expanding window of r_total = pnl/V0.
             Extract parameters ω, α, β, μ and last in-sample σ².
             Refit every REFIT_EVERY=50 days (expanding window).
    Step 2 — Daily GARCH recursion (between refits, zero cost):
             σ²_t = ω + α·(r_{t-1}−μ)² + β·σ²_{t-1}   [all in % units]
    Step 3 — Append daily ẑ_{t-1} = (r_{t-1}−μ)/σ̂_{t-1} to rolling buffer.
    Step 4 — FHS scenarios: r̃_i = ẑ_i × σ̂_t  (last WINDOW residuals).
    Step 5 — Simulated P&L: PnL_i = V0 × r̃_i.
    Step 6 — VaR_t = −quantile(PnL_i, 0.01).

References:
    Barone-Adesi, Giannopoulos, Vosper (1999) — original FHS paper
    McNeil, Frey, Embrechts (2005) §2 — GARCH pre-filtering motivation
    Irle lecture notes, Section 8.3
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from arch import arch_model
from scipy import stats

# ---------------------------------------------------------------------------
# Configuration  — mirrors garch_evt.py and config.py
# ---------------------------------------------------------------------------
WINDOW       = 500    # residual look-back window (trading days)
REFIT_EVERY  = 50     # re-estimate GARCH every N days (expanding window)
CONFIDENCE   = 0.99   # VaR confidence level
ALPHA        = 1 - CONFIDENCE          # 0.01
V0           = 1_000_000               # portfolio notional ($)

# Paths
PNL_PATH  = "data/processed/total_portfolio_pnl.csv"
OUT_VAR   = "data/processed/var_fhs.csv"
OUT_BT    = "outputs/tables/backtest_fhs.csv"
FIG_DIAG  = "outputs/figures/fhs_01_diagnostics.png"
FIG_VAR   = "outputs/figures/fhs_02_var_and_exceptions.png"


# ===========================================================================
# 1.  DATA LOADING
# ===========================================================================

def load_data(pnl_path: str):
    """
    Load total portfolio P&L and derive the total portfolio return series
    r_total = pnl_total / V0.

    Using r_total (not linear log-returns alone) ensures GARCH captures
    the full portfolio dynamics including IRS and straddle.
    """
    pnl_df = pd.read_csv(pnl_path, index_col=0, parse_dates=True)
    pnl    = pnl_df["pnl_total"] if "pnl_total" in pnl_df.columns \
             else pnl_df.iloc[:, 0]

    # Total portfolio return (decimal) — the series GARCH is fitted on
    r_total = pnl / V0
    r_total.name = "r_total"

    print(f"[Data]  {len(pnl)} trading days  "
          f"({pnl.index[0].date()} → {pnl.index[-1].date()})")
    print(f"        r_total: mean={r_total.mean()*100:.4f}%  "
          f"std={r_total.std()*100:.4f}%  "
          f"min={r_total.min()*100:.3f}%  max={r_total.max()*100:.3f}%\n")
    return r_total, pnl


# ===========================================================================
# 2.  GARCH(1,1)-t FIT
# ===========================================================================

def fit_garch(train: pd.Series):
    """
    Fit GARCH(1,1) with Student-t innovations.

    Parameters
    ----------
    train : total portfolio return series (decimal), e.g. pnl/V0

    Returns
    -------
    res    : arch ModelResult (in % units internally)
    params : dict with mu, alpha1, beta, nu in natural units
    """
    train_pct = train * 100.0   # arch needs % for numerical stability (F.2)

    model = arch_model(
        train_pct,
        vol="Garch", p=1, q=1,
        dist="t",
        mean="Constant",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = model.fit(disp="off", show_warning=False)

    p      = res.params
    mu     = p["mu"] / 100.0
    alpha1 = p.get("alpha[1]", p.get("alpha", np.nan))
    beta   = p.get("beta[1]",  p.get("beta",  np.nan))
    nu     = p.get("nu", 5.0)
    ab     = alpha1 + beta

    print(f"   GARCH refit  μ={mu*100:.4f}%  α+β={ab:.4f} "
          f"{'OK' if ab < 1 else '!! UNIT ROOT'}  ν={nu:.2f}")

    return res, {"mu": mu, "alpha1": alpha1, "beta": beta, "nu": nu, "ab": ab}


# ===========================================================================
# 3.  MAIN FHS LOOP
# ===========================================================================

def compute_fhs_var(r_total: pd.Series) -> pd.DataFrame:
    """
    Expanding-window FHS VaR — v3 (daily GARCH recursion).

    Bug fixed vs. v2:
        v2 called res_current.forecast() every day, which returns the SAME
        one-step-ahead forecast from the last refit date — σ̂_t was constant
        for 50 days at a time (visible as a staircase in the VaR plot).

        v3 applies the GARCH variance recursion manually every day:
            σ²_t = ω + α·(r_{t-1} − μ)² + β·σ²_{t-1}
        using the fixed parameters from the last refit. This gives a truly
        time-varying σ̂_t between refits at zero extra fitting cost.

        Similarly, the standardised-residual buffer z_buf is updated daily:
            ẑ_{t-1} = (r_{t-1} − μ) / σ̂_{t-1}
        so the empirical shock distribution always contains the most recent
        500 observations, not just those from the last refit.

    Algorithm per day t (after burn-in):
        (a) Refit GARCH every REFIT_EVERY days — extract ω, α, β, μ, and
            initialise σ²_prev from the last in-sample conditional variance.
        (b) Daily GARCH recursion: σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
            where ε_{t-1} = (r_{t-1} − μ) × 100  (in % units).
        (c) Append ẑ_{t-1} = ε_{t-1} / σ̂_{t-1} to rolling z-buffer.
        (d) FHS scenarios: r̃_i = ẑ_i × σ̂_t  for last WINDOW residuals.
        (e) VaR_t = −quantile(V0 · r̃_i, 0.01).

    Parameters
    ----------
    r_total : pd.Series — total portfolio return = pnl_total / V0 (decimal)

    Returns
    -------
    pd.DataFrame with columns: sigma_hat, var_fhs
    """
    n         = len(r_total)
    dates_all = r_total.index
    r_vals    = r_total.values          # numpy array, decimal

    var_arr   = np.full(n, np.nan)
    sigma_arr = np.full(n, np.nan)

    # GARCH parameters (updated at each refit, held fixed between refits)
    omega, alpha1, beta1, mu = 0.0, 0.0, 0.0, 0.0

    # σ²_{t-1} in (%)² — propagated daily via recursion
    sigma_sq_prev = np.nan

    # Rolling buffer of standardised residuals ẑ_t (dimensionless)
    # We use a deque-like list and keep at most WINDOW entries
    z_buf = []

    res_current  = None
    last_refit_t = WINDOW

    print(f"[FHS]  {n - WINDOW} backtest days  "
          f"(burn-in={WINDOW}, refit every {REFIT_EVERY} days)\n")

    for t in range(WINDOW, n):

        # ------------------------------------------------------------------
        # (a) Refit GARCH on expanding window when due
        # ------------------------------------------------------------------
        if res_current is None or (t - last_refit_t) >= REFIT_EVERY:
            train = r_total.iloc[:t]
            res_current, params = fit_garch(train)
            last_refit_t = t

            # Extract GARCH parameters (all in % units — arch convention)
            p      = res_current.params
            omega  = p.get("omega",   p.get("Omega",  np.nan))   # ω in (%)²
            alpha1 = p.get("alpha[1]",p.get("alpha",  np.nan))   # α
            beta1  = p.get("beta[1]", p.get("beta",   np.nan))   # β
            mu     = p["mu"]                                      # μ in %

            # Initialise σ²_prev from the last in-sample conditional variance
            # conditional_volatility is in %; square it for the recursion
            sigma_sq_prev = float(
                res_current.conditional_volatility.iloc[-1] ** 2
            )

            # Rebuild z_buf from in-sample residuals at this refit point
            # Use the last WINDOW in-sample standardised residuals
            z_insample = (res_current.resid /
                          res_current.conditional_volatility).values
            z_buf = list(z_insample[-WINDOW:])   # at most WINDOW entries

        # ------------------------------------------------------------------
        # (b) Daily GARCH recursion — σ²_t from yesterday's return
        #     All quantities in % units to match arch convention
        # ------------------------------------------------------------------
        r_prev_pct = r_vals[t - 1] * 100.0          # r_{t-1} in %
        eps_prev   = r_prev_pct - mu                 # ε_{t-1} in %

        sigma_sq_t = omega + alpha1 * (eps_prev ** 2) + beta1 * sigma_sq_prev
        sigma_sq_t = max(sigma_sq_t, 1e-8)          # numerical floor

        sigma_hat_t   = np.sqrt(sigma_sq_t) / 100.0  # back to decimal
        sigma_arr[t]  = sigma_hat_t

        # ------------------------------------------------------------------
        # (c) Append today's standardised residual to the rolling buffer
        #     ẑ_{t-1} = ε_{t-1} / σ̂_{t-1}   (dimensionless)
        # ------------------------------------------------------------------
        sigma_prev = np.sqrt(max(sigma_sq_prev, 1e-8))   # in %
        z_new = eps_prev / sigma_prev
        if np.isfinite(z_new):
            z_buf.append(z_new)
            if len(z_buf) > WINDOW:
                z_buf.pop(0)          # maintain rolling window

        # Update σ²_prev for next iteration
        sigma_sq_prev = sigma_sq_t

        # ------------------------------------------------------------------
        # (d) FHS scenarios — scale last WINDOW residuals by today's σ̂_t
        # ------------------------------------------------------------------
        z_window = np.array(z_buf)
        z_window = z_window[np.isfinite(z_window)]

        if len(z_window) < 50:
            continue   # not enough history yet

        sim_returns = z_window * sigma_hat_t    # decimal portfolio returns

        # ------------------------------------------------------------------
        # (e) VaR = negative of the 1st percentile of simulated P&L
        # ------------------------------------------------------------------
        sim_pnl    = V0 * sim_returns
        var_arr[t] = -np.quantile(sim_pnl, ALPHA)

    result = pd.DataFrame(
        {"sigma_hat": sigma_arr, "var_fhs": var_arr},
        index=dates_all
    )

    n_valid = result["var_fhs"].notna().sum()
    print(f"\n[FHS]  Done — {n_valid} valid VaR estimates.\n")
    return result


# ===========================================================================
# 4.  BACKTESTING  (Kupiec + Christoffersen)
# ===========================================================================

def kupiec_test(n_exc, n_obs, alpha=0.01):
    if n_exc == 0 or n_exc >= n_obs:
        return np.nan, np.nan
    p_hat = n_exc / n_obs
    lr = -2 * (
        n_exc * np.log(alpha / p_hat)
        + (n_obs - n_exc) * np.log((1 - alpha) / (1 - p_hat))
    )
    return lr, 1 - stats.chi2.cdf(lr, df=1)


def christoffersen_test(exc: np.ndarray):
    it = exc.astype(int)
    n00 = np.sum((it[:-1]==0)&(it[1:]==0))
    n01 = np.sum((it[:-1]==0)&(it[1:]==1))
    n10 = np.sum((it[:-1]==1)&(it[1:]==0))
    n11 = np.sum((it[:-1]==1)&(it[1:]==1))
    if (n00+n01)==0 or (n10+n11)==0:
        return np.nan, np.nan
    pi01 = n01/(n00+n01)
    pi11 = n11/(n10+n11) if (n10+n11) > 0 else 0
    pi   = (n01+n11)/(n00+n01+n10+n11)
    def sl(x): return np.log(x) if x > 0 else 0.0
    lr = -2*(
        (n00+n10)*sl(1-pi) + (n01+n11)*sl(pi)
        - n00*sl(1-pi01) - n01*sl(pi01 if pi01>0 else 1e-10)
        - n10*sl(1-pi11) - n11*sl(pi11 if pi11>0 else 1e-10)
    )
    return lr, 1 - stats.chi2.cdf(lr, df=1)


def run_backtest(var_series: pd.Series, pnl: pd.Series,
                 alpha=0.01, label="FHS"):
    common   = var_series.dropna().index.intersection(pnl.index)
    var_bt   = var_series.loc[common]
    pnl_bt   = pnl.loc[common]
    T        = len(common)
    exc      = (pnl_bt < -var_bt).astype(int)
    n_exc    = int(exc.sum())
    n_exp    = round(T * alpha, 1)

    lr_uc, p_uc   = kupiec_test(n_exc, T, alpha)
    lr_ind, p_ind = christoffersen_test(exc.values)
    lr_cc  = (lr_uc  + lr_ind)  if not (np.isnan(lr_uc) or np.isnan(lr_ind)) else np.nan
    p_cc   = (1 - stats.chi2.cdf(lr_cc, 2)) if not np.isnan(lr_cc) else np.nan

    res = dict(model=label, T=T, N_exc=n_exc, N_exp=n_exp,
               exc_rate=round(n_exc/T*100, 3),
               p_UC=round(p_uc,4), p_IND=round(p_ind,4), p_CC=round(p_cc,4))

    print(f"\n{'='*60}")
    print(f"  Backtest: {label}")
    print(f"  T={T} | Exceptions: {n_exc} obs, {n_exp} exp | "
          f"Rate={n_exc/T*100:.2f}%")
    print(f"  Kupiec  p={p_uc:.4f}  |  Indep. p={p_ind:.4f}  |  "
          f"CC p={p_cc:.4f}")
    print(f"{'='*60}\n")

    return res, exc, pnl_bt, var_bt


# ===========================================================================
# 5.  DIAGNOSTIC PLOT
# ===========================================================================

def plot_diagnostics(r_total: pd.Series, save_path=None):
    print("[Diagnostics]  Fitting GARCH on full sample ...")
    model = arch_model(r_total * 100, vol="Garch", p=1, q=1,
                       dist="t", mean="Constant")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = model.fit(disp="off")

    z  = (res.resid / res.conditional_volatility).values
    nu = res.params.get("nu", 5.0)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("FHS — GARCH(1,1)-t Diagnostics  (fitted on pnl_total/V0)",
                 fontsize=13, fontweight="bold")

    # Standardised residuals
    ax = axes[0, 0]
    idx = r_total.index[-len(z):]
    ax.plot(idx, z, lw=0.5, color="steelblue", alpha=0.8)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_title("Standardised Residuals  ẑ_t")
    ax.set_ylabel("ẑ_t")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # ACF of ẑ²
    ax = axes[0, 1]
    z2 = z**2
    lags = range(1, 31)
    acf  = [pd.Series(z2).autocorr(lag=l) for l in lags]
    ax.bar(lags, acf, color="steelblue", alpha=0.7)
    ci = 1.96 / np.sqrt(len(z))
    ax.axhline( ci, color="red", ls="--", lw=1, label="95% CI")
    ax.axhline(-ci, color="red", ls="--", lw=1)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("ACF of  ẑ_t²  (should be near zero)")
    ax.set_xlabel("Lag")
    ax.legend(fontsize=8)

    # QQ-plot
    ax = axes[1, 0]
    (osm, osr), (slope, intercept, _) = stats.probplot(
        z, dist=stats.t, sparams=(nu,))
    ax.scatter(osm, osr, s=4, alpha=0.4, color="steelblue")
    ax.plot(osm, slope*np.array(osm)+intercept,
            color="red", lw=1.5, label=f"t(ν={nu:.1f})")
    ax.set_title("QQ-Plot: ẑ_t vs Student-t")
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Sample Quantiles")
    ax.legend(fontsize=8)

    # Conditional volatility
    ax = axes[1, 1]
    cv = res.conditional_volatility.values / 100
    ax.plot(idx, cv*100, lw=0.7, color="darkorange")
    ax.set_title("Conditional Volatility  σ̂_t  (%/day)")
    ax.set_ylabel("σ̂_t (%)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Plot]  Saved → {save_path}")
    plt.show()
    print(f"  Residual mean={z.mean():.4f}  std={z.std():.4f}  "
          f"skew={stats.skew(z):.4f}  kurt={stats.kurtosis(z):.4f}\n")


# ===========================================================================
# 6.  VaR + EXCEPTIONS PLOT
# ===========================================================================

def plot_var_and_exceptions(fhs_result, pnl, exc, var_series,
                             bt_res, save_path=None):
    common = var_series.dropna().index.intersection(pnl.index)

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    fig.suptitle(
        f"FHS VaR (99%, 1-day)  |  Exceptions: {bt_res['N_exc']} / "
        f"{bt_res['T']}  (exp {bt_res['N_exp']})  |  "
        f"Kupiec p={bt_res['p_UC']:.4f}   Christo. p={bt_res['p_IND']:.4f}",
        fontsize=10, fontweight="bold"
    )

    # P&L vs -VaR
    ax = axes[0]
    ax.plot(common, pnl.loc[common]/1000, lw=0.6,
            color="steelblue", alpha=0.8, label="Total P&L ($k)")
    ax.plot(common, -var_series.loc[common]/1000, lw=1.2,
            color="firebrick", label="-VaR FHS ($k)")
    exc_dates = exc[exc==1].index
    ax.scatter(exc_dates, pnl.loc[exc_dates]/1000,
               color="red", s=18, zorder=5,
               label=f"Exceptions (n={len(exc_dates)})")
    ax.axhline(0, color="black", lw=0.5, ls="--")
    ax.set_ylabel("P&L ($k)")
    ax.legend(fontsize=9, loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for start, end, lbl in [("2008-09-01","2009-06-01","GFC"),
                              ("2020-02-15","2020-05-01","COVID")]:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                   alpha=0.08, color="grey")

    # VaR time series
    ax  = axes[1]
    ax2 = ax.twinx()
    ax.plot(common, var_series.loc[common]/1000, lw=1.0,
            color="firebrick", label="VaR FHS ($k)")
    sig = fhs_result["sigma_hat"].loc[common] * 100
    ax2.fill_between(common, sig, alpha=0.15, color="darkorange")
    ax2.plot(common, sig, lw=0.5, color="darkorange", label="σ̂_t GARCH (%)")
    ax.set_ylabel("VaR ($k)")
    ax2.set_ylabel("σ̂_t (%/day)", color="darkorange")
    ax2.tick_params(axis="y", labelcolor="darkorange")
    ax.legend(fontsize=9, loc="upper left")
    ax2.legend(fontsize=9, loc="upper right")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Plot]  Saved → {save_path}")
    plt.show()


# ===========================================================================
# 7.  MAIN
# ===========================================================================

def main():
    print("=" * 65)
    print("  FHS VaR Pipeline  —  v2 (total portfolio return)")
    print("=" * 65 + "\n")

    # Load
    r_total, pnl = load_data(PNL_PATH)

    # Compute VaR
    fhs_result = compute_fhs_var(r_total)
    fhs_result.to_csv(OUT_VAR)
    print(f"[Output]  VaR saved → {OUT_VAR}")

    # Backtest
    var_series = fhs_result["var_fhs"]
    bt_res, exc, pnl_bt, var_bt = run_backtest(
        var_series, pnl, alpha=ALPHA, label="FHS"
    )
    pd.DataFrame([bt_res]).to_csv(OUT_BT, index=False)
    print(f"[Output]  Backtest saved → {OUT_BT}")

    # Plots
    plot_diagnostics(r_total, save_path=FIG_DIAG)
    plot_var_and_exceptions(fhs_result, pnl, exc,
                            var_series, bt_res, save_path=FIG_VAR)

    # Comparison table
    print("\n" + "=" * 68)
    print("  COMPARISON TABLE")
    print(f"  {'Model':<16} {'T':>5} {'Exc':>5} {'Exp':>5} "
          f"{'Rate%':>7} {'p_UC':>8} {'p_IND':>8} {'p_CC':>8}")
    print("-" * 68)
    refs = [
        ("EVT",       4269, 56, 42.7, 1.31, 0.0507, 0.0064, 0.0036),
        ("GARCH+EVT", 4269, 41, 42.7, 0.96, 0.7936, 0.4132, 0.6914),
        ("Copula",    4269, 51, 42.7, 1.19, 0.2149, 0.0003, 0.0007),
    ]
    for r in refs:
        print(f"  {r[0]:<16} {r[1]:>5} {r[2]:>5} {r[3]:>5} "
              f"{r[4]:>7.2f} {r[5]:>8.4f} {r[6]:>8.4f} {r[7]:>8.4f}")
    r = bt_res
    print(f"  {'FHS (v2)':<16} {r['T']:>5} {r['N_exc']:>5} "
          f"{r['N_exp']:>5} {r['exc_rate']:>7.2f} "
          f"{r['p_UC']:>8.4f} {r['p_IND']:>8.4f} {r['p_CC']:>8.4f}")
    print("=" * 68 + "\n")


if __name__ == "__main__":
    main()