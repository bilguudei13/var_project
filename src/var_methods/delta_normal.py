# =============================================================================
# delta_normal.py
# 1-day 99% Delta-Normal VaR — aligned with Irle lecture notes
#
# Theory references (Irle lecture notes):
#
# [Section 2] Risk mapping for a linear portfolio V = sum(a_i * S_i):
#   Delta V ≈ sum_i [ a_i * S_i_0 * R_i ]   (delta equivalents, Eq. 5-6)
#           = V0 * sum_i [ w_i * R_i ]
#           = V0 * R^{d,P}
#
# [Section 4, Eq. 10] Discrete returns are additive across the portfolio:
#   R^{d,P} = sum_i w_i * R^{d,i}
#   Note: log-returns ≈ discrete returns for small dt (1 day), Irle page 35
#
# [Section 5.4, Eq. 14] VaR for normally distributed value changes:
#   Delta V ~ N(mu, sigma^2)
#   VaR_{alpha} = sigma * N_alpha - mu        (full formula)
#
# [Section 5.4, Eq. 15] Ignoring drift (standard for 1-day horizon):
#   VaR_{alpha} = N_alpha * sigma
#   For alpha=99%: N_alpha = 2.3263  (Irle table, page 56)
#
# [Section 5.4] Portfolio variance:
#   sigma^2_{Delta V} = V0^2 * w' * Sigma * w
#
# [Section 8, page 183] Backtesting: N ~ B(T, 1-alpha)
#
# Rolling window : 500 trading days
# Confidence     : alpha = 99%  =>  N_alpha = 2.3263
# Horizon        : 1 day
# =============================================================================

import os
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# =============================================================================
# SETTINGS
# =============================================================================

WINDOW  = 500
ALPHA   = 0.99
N_ALPHA = norm.ppf(ALPHA)   # = 2.3263  (Irle table, page 56)
V0      = 1_000_000
WEIGHTS = np.ones(6) / 6    # equal weights w_i = 1/6

PROCESSED_DIR = os.path.join("data", "processed")
OUTPUT_FIGS   = os.path.join("outputs", "figures")
OUTPUT_TABLES = os.path.join("outputs", "tables")

# =============================================================================
# STEP 1 — LOAD DATA
# =============================================================================

def load_data():
    """
    Load log-returns as approximation of discrete returns R^{d,i}.
    Valid for 1-day horizon (Irle page 35: log ≈ discrete for small dt).
    """
    returns = pd.read_csv(
        os.path.join(PROCESSED_DIR, "log_returns.csv"),
        index_col=0, parse_dates=True
    )
    print(f"Loaded: {returns.shape[0]} days x {returns.shape[1]} assets")
    print(f"Risk factors Y: {list(returns.columns)}")
    return returns

# =============================================================================
# STEP 2 — ROLLING DELTA-NORMAL VaR
# =============================================================================

def compute_delta_normal_var(returns, weights, window, n_alpha, V0):
    """
    Rolling 1-day Delta-Normal VaR (Irle Section 5.4).

    For each day t, estimate from rolling window of k=500 obs:
      mu_R   = sample mean of risk factor returns       (d-vector)
      Sigma  = sample covariance matrix of returns      (d x d)

    Then (Section 5.4):
      mu_{DV}    = V0 * w' * mu_R
      sigma_{DV} = V0 * sqrt(w' * Sigma * w)

    Eq.14 (full):     VaR = sigma_{DV} * N_alpha - mu_{DV}
    Eq.15 (no drift): VaR = N_alpha * sigma_{DV}

    Drift is negligible at 1-day horizon (Irle page 57).
    """
    n = len(returns)
    w = weights.reshape(-1, 1)  # (d x 1) column vector
    records, dates = [], []

    print(f"\n{'='*60}")
    print(f"Delta-Normal VaR  —  Irle Section 5.4")
    print(f"k={window} days | alpha={ALPHA:.0%} | N_alpha={n_alpha:.4f}")
    print(f"V0=${V0:,.0f} | weights={weights.round(4)}")
    print(f"Computing VaR for {n - window} days...")

    for t in range(window, n):
        W = returns.iloc[t - window : t]       # (k x d) window
        mu_R  = W.mean().values                # d-vector
        Sigma = W.cov().values                 # (d x d)

        # mu_{Delta V} = V0 * w' * mu_R
        mu_dv    = V0 * float(np.dot(weights, mu_R))

        # sigma_{Delta V} = V0 * sqrt(w' * Sigma * w)
        sigma_dv = V0 * np.sqrt(float(weights @ Sigma @ weights))

        # Eq.14: VaR = sigma * N_alpha - mu
        VaR_full     = sigma_dv * n_alpha - mu_dv
        # Eq.15: VaR = N_alpha * sigma  (mu = 0)
        VaR_no_drift = sigma_dv * n_alpha

        records.append({
            "VaR_full"     : VaR_full,
            "VaR_no_drift" : VaR_no_drift,
            "mu_deltaV"    : mu_dv,
            "sigma_deltaV" : sigma_dv,
        })
        dates.append(returns.index[t])

    results = pd.DataFrame(records, index=dates)

    print(f"\n{'':25s} {'Eq.14 (full)':>14s} {'Eq.15 (no drift)':>16s}")
    for label, col1, col2 in [
        ("Mean VaR", "VaR_full", "VaR_no_drift"),
        ("Min VaR",  "VaR_full", "VaR_no_drift"),
        ("Max VaR",  "VaR_full", "VaR_no_drift"),
    ]:
        f = {"Mean VaR": "mean", "Min VaR": "min", "Max VaR": "max"}[label]
        v1 = getattr(results[col1], f)()
        v2 = getattr(results[col2], f)()
        print(f"  {label:23s} ${v1:>12,.0f} ${v2:>14,.0f}")

    print(f"\nDrift negligible at 1-day (Irle p.57). Using Eq.14 going forward.")
    return results

# =============================================================================
# STEP 3 — BACKTESTING
# =============================================================================

def backtest(var_series, returns, weights, V0):
    """
    Count VaR exceptions (Irle page 183-184).

    Exception: actual loss -Delta V_t > VaR_t
    Under H0 (correct model): N ~ B(T, 1-alpha) = B(T, 0.01)
    """
    # Delta V = V0 * R^{d,P} = V0 * w' * R  (Irle Eq.10)
    pnl         = V0 * returns.dot(weights)
    pnl_aligned = pnl.reindex(var_series.index)
    actual_loss = -pnl_aligned
    exceptions  = actual_loss > var_series

    T        = len(exceptions)
    N_exc    = int(exceptions.sum())
    exc_rate = N_exc / T
    p        = 1 - ALPHA   # 0.01

    # 95% Binomial CI
    lower = int(np.floor(T * p - 1.96 * np.sqrt(T * p * (1 - p))))
    upper = int(np.ceil( T * p + 1.96 * np.sqrt(T * p * (1 - p))))

    print(f"\n{'='*60}")
    print(f"BACKTESTING  —  Irle page 183-184")
    print(f"{'='*60}")
    print(f"T (observations)        : {T}")
    print(f"Expected E[N] = T*0.01  : {T*p:.1f}")
    print(f"95% CI  B(T, 0.01)      : [{lower}, {upper}]")
    print(f"Actual exceptions N     : {N_exc}")
    print(f"Actual rate             : {exc_rate:.2%}")
    print(f"Within 95% CI?          : {'YES ✓' if lower <= N_exc <= upper else 'NO ✗'}")

    if N_exc > upper:
        print(f"\n→ VaR UNDERESTIMATES risk (too many exceptions)")
        print(f"  Consistent with fat tails violating normality (KS test)")
    elif N_exc < lower:
        print(f"\n→ VaR OVERESTIMATES risk (too few exceptions)")
    else:
        print(f"\n→ VaR adequately calibrated")

    return exceptions, pnl_aligned

# =============================================================================
# STEP 4 — PLOT
# =============================================================================

def plot_var(results, pnl, exceptions):
    var_full     = results["VaR_full"]
    var_no_drift = results["VaR_no_drift"]
    actual_loss  = -pnl

    crises = [
        ("2008-09-15", "2009-03-09", "GFC 2008"),
        ("2020-02-19", "2020-03-23", "COVID 2020"),
        ("2022-01-01", "2022-10-01", "Rate Hikes"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # Panel 1: VaR vs actual loss
    axes[0].fill_between(var_full.index, 0, var_full,
                         alpha=0.15, color="#2196F3")
    axes[0].plot(var_full.index, var_full, color="#2196F3",
                 linewidth=1.2, label="Delta-Normal VaR 99% (Eq.14)")
    axes[0].plot(actual_loss.index, actual_loss, color="#90A4AE",
                 linewidth=0.6, alpha=0.7, label="Actual loss (−ΔV)")
    breaches = exceptions[exceptions].index
    axes[0].scatter(breaches, actual_loss.loc[breaches],
                    color="#F44336", s=15, zorder=5,
                    label=f"Exceptions (N={len(breaches)})")
    axes[0].axhline(0, color="black", linewidth=0.5, linestyle="--")
    axes[0].set_title("Delta-Normal VaR vs Actual Loss | Irle Eq.14",
                      fontsize=12, fontweight="bold")
    axes[0].set_ylabel("USD"); axes[0].legend(fontsize=9)

    # Panel 2: Eq.14 vs Eq.15
    axes[1].plot(var_full.index, var_full, color="#2196F3",
                 linewidth=1.2, label="Eq.14: σ·Nα − μ  (with drift)")
    axes[1].plot(var_no_drift.index, var_no_drift, color="#FF9800",
                 linewidth=1.0, linestyle="--",
                 label="Eq.15: Nα·σ  (drift = 0)")
    axes[1].set_title("Eq.14 vs Eq.15 — Drift negligible at 1-day horizon (Irle p.57)",
                      fontsize=12, fontweight="bold")
    axes[1].set_ylabel("VaR (USD)"); axes[1].legend(fontsize=9)

    # Panel 3: sigma_{Delta V}
    axes[2].plot(results.index, results["sigma_deltaV"],
                 color="#4CAF50", linewidth=1.0)
    axes[2].set_title("Rolling σ(ΔV) — Volatility clustering visible in crises",
                      fontsize=12, fontweight="bold")
    axes[2].set_ylabel("σ(ΔV) (USD)")
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    for ax in axes:
        for s, e, lbl in crises:
            ax.axvspan(s, e, alpha=0.08, color="red")

    plt.tight_layout()
    path = os.path.join(OUTPUT_FIGS, "06_delta_normal_var.png")
    plt.savefig(path, dpi=150); plt.show()
    print(f"\nPlot saved → {path}")

# =============================================================================
# STEP 5 — SAVE
# =============================================================================

def save_results(results, exceptions, pnl):
    results.to_csv(os.path.join(PROCESSED_DIR, "var_delta_normal.csv"))
    print(f"VaR saved  → {os.path.join(PROCESSED_DIR, 'var_delta_normal.csv')}")

    pd.DataFrame({
        "VaR"         : results["VaR_full"].values,
        "actual_loss" : (-pnl).values,
        "exception"   : exceptions.values,
    }, index=exceptions.index).to_csv(
        os.path.join(OUTPUT_TABLES, "backtest_delta_normal.csv")
    )
    print(f"Backtest → {os.path.join(OUTPUT_TABLES, 'backtest_delta_normal.csv')}")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print(f"N_alpha = {N_ALPHA:.4f}  (Irle table page 56: N_0.99 = 2.3263)")
    os.makedirs(OUTPUT_FIGS,   exist_ok=True)
    os.makedirs(OUTPUT_TABLES, exist_ok=True)

    returns    = load_data()
    results    = compute_delta_normal_var(returns, WEIGHTS, WINDOW, N_ALPHA, V0)
    exc, pnl   = backtest(results["VaR_full"], returns, WEIGHTS, V0)
    plot_var(results, pnl, exc)
    save_results(results, exc, pnl)

    print(f"\n{'='*60}\nDelta-Normal VaR complete!\n{'='*60}")