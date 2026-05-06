"""
sweep_refit_frequency.py
========================
Grid-search over REFIT_EVERY values for the GARCH(1,1)-t copula VaR model.

Runs the full walk-forward loop with WINDOW=750 fixed and three different
REFIT_EVERY values in parallel. Prints a side-by-side comparison of
Kupiec + Christoffersen backtest results.

Usage
-----
    python src/var_methods/sweep_refit_frequency.py

Candidate refit frequencies (trading days):
    10  — very reactive, ~5x more fits than baseline
    25  — moderately reactive
    50  — lecture-specified baseline

WINDOW is fixed at 750 (best result from window sweep).
All other settings (M, ALPHA, EWMA_LAMBDA, SEED) are inherited from
mc_garch_t_copula.py unchanged.
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# ── Path setup ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent   # var_project/
_HERE = Path(__file__).resolve().parent
for p in (_ROOT / "src" / "var_methods", _ROOT / "src" / "data",
          _ROOT / "backtesting", _ROOT, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# ── Sweep config ──────────────────────────────────────────────────────────────
FIXED_WINDOW      = 750          # best result from window sweep
REFITS_TO_TEST    = [10, 25, 50] # trading days between parameter refits


# =============================================================================
# WORKER  — runs the full rolling loop for one refit frequency
# Called in a separate process by ProcessPoolExecutor
# =============================================================================

def _run_one_refit(refit_every: int) -> dict:
    """
    Full walk-forward GARCH-copula VaR loop with WINDOW=750 and a given
    REFIT_EVERY. Returns a dict of backtest statistics.
    """
    import sys, warnings, numpy as np
    from pathlib import Path

    _ROOT = Path(__file__).resolve().parent.parent   # var_project/
    _HERE = Path(__file__).resolve().parent
    for p in (_ROOT / "src" / "var_methods", _ROOT / "src" / "data",
              _ROOT / "backtesting", _ROOT, _HERE):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

    import mc_garch_t_copula as v2
    from backtesting.backtest import run_backtest
    from config import V0, WEIGHTS_DICT

    window = FIXED_WINDOW

    # ── Load data ─────────────────────────────────────────────────────────────
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        factors, pnl_series, prices, vix_series, dgs10_series = v2.load_data()

    returns_arr = factors.values
    dates       = factors.index
    T           = len(returns_arr)

    LINEAR_TICKERS = ["EURUSD", "GLD", "IEF", "SPY"]
    prices_linear  = prices.rename(columns={"EURUSD=X": "EURUSD"})[LINEAR_TICKERS]
    prices_linear  = prices_linear.reindex(dates, method="ffill")
    weights        = pd.Series(
        {("EURUSD" if k == "EURUSD=X" else k): v_ for k, v_ in WEIGHTS_DICT.items()}
    )[LINEAR_TICKERS]
    linear_shares  = (V0 * weights / prices_linear.iloc[0]).values
    spy_prices     = prices["SPY"].reindex(dates, method="ffill")

    # ── Rolling loop ──────────────────────────────────────────────────────────
    rng            = np.random.default_rng(v2.SEED)
    var_arr        = np.full(T, np.nan)
    omega_arr      = alpha_arr = beta_arr = mu_arr = nu_marginals = sigma_current = None
    nu_copula_cached = None
    Q_ewma           = None
    n_refits         = 0

    straddle_state = v2.build_straddle_state(spy_prices, straddle_days=v2.STRADDLE_DAYS)

    for t in range(window, T):
        need_refit = (omega_arr is None) or ((t - window) % refit_every == 0)

        if need_refit:
            W = returns_arr[t - window : t]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                std_resids, garch_params = v2.fit_garch_marginals(W)

            omega_arr     = np.array([p["omega"]      for p in garch_params])
            alpha_arr     = np.array([p["alpha"]      for p in garch_params])
            beta_arr      = np.array([p["beta"]       for p in garch_params])
            mu_arr        = np.array([p["mu"]         for p in garch_params])
            nu_marginals  = np.array([p["nu"]         for p in garch_params])
            sigma_current = np.array([p["sigma_last"] for p in garch_params])

            U_resid = v2.compute_pseudo_observations(std_resids)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                R_static, nu_copula_cached, _ = v2.fit_t_copula(U_resid)
            n_refits += 1

            if Q_ewma is None:
                Q_ewma = np.cov(std_resids.T) + 1e-8 * np.eye(v2.N_ASSETS)
            else:
                Q_warm = np.cov(std_resids.T) + 1e-8 * np.eye(v2.N_ASSETS)
                for _z_row in std_resids:
                    Q_warm = (v2.EWMA_LAMBDA * Q_warm
                              + (1.0 - v2.EWMA_LAMBDA) * np.outer(_z_row, _z_row))
                Q_ewma = Q_warm

        # EWMA correlation update
        r_prev = returns_arr[t - 1]
        x_prev = r_prev - mu_arr
        z_prev = x_prev / np.where(sigma_current > 1e-8, sigma_current, 1e-8)
        Q_ewma, R_dynamic = v2._ewma_corr_update(Q_ewma, z_prev, v2.EWMA_LAMBDA)

        # GARCH one-step-ahead forecast
        sigma_sq_next  = (omega_arr
                          + alpha_arr * x_prev ** 2
                          + beta_arr  * sigma_current ** 2)
        sigma_forecast = np.sqrt(np.maximum(sigma_sq_next, 1e-12))
        sigma_current  = sigma_forecast.copy()

        # Instrument state from t-1 (no look-ahead)
        linear_prices_now = prices_linear.iloc[t - 1].values
        S_now     = float(spy_prices.iloc[t - 1])
        sigma_now = float(vix_series.iloc[t - 1]) / 100.0
        rate_now  = float(dgs10_series.iloc[t - 1]) / 100.0
        K         = float(straddle_state.iloc[t - 1]["strike_spy"])
        T_now     = float(straddle_state.iloc[t - 1]["tenor_years"])

        pnl_sim    = v2.scenarios_to_pnl(
            nu_marginals, sigma_forecast, mu_arr,
            R_dynamic, nu_copula_cached, rng,
            linear_prices_now, linear_shares,
            S_now, sigma_now, rate_now, K, T_now,
        )
        var_arr[t] = v2.extract_var(pnl_sim)

    # ── Backtest ──────────────────────────────────────────────────────────────
    dates_bt   = dates[window:]
    var_series = pd.Series(var_arr[window:], index=dates_bt)
    pnl_bt     = pnl_series.iloc[window:].reindex(dates_bt)

    bt = run_backtest(
        pnl=pnl_bt,
        var=var_series,
        confidence=v2.ALPHA,
        method_name=f"R{refit_every}",
    )

    return {
        "refit_every" : refit_every,
        "T_bt"        : bt.T,
        "n_refits"    : n_refits,
        "N"           : bt.N,
        "exc_rate"    : bt.exception_rate,
        "lr_uc"       : bt.lr_uc,
        "pvalue_uc"   : bt.pvalue_uc,
        "reject_uc"   : bt.reject_uc,
        "lr_ind"      : bt.lr_ind,
        "pvalue_ind"  : bt.pvalue_ind,
        "reject_ind"  : bt.reject_ind,
        "lr_cc"       : bt.lr_cc,
        "pvalue_cc"   : bt.pvalue_cc,
        "reject_cc"   : bt.reject_cc,
        "mean_var"    : float(np.nanmean(var_series)),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("  REFIT SWEEP  —  GARCH(1,1)-t Copula MC VaR  (v2_audit core)")
    print(f"  Fixed window : {FIXED_WINDOW} trading days")
    print(f"  Refit every  : {REFITS_TO_TEST} trading days")
    print(f"  Running {len(REFITS_TO_TEST)} workers in parallel …")
    print("=" * 70)

    results = {}

    with ProcessPoolExecutor(max_workers=len(REFITS_TO_TEST)) as executor:
        futures = {executor.submit(_run_one_refit, r): r for r in REFITS_TO_TEST}
        for future in as_completed(futures):
            r = futures[future]
            try:
                res = future.result()
                results[r] = res
                print(f"  Refit={r:3d}d  done  "
                      f"({res['n_refits']} refits, {res['N']} exceptions, "
                      f"exc rate={res['exc_rate']:.2%})")
            except Exception as exc:
                print(f"  Refit={r:3d}d  ERROR: {exc}")

    # ── Comparison table ──────────────────────────────────────────────────────
    alpha = 0.99
    print("\n")
    print("=" * 100)
    print(f"  {'Refit':>8}  {'#Refits':>7}  {'T_bt':>6}  {'Exp':>5}  {'Obs':>5}  {'Exc%':>6}  "
          f"{'Kupiec LR':>10}  {'Kup p':>7}  {'Kup?':>5}  "
          f"{'Ind p':>7}  {'CC p':>7}  {'Mean VaR':>10}")
    print(f"  {'-'*93}")
    for r in sorted(results):
        res      = results[r]
        expected = res["T_bt"] * (1 - alpha)
        marker   = " <-- baseline" if r == 50 else ""
        print(f"  {r:>5}d        "
              f"{res['n_refits']:>7}  "
              f"{res['T_bt']:>6}  "
              f"{expected:>5.1f}  "
              f"{res['N']:>5}  "
              f"{res['exc_rate']:>5.2%}  "
              f"{res['lr_uc']:>10.4f}  "
              f"{res['pvalue_uc']:>7.4f}  "
              f"{'YES' if res['reject_uc'] else 'no ':>5}  "
              f"{res['pvalue_ind']:>7.4f}  "
              f"{res['pvalue_cc']:>7.4f}  "
              f"${res['mean_var']:>9,.0f}"
              f"{marker}")
    print("=" * 100)
    print("\n  Kupiec H0: exception rate = 1%  (reject = VaR too tight or too loose)")
    print("  Christoffersen Ind H0: exceptions are i.i.d.  (reject = clustering)")
    print(f"\n  All runs: WINDOW={FIXED_WINDOW}d fixed.")


if __name__ == "__main__":
    main()
