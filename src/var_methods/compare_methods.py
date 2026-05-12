"""
compare_methods.py
==================
Produces Figures 12 and 13 from backtesting/plot_backtest.py.

  Fig 12 — Exception rate comparison  (all methods, horizontal bar chart)
  Fig 13 — LR statistics dot plot     (UC / IND / CC per method)

Run this AFTER all individual method scripts have completed and written
their VaR CSVs to data/processed/.

Required files (data/processed/):
  var_delta_normal.csv   column: VaR_DN_full
  var_hist_sim.csv       column: VaR_HS          (Cornelius)
  var_monte_carlo.csv    column: VaR_MC           (Kira)
  var_garch.csv          column: VaR_GARCH        (Ben)
  var_evt.csv            column: VaR_EVT          (Jonas)
  var_garch_evt.csv      column: VaR_GARCH_EVT    (Jonas)
  total_portfolio_pnl.csv column: pnl_total

Usage
-----
  python compare_methods.py
"""

import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[2]
sys.path.insert(0, str(_ROOT))

import pandas as pd
from backtesting.backtest import run_backtest
from backtesting.plot_backtest import (
    plot_exception_rate_comparison,
    plot_lr_statistics,
)

PROCESSED_DIR = _ROOT / "data" / "processed"
ALPHA = 0.99

# ---------------------------------------------------------------------------
# Method registry
# Each entry: (display name, csv filename, VaR column name)
# Comment out any method not yet available.
# ---------------------------------------------------------------------------

METHODS = [
    ("Delta-Normal",  "var_delta_normal.csv",  "VaR_DN_full"),
    ("Hist Sim",      "var_historical_sim.csv",       "VaR_HistSim"),
    ("Monte Carlo",   "var_mc_garch_copula_v2.csv",    "VaR_MC_GARCH_COPULA_V2"),
    ("GARCH",         "var_copula.csv",          "VaR_COPULA"),  # Ben's GARCH with Gaussian copula
    ("EVT",           "var_evt.csv",            "VaR_EVT"),
    ("GARCH+EVT",     "var_garch_evt.csv",      "VaR_GARCH_EVT"),
]


def load_results() -> list:
    """
    Load pnl_total and each available VaR series, run_backtest for each,
    return list of BacktestResult objects.
    """
    pnl = pd.read_csv(
        PROCESSED_DIR / "total_portfolio_pnl.csv",
        index_col=0, parse_dates=True,
    )["pnl_total"]

    results = []
    skipped = []

    for name, filename, col in METHODS:
        path = PROCESSED_DIR / filename
        if not path.exists():
            skipped.append(name)
            continue
        try:
            var = pd.read_csv(path, index_col=0, parse_dates=True)[col]
        except KeyError:
            print(f"  WARNING: {filename} has no column '{col}' — skipping {name}")
            skipped.append(name)
            continue

        bt = run_backtest(
            pnl=pnl,
            var=var,
            confidence=ALPHA,
            method_name=name,
        )
        results.append(bt)
        print(f"  {name:<18}  N={bt.N:>4}  rate={bt.exception_rate:.2%}"
              f"  Kupiec={'REJECT' if bt.reject_uc else 'pass  '}"
              f"  (p={bt.pvalue_uc:.4f})")

    if skipped:
        print(f"\n  Skipped (files not found): {', '.join(skipped)}")

    return results


def main() -> None:
    print("=" * 60)
    print("Method comparison  |  Figs 12 + 13")
    print("=" * 60)

    results = load_results()

    if len(results) < 2:
        print("\nNeed at least 2 methods to produce comparison figures.")
        return

    print(f"\nGenerating Fig 12 — exception rate comparison ({len(results)} methods)...")
    plot_exception_rate_comparison(results, save=True)

    print(f"Generating Fig 13 — LR statistics ({len(results)} methods)...")
    plot_lr_statistics(results, save=True)

    print("\nDone.")
    print(f"  outputs/figures/12_exception_rate_comparison.png")
    print(f"  outputs/figures/13_lr_statistics.png")


if __name__ == "__main__":
    main()