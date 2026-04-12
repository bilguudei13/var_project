"""
historical_sim.py
=================
1-day 99% Historical Simulation VaR for the Market Risk Modelling project.

This module is intentionally scoped to the Historical Simulation workstream.
It plugs into the group's shared pipeline and reuses the centralised
backtesting and plotting utilities under /backtesting.

Implementation choice
---------------------
Whenever available, Historical Simulation is estimated from the realised
portfolio P&L series in ``data/processed/total_portfolio_pnl.csv``. This keeps
the method aligned with the full project portfolio, including the linear book,
the IRS, and the SPY straddle. If that file is not available, the script falls
back to the linear portfolio approximation based on ``log_returns.csv``.

This keeps the Historical Simulation contribution robust inside the shared
repository while preserving consistency with the broader portfolio design.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from backtesting.backtest import run_backtest
from backtesting.plot_backtest import plot_all


# =============================================================================
# SETTINGS
# =============================================================================

WINDOW = 500
ALPHA = 0.99
V0 = 1_000_000

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
OUTPUT_FIGS = REPO_ROOT / "outputs" / "figures"
OUTPUT_TABLES = REPO_ROOT / "outputs" / "tables"


# =============================================================================
# DATA LOADING
# =============================================================================

def load_returns() -> pd.DataFrame:
    """
    Load daily log returns for the linear risk factors.

    The file is produced by ``src/data/compute_returns.py`` and serves as a
    fallback data source if the full portfolio P&L file is not available.
    """
    path = PROCESSED_DIR / "log_returns.csv"
    returns = pd.read_csv(path, index_col=0, parse_dates=True)
    print(f"Loaded risk-factor returns: {returns.shape[0]} x {returns.shape[1]}")
    print(f"Risk factors: {list(returns.columns)}")
    return returns


def load_total_portfolio_pnl() -> pd.Series | None:
    """
    Load the realised full portfolio P&L if the group pipeline has created it.

    Returns
    -------
    pd.Series or None
        Daily total P&L including the linear book, IRS, and straddle.
    """
    path = PROCESSED_DIR / "total_portfolio_pnl.csv"
    if not path.exists():
        return None

    total_pnl = pd.read_csv(path, index_col=0, parse_dates=True)
    if "pnl_total" not in total_pnl.columns:
        return None

    series = total_pnl["pnl_total"].dropna().rename("pnl_total")
    print(f"Loaded full portfolio P&L: {len(series)} observations")
    return series


def build_linear_portfolio_pnl(returns: pd.DataFrame, portfolio_value: float) -> pd.Series:
    """
    Build a fallback linear portfolio P&L series from risk-factor returns.

    This fallback mirrors the equal-weight convention used elsewhere in the
    repository. It is only used if the richer full-portfolio P&L file has not
    been generated yet.
    """
    weights = np.ones(len(returns.columns)) / len(returns.columns)
    pnl = pd.Series(
        portfolio_value * returns.dot(weights),
        index=returns.index,
        name="pnl_linear",
    )
    print(
        "Full portfolio P&L file missing. "
        f"Falling back to equal-weight linear P&L across {len(weights)} factors."
    )
    return pnl


def load_portfolio_pnl() -> tuple[pd.Series, str]:
    """
    Load the P&L series used for Historical Simulation.

    Preference order:
    1. Full portfolio realised P&L (includes non-linear instruments)
    2. Linear equal-weight P&L reconstructed from risk-factor returns
    """
    full_pnl = load_total_portfolio_pnl()
    if full_pnl is not None:
        return full_pnl, "full_portfolio_pnl"

    returns = load_returns()
    linear_pnl = build_linear_portfolio_pnl(returns, V0)
    return linear_pnl, "linear_fallback"


# =============================================================================
# HISTORICAL SIMULATION VAR
# =============================================================================

def compute_historical_sim_var(
    pnl: pd.Series,
    window: int,
    alpha: float,
) -> pd.DataFrame:
    """
    Estimate rolling 1-day Historical Simulation VaR from historical P&L.

    For each date t, the method takes the previous ``window`` daily portfolio
    P&L observations as the empirical scenario set and computes:

        VaR_t = - empirical_quantile_{1-alpha}(P&L_{t-window:t-1})

    Parameters
    ----------
    pnl : pd.Series
        Daily portfolio P&L where positive values are gains and negative values
        are losses.
    window : int
        Rolling estimation window in trading days.
    alpha : float
        Confidence level, e.g. 0.99 for 99% VaR.
    """
    pnl = pnl.dropna().sort_index()
    if len(pnl) <= window:
        raise ValueError(
            f"Not enough observations for Historical Simulation: "
            f"{len(pnl)} available, {window + 1} required."
        )

    records: list[dict[str, float]] = []
    dates: list[pd.Timestamp] = []

    print("\n" + "=" * 60)
    print("Historical Simulation VaR")
    print("=" * 60)
    print(f"Window        : {window} trading days")
    print(f"Confidence    : {alpha:.0%}")
    print(f"Observations  : {len(pnl)}")
    print(f"VaR estimates : {len(pnl) - window}")

    tail_probability = 1.0 - alpha

    for t in range(window, len(pnl)):
        history = pnl.iloc[t - window:t]
        var_t = float(-history.quantile(tail_probability))

        records.append(
            {
                "VaR_HistSim": max(var_t, 0.0),
                "window_mean_pnl": float(history.mean()),
                "window_vol_pnl": float(history.std(ddof=1)),
            }
        )
        dates.append(pnl.index[t])

    results = pd.DataFrame(records, index=dates)

    print(f"Mean VaR      : ${results['VaR_HistSim'].mean():,.2f}")
    print(f"Min VaR       : ${results['VaR_HistSim'].min():,.2f}")
    print(f"Max VaR       : ${results['VaR_HistSim'].max():,.2f}")

    return results


# =============================================================================
# BACKTESTING + OUTPUTS
# =============================================================================

def save_results(
    results: pd.DataFrame,
    pnl: pd.Series,
    data_source: str,
    backtest_result,
) -> None:
    """
    Save Historical Simulation outputs in the shared repository structure.
    """
    OUTPUT_FIGS.mkdir(parents=True, exist_ok=True)
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    var_path = PROCESSED_DIR / "var_historical_sim.csv"
    results.to_csv(var_path)
    print(f"Saved VaR series      -> {var_path.relative_to(REPO_ROOT)}")

    common = pnl.index.intersection(results.index)
    pnl_aligned = pnl.loc[common]
    var_aligned = results.loc[common, "VaR_HistSim"]
    exceptions = (pnl_aligned < -var_aligned).astype(int)

    detail = pd.DataFrame(
        {
            "pnl": pnl_aligned,
            "actual_loss": -pnl_aligned,
            "VaR_HistSim": var_aligned,
            "exception": exceptions,
        },
        index=common,
    )
    detail_path = OUTPUT_TABLES / "backtest_historical_sim.csv"
    detail.to_csv(detail_path)
    print(f"Saved backtest table  -> {detail_path.relative_to(REPO_ROOT)}")

    summary = pd.DataFrame(
        [
            {
                "method": backtest_result.method_name,
                "confidence": backtest_result.confidence,
                "window": WINDOW,
                "data_source": data_source,
                "observations": backtest_result.T,
                "expected_exceptions": backtest_result.expected_N,
                "observed_exceptions": backtest_result.N,
                "exception_rate": backtest_result.exception_rate,
                "lr_uc": backtest_result.lr_uc,
                "pvalue_uc": backtest_result.pvalue_uc,
                "reject_uc_5pct": backtest_result.reject_uc,
                "lr_ind": backtest_result.lr_ind,
                "pvalue_ind": backtest_result.pvalue_ind,
                "reject_ind_5pct": backtest_result.reject_ind,
                "lr_cc": backtest_result.lr_cc,
                "pvalue_cc": backtest_result.pvalue_cc,
                "reject_cc_5pct": backtest_result.reject_cc,
            }
        ]
    )
    summary_path = OUTPUT_TABLES / "backtest_summary_historical_sim.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary table   -> {summary_path.relative_to(REPO_ROOT)}")


def main() -> None:
    """
    Run the full Historical Simulation workflow for the group repository.
    """
    pnl, data_source = load_portfolio_pnl()
    results = compute_historical_sim_var(pnl, WINDOW, ALPHA)

    backtest_result = run_backtest(
        pnl=pnl,
        var=results["VaR_HistSim"],
        confidence=ALPHA,
        method_name="HistSim",
    )

    print("\n" + str(backtest_result))

    plot_all(
        backtest_result,
        pnl=pnl,
        var=results["VaR_HistSim"],
        save=True,
    )

    save_results(results, pnl, data_source, backtest_result)

    print("\n" + "=" * 60)
    print("Historical Simulation workflow complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
