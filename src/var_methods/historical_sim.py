"""
historical_sim.py
=================
1-day 99% Historical Simulation VaR using historical risk-factor shocks and
full repricing of the current portfolio snapshot.

Methodological upgrade
----------------------
This implementation does not estimate VaR from the realised portfolio P&L
series alone. Instead, for each forecast date it:

1. takes the current end-of-day portfolio snapshot,
2. applies the previous 500 daily risk-factor shocks,
3. reprices all positions under each scenario using the portfolio's pricing
   functions, and
4. extracts the empirical 99% loss quantile as Historical Simulation VaR.

This is the stronger Historical Simulation setup for a multi-asset portfolio
with non-linear instruments because it preserves the current composition of the
book and makes the pricing-function approach explicit.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DATA_DIR = REPO_ROOT / "src" / "data"

for path in (REPO_ROOT, SRC_DATA_DIR):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from backtesting.backtest import run_backtest
from backtesting.plot_backtest import plot_all
from config import (
    IRS_FIXED_RATE,
    IRS_NOTIONAL,
    PROCESSED_DIR as CONFIG_PROCESSED_DIR,
    RAW_DIR,
    RF_RATE,
    STRADDLE_DAYS,
    STRADDLE_SHARES,
    V0,
    WEIGHTS_DICT,
)
from portfolio_pricing import price_irs, price_straddle


# =============================================================================
# SETTINGS
# =============================================================================

WINDOW = 500
ALPHA = 0.99
TRADING_DAYS = 252

PROCESSED_DIR = REPO_ROOT / CONFIG_PROCESSED_DIR
RAW_DIR = REPO_ROOT / RAW_DIR
OUTPUT_FIGS = REPO_ROOT / "outputs" / "figures"
OUTPUT_TABLES = REPO_ROOT / "outputs" / "tables"


# =============================================================================
# DATA PREPARATION
# =============================================================================

def aligned_weights(price_columns: list[str]) -> pd.Series:
    """
    Build the portfolio weights vector aligned to the price columns.
    """
    weights = {
        ("EURUSD" if key == "EURUSD=X" else key): value
        for key, value in WEIGHTS_DICT.items()
    }
    weight_series = pd.Series(weights).reindex(price_columns)

    if weight_series.isna().any():
        missing = list(weight_series[weight_series.isna()].index)
        raise ValueError(f"Missing portfolio weights for columns: {missing}")

    return weight_series.astype(float)


def load_market_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Load aligned market levels and derive the historical shock matrix.

    Returns
    -------
    market_levels : pd.DataFrame
        Daily levels for linear assets, VIX, and DGS10 on a common date index.
    shock_frame : pd.DataFrame
        Historical factor shocks used as Historical Simulation scenarios.
        Price and VIX shocks are log returns; DGS10 uses absolute daily change.
    straddle_state : pd.DataFrame
        Rolling strike and tenor for the current ATM straddle position.
    """
    prices = pd.read_csv(RAW_DIR / "prices.csv", index_col=0, parse_dates=True).sort_index()
    vix = pd.read_csv(RAW_DIR / "vix.csv", index_col=0, parse_dates=True).sort_index().iloc[:, 0]
    dgs10 = pd.read_csv(RAW_DIR / "dgs10.csv", index_col=0, parse_dates=True).sort_index().iloc[:, 0]

    vix.name = "VIX"
    dgs10.name = "DGS10"

    common = prices.index.intersection(vix.index).intersection(dgs10.index)
    prices = prices.loc[common].copy()
    vix = vix.loc[common].astype(float).copy()
    dgs10 = dgs10.loc[common].astype(float).copy()

    market_levels = prices.copy()
    market_levels["VIX"] = vix
    market_levels["DGS10"] = dgs10

    shock_frame = pd.DataFrame(index=market_levels.index[1:])
    for column in prices.columns:
        shock_frame[f"{column}_ret"] = np.log(prices[column] / prices[column].shift(1)).loc[shock_frame.index]
    shock_frame["VIX_ret"] = np.log(vix / vix.shift(1)).loc[shock_frame.index]
    shock_frame["DGS10_change"] = dgs10.diff().loc[shock_frame.index]

    straddle_state = build_straddle_state(prices["SPY"])

    print(f"Loaded aligned market data: {len(market_levels)} dates")
    print(f"Linear risk factors       : {list(prices.columns)}")
    print("Additional risk factors   : ['VIX', 'DGS10']")
    print(f"Historical shock sample   : {len(shock_frame)} observations")

    return market_levels, shock_frame, straddle_state


def build_straddle_state(spot_series: pd.Series) -> pd.DataFrame:
    """
    Reconstruct the daily state of the rolling ATM straddle.

    The position is modelled as a 30-trading-day ATM straddle that is rolled
    into a fresh ATM contract whenever the holding period reaches
    ``STRADDLE_DAYS``. Each row describes the position held at that date's
    close and used as the current snapshot for the next-day VaR forecast.
    """
    strikes: list[float] = []
    tenors: list[float] = []
    days_held_list: list[int] = []
    rolled_list: list[bool] = []

    current_strike = float(spot_series.iloc[0])
    days_held = 0

    for index, spot in enumerate(spot_series):
        rolled = False
        if index > 0 and days_held >= STRADDLE_DAYS:
            current_strike = float(spot)
            days_held = 0
            rolled = True

        tenor = max((STRADDLE_DAYS - days_held) / TRADING_DAYS, 1.0 / TRADING_DAYS)

        strikes.append(current_strike)
        tenors.append(tenor)
        days_held_list.append(days_held)
        rolled_list.append(rolled)

        days_held += 1

    return pd.DataFrame(
        {
            "strike_spy": strikes,
            "tenor_years": tenors,
            "days_held": days_held_list,
            "rolled_today": rolled_list,
        },
        index=spot_series.index,
    )


# =============================================================================
# PORTFOLIO REPRICING
# =============================================================================

def current_portfolio_value(
    snapshot: pd.Series,
    state: pd.Series,
    weights: pd.Series,
) -> tuple[float, dict[str, float], np.ndarray]:
    """
    Value the current portfolio snapshot before applying any scenario shocks.
    """
    current_prices = snapshot[weights.index].to_numpy(dtype=float)
    weight_array = weights.to_numpy(dtype=float)
    linear_shares = V0 * weight_array / current_prices

    linear_value = float(np.dot(linear_shares, current_prices))

    irs_value, _ = price_irs(
        IRS_NOTIONAL,
        IRS_FIXED_RATE,
        float(snapshot["DGS10"]) / 100.0,
    )

    straddle_price, _ = price_straddle(
        float(snapshot["SPY"]),
        float(state["strike_spy"]),
        float(state["tenor_years"]),
        RF_RATE,
        max(float(snapshot["VIX"]) / 100.0, 1e-6),
    )
    straddle_value = float(straddle_price) * STRADDLE_SHARES

    components = {
        "linear_value": linear_value,
        "irs_value": float(irs_value),
        "straddle_value": straddle_value,
    }

    total_value = components["linear_value"] + components["irs_value"] + components["straddle_value"]
    return total_value, components, linear_shares


def scenario_loss_distribution(
    snapshot: pd.Series,
    state: pd.Series,
    shock_window: pd.DataFrame,
    weights: pd.Series,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Reprice the current snapshot under all historical shocks in the window.

    Returns
    -------
    losses : np.ndarray
        Scenario losses (positive values = losses).
    pnl : np.ndarray
        Scenario P&L (positive values = gains).
    current_total_value : float
        Current mark-to-market value of the portfolio snapshot.
    """
    current_total_value, components, linear_shares = current_portfolio_value(snapshot, state, weights)

    current_prices = snapshot[weights.index].to_numpy(dtype=float)
    shocked_prices = np.exp(
        shock_window[[f"{column}_ret" for column in weights.index]].to_numpy(dtype=float)
    ) * current_prices
    scenario_linear_values = shocked_prices @ linear_shares

    shocked_rates = np.clip(
        float(snapshot["DGS10"]) + shock_window["DGS10_change"].to_numpy(dtype=float),
        0.0,
        None,
    ) / 100.0
    scenario_irs_values, _ = price_irs(IRS_NOTIONAL, IRS_FIXED_RATE, shocked_rates)

    horizon_tenor = max(float(state["tenor_years"]) - 1.0 / TRADING_DAYS, 1.0 / TRADING_DAYS)
    scenario_spy = shocked_prices[:, list(weights.index).index("SPY")]
    scenario_vol = np.clip(
        float(snapshot["VIX"]) * np.exp(shock_window["VIX_ret"].to_numpy(dtype=float)) / 100.0,
        1e-6,
        None,
    )
    scenario_straddle_prices, _ = price_straddle(
        scenario_spy,
        float(state["strike_spy"]),
        horizon_tenor,
        RF_RATE,
        scenario_vol,
    )
    scenario_straddle_values = np.asarray(scenario_straddle_prices, dtype=float) * STRADDLE_SHARES

    scenario_total_values = (
        scenario_linear_values
        + np.asarray(scenario_irs_values, dtype=float)
        + scenario_straddle_values
    )
    pnl = scenario_total_values - current_total_value
    losses = current_total_value - scenario_total_values

    return losses, pnl, current_total_value


# =============================================================================
# HISTORICAL SIMULATION
# =============================================================================

def compute_historical_sim_var(
    market_levels: pd.DataFrame,
    shock_frame: pd.DataFrame,
    straddle_state: pd.DataFrame,
    weights: pd.Series,
    window: int,
    alpha: float,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Compute Historical Simulation VaR by shocking the current snapshot and
    fully repricing the portfolio under the historical scenario set.
    """
    if len(shock_frame) <= window:
        raise ValueError(
            f"Not enough shock observations for Historical Simulation: "
            f"{len(shock_frame)} available, {window + 1} required."
        )

    records: list[dict[str, float | str]] = []
    realised_pnl: dict[pd.Timestamp, float] = {}

    print("\n" + "=" * 70)
    print("Historical Simulation VaR | Full Repricing")
    print("=" * 70)
    print(f"Window          : {window} trading days")
    print(f"Confidence       : {alpha:.0%}")
    print(f"Forecast points  : {len(shock_frame) - window}")

    common_dates = market_levels.index

    for t in range(window, len(shock_frame)):
        snapshot_date = common_dates[t]
        forecast_date = shock_frame.index[t]

        snapshot = market_levels.loc[snapshot_date]
        state = straddle_state.loc[snapshot_date]
        historical_shocks = shock_frame.iloc[t - window:t]

        scenario_losses, _, current_total_value = scenario_loss_distribution(
            snapshot=snapshot,
            state=state,
            shock_window=historical_shocks,
            weights=weights,
        )

        var_t = float(np.quantile(scenario_losses, alpha))
        tail_losses = scenario_losses[scenario_losses >= var_t]
        es_t = float(tail_losses.mean()) if len(tail_losses) else var_t

        realised_loss, realised_pnl_array, _ = scenario_loss_distribution(
            snapshot=snapshot,
            state=state,
            shock_window=shock_frame.iloc[[t]],
            weights=weights,
        )
        realised_pnl_t = float(realised_pnl_array[0])
        realised_pnl[forecast_date] = realised_pnl_t

        records.append(
            {
                "snapshot_date": snapshot_date.strftime("%Y-%m-%d"),
                "current_portfolio_value": current_total_value,
                "VaR_HistSim": max(var_t, 0.0),
                "ES_HistSim": max(es_t, 0.0),
                "historical_window_mean_loss": float(scenario_losses.mean()),
                "historical_window_vol_loss": float(scenario_losses.std(ddof=1)),
                "realised_loss": float(realised_loss[0]),
                "realised_pnl": realised_pnl_t,
                "exception": int(realised_pnl_t < -var_t),
            }
        )

    results = pd.DataFrame(records, index=shock_frame.index[window:])
    realised_pnl_series = pd.Series(realised_pnl, name="pnl_realised_repricing").sort_index()

    print(f"Mean VaR        : ${results['VaR_HistSim'].mean():,.2f}")
    print(f"Mean ES         : ${results['ES_HistSim'].mean():,.2f}")
    print(f"Min VaR         : ${results['VaR_HistSim'].min():,.2f}")
    print(f"Max VaR         : ${results['VaR_HistSim'].max():,.2f}")

    return results, realised_pnl_series


# =============================================================================
# OUTPUTS
# =============================================================================

def plot_var_evolution(results: pd.DataFrame) -> None:
    """
    Save a dedicated Historical Simulation VaR evolution plot.
    """
    OUTPUT_FIGS.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13, 5))

    actual_loss = -results["realised_pnl"]
    exceptions = results.index[results["exception"] == 1]

    ax.fill_between(
        results.index,
        results["VaR_HistSim"],
        results["ES_HistSim"],
        color="#f6d6bd",
        alpha=0.8,
        label="ES minus VaR band",
    )
    ax.plot(results.index, results["VaR_HistSim"], color="#d35400", linewidth=1.4, label="Historical Simulation VaR")
    ax.plot(results.index, results["ES_HistSim"], color="#8e5a2b", linewidth=1.1, linestyle="--", label="Historical Simulation ES")
    ax.plot(results.index, actual_loss, color="#2e86ab", linewidth=0.8, alpha=0.85, label="Realised loss")
    ax.scatter(exceptions, actual_loss.loc[exceptions], color="#c0392b", s=18, zorder=5, label="Exceptions")

    ax.set_title("Historical Simulation VaR Evolution | Full Repricing", loc="left", fontweight="bold")
    ax.set_ylabel("USD")
    ax.set_xlabel("Date")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="upper left", ncol=4, fontsize=8)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    fig.tight_layout()

    path = OUTPUT_FIGS / "12_var_evolution_HistSim.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved VaR plot       -> {path.relative_to(REPO_ROOT)}")


def save_results(
    results: pd.DataFrame,
    realised_pnl: pd.Series,
    backtest_result,
) -> None:
    """
    Save Historical Simulation outputs to the shared project folders.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)

    var_path = PROCESSED_DIR / "var_historical_sim.csv"
    results.to_csv(var_path)
    print(f"Saved VaR series      -> {var_path.relative_to(REPO_ROOT)}")

    common = realised_pnl.index.intersection(results.index)
    detail = pd.DataFrame(
        {
            "pnl": realised_pnl.loc[common],
            "actual_loss": -realised_pnl.loc[common],
            "VaR_HistSim": results.loc[common, "VaR_HistSim"],
            "ES_HistSim": results.loc[common, "ES_HistSim"],
            "snapshot_date": results.loc[common, "snapshot_date"],
            "current_portfolio_value": results.loc[common, "current_portfolio_value"],
            "exception": (realised_pnl.loc[common] < -results.loc[common, "VaR_HistSim"]).astype(int),
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
                "implementation": "historical shocks plus full repricing",
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


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """
    Run the Historical Simulation workflow using historical shocks and
    full repricing of the current portfolio snapshot.
    """
    market_levels, shock_frame, straddle_state = load_market_data()
    weights = aligned_weights([column for column in market_levels.columns if column in {"SPY", "IEF", "GLD", "EURUSD"}])

    results, realised_pnl = compute_historical_sim_var(
        market_levels=market_levels,
        shock_frame=shock_frame,
        straddle_state=straddle_state,
        weights=weights,
        window=WINDOW,
        alpha=ALPHA,
    )

    backtest_result = run_backtest(
        pnl=realised_pnl,
        var=results["VaR_HistSim"],
        confidence=ALPHA,
        method_name="HistSim",
    )

    print("\n" + str(backtest_result))

    plot_var_evolution(results)
    plot_all(backtest_result, pnl=realised_pnl, var=results["VaR_HistSim"], save=True)
    save_results(results, realised_pnl, backtest_result)

    print("\n" + "=" * 70)
    print("Historical Simulation workflow complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
