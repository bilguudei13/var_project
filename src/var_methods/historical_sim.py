"""
historical_sim.py
=================
1-day 99% Historical Simulation VaR using historical risk-factor shocks and
full repricing of the current portfolio snapshot.

Methodological upgrade
----------------------
This implementation estimates Historical Simulation VaR by:

1. taking the current end-of-day portfolio snapshot,
2. applying the previous 500 daily risk-factor shocks,
3. repricing all positions under each scenario with explicit pricing
   functions, and
4. extracting the empirical 99% loss order statistic and empirical ES.

The backtest uses the actual next-day market levels for the realised P&L,
rather than routing the realised day through the generic historical-scenario
engine. This keeps the scenario generation and the realised one-day outcome
conceptually separate while preserving the same pricing conventions.
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
    RAW_DIR as CONFIG_RAW_DIR,
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
RAW_DIR = REPO_ROOT / CONFIG_RAW_DIR
OUTPUT_FIGS = REPO_ROOT / "outputs" / "figures"
OUTPUT_TABLES = REPO_ROOT / "outputs" / "tables"

CRISIS_PERIODS = [
    ("2008-09-15", "2009-06-30", "GFC"),
    ("2011-07-01", "2012-01-31", "Euro Debt"),
    ("2020-02-20", "2020-05-31", "COVID"),
    ("2022-01-01", "2022-12-31", "Inflation Shock"),
]


# =============================================================================
# DATA PREPARATION
# =============================================================================

def aligned_weights(price_columns: list[str]) -> pd.Series:
    """
    Build the portfolio weights vector aligned to the linear asset columns.
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


def build_linear_shares(prices: pd.DataFrame, weights: pd.Series, portfolio_value: float) -> np.ndarray:
    """
    Freeze linear positions at inception as share quantities.

    This removes the hidden daily-rebalancing assumption from the previous
    implementation and lets the linear book accumulate gains and losses through
    time like a static trading-book position set.
    """
    initial_prices = prices.iloc[0].reindex(weights.index).to_numpy(dtype=float)
    return portfolio_value * weights.to_numpy(dtype=float) / initial_prices


def load_market_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
# PRICING HELPERS
# =============================================================================

def price_straddle_position(
    spot,
    strike: float,
    tenor,
    rate,
    sigma,
):
    """
    Price a straddle position with explicit expiry handling.

    If tenor reaches zero, the position is priced at intrinsic value
    ``|S - K|`` rather than forcing a one-day Black-Scholes tenor floor.
    """
    spot_arr, tenor_arr, rate_arr, sigma_arr = np.broadcast_arrays(
        np.asarray(spot, dtype=float),
        np.asarray(tenor, dtype=float),
        np.asarray(rate, dtype=float),
        np.asarray(sigma, dtype=float),
    )
    strike_arr = np.broadcast_to(np.asarray(strike, dtype=float), spot_arr.shape)

    values = np.empty_like(spot_arr, dtype=float)
    active = tenor_arr > 0.0

    if np.any(active):
        priced, _ = price_straddle(
            spot_arr[active],
            strike_arr[active],
            tenor_arr[active],
            rate_arr[active],
            np.clip(sigma_arr[active], 1e-6, None),
        )
        values[active] = np.asarray(priced, dtype=float)

    if np.any(~active):
        values[~active] = np.abs(spot_arr[~active] - strike_arr[~active])

    return float(values) if values.ndim == 0 else values


def add_crisis_annotations(ax: plt.Axes, index: pd.Index) -> None:
    """
    Shade major crisis periods that overlap the plot range.
    """
    if len(index) == 0:
        return

    start = pd.Timestamp(index.min())
    end = pd.Timestamp(index.max())
    upper = ax.get_ylim()[1]

    for period_start, period_end, label in CRISIS_PERIODS:
        period_start_ts = pd.Timestamp(period_start)
        period_end_ts = pd.Timestamp(period_end)
        if period_end_ts < start or period_start_ts > end:
            continue

        ax.axvspan(period_start_ts, period_end_ts, color="#7f8c8d", alpha=0.08, zorder=0)
        midpoint = period_start_ts + (period_end_ts - period_start_ts) / 2
        ax.text(
            midpoint,
            upper * 0.96,
            label,
            fontsize=8,
            color="#555555",
            ha="center",
            va="top",
        )


# =============================================================================
# PORTFOLIO REPRICING
# =============================================================================

def current_portfolio_value(
    snapshot: pd.Series,
    state: pd.Series,
    linear_shares: np.ndarray,
    price_columns: list[str],
) -> tuple[float, dict[str, float]]:
    """
    Value the current portfolio snapshot before applying any scenario shocks.
    """
    current_prices = snapshot[price_columns].to_numpy(dtype=float)
    linear_value = float(np.dot(linear_shares, current_prices))

    current_rate = max(float(snapshot["DGS10"]) / 100.0, 0.0)
    irs_value, _ = price_irs(
        IRS_NOTIONAL,
        IRS_FIXED_RATE,
        current_rate,
    )

    straddle_price = price_straddle_position(
        float(snapshot["SPY"]),
        float(state["strike_spy"]),
        float(state["tenor_years"]),
        current_rate,
        max(float(snapshot["VIX"]) / 100.0, 1e-6),
    )
    straddle_value = float(straddle_price) * STRADDLE_SHARES

    components = {
        "linear_value": linear_value,
        "irs_value": float(irs_value),
        "straddle_value": straddle_value,
    }

    total_value = components["linear_value"] + components["irs_value"] + components["straddle_value"]
    return total_value, components


def scenario_loss_distribution(
    snapshot: pd.Series,
    state: pd.Series,
    shock_window: pd.DataFrame,
    linear_shares: np.ndarray,
    price_columns: list[str],
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
    current_total_value, _ = current_portfolio_value(snapshot, state, linear_shares, price_columns)

    current_prices = snapshot[price_columns].to_numpy(dtype=float)
    shocked_prices = np.exp(
        shock_window[[f"{column}_ret" for column in price_columns]].to_numpy(dtype=float)
    ) * current_prices
    scenario_linear_values = shocked_prices @ linear_shares

    shocked_rates = np.clip(
        float(snapshot["DGS10"]) + shock_window["DGS10_change"].to_numpy(dtype=float),
        0.0,
        None,
    ) / 100.0
    scenario_irs_values, _ = price_irs(IRS_NOTIONAL, IRS_FIXED_RATE, shocked_rates)

    horizon_tenor = max(float(state["tenor_years"]) - 1.0 / TRADING_DAYS, 0.0)
    scenario_spy = shocked_prices[:, price_columns.index("SPY")]
    scenario_vol = np.clip(
        float(snapshot["VIX"]) * np.exp(shock_window["VIX_ret"].to_numpy(dtype=float)) / 100.0,
        1e-6,
        None,
    )
    scenario_straddle_values = (
        np.asarray(
            price_straddle_position(
                scenario_spy,
                float(state["strike_spy"]),
                horizon_tenor,
                shocked_rates,
                scenario_vol,
            ),
            dtype=float,
        )
        * STRADDLE_SHARES
    )

    scenario_total_values = (
        scenario_linear_values
        + np.asarray(scenario_irs_values, dtype=float)
        + scenario_straddle_values
    )
    pnl = scenario_total_values - current_total_value
    losses = current_total_value - scenario_total_values

    return losses, pnl, current_total_value


def realised_next_day_pnl(
    snapshot: pd.Series,
    next_snapshot: pd.Series,
    state: pd.Series,
    linear_shares: np.ndarray,
    price_columns: list[str],
) -> tuple[float, float]:
    """
    Reprice the current portfolio against actual next-day market levels.

    This keeps the backtest tied to realised market data rather than feeding the
    realised day back through the generic scenario engine.
    """
    current_total_value, _ = current_portfolio_value(snapshot, state, linear_shares, price_columns)

    next_prices = next_snapshot[price_columns].to_numpy(dtype=float)
    next_linear_value = float(np.dot(linear_shares, next_prices))

    next_rate = max(float(next_snapshot["DGS10"]) / 100.0, 0.0)
    next_irs_value, _ = price_irs(IRS_NOTIONAL, IRS_FIXED_RATE, next_rate)

    horizon_tenor = max(float(state["tenor_years"]) - 1.0 / TRADING_DAYS, 0.0)
    next_straddle_value = float(
        price_straddle_position(
            float(next_snapshot["SPY"]),
            float(state["strike_spy"]),
            horizon_tenor,
            next_rate,
            max(float(next_snapshot["VIX"]) / 100.0, 1e-6),
        )
    ) * STRADDLE_SHARES

    next_total_value = next_linear_value + float(next_irs_value) + next_straddle_value
    pnl = next_total_value - current_total_value
    loss = -pnl
    return float(loss), float(pnl)


# =============================================================================
# HISTORICAL SIMULATION
# =============================================================================

def empirical_var_es(losses: np.ndarray, alpha: float) -> tuple[float, float, int]:
    """
    Empirical Historical Simulation VaR and ES based on exact order statistics.
    """
    ordered = np.sort(np.asarray(losses, dtype=float))
    tail_count = max(1, int(np.ceil(len(ordered) * (1.0 - alpha))))
    tail = ordered[-tail_count:]
    var_value = float(tail[0])
    es_value = float(tail.mean())
    return var_value, es_value, tail_count


def compute_historical_sim_var(
    market_levels: pd.DataFrame,
    shock_frame: pd.DataFrame,
    straddle_state: pd.DataFrame,
    price_columns: list[str],
    linear_shares: np.ndarray,
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

    records: list[dict[str, float | int | str]] = []
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
        next_snapshot = market_levels.loc[forecast_date]
        state = straddle_state.loc[snapshot_date]
        historical_shocks = shock_frame.iloc[t - window:t]

        scenario_losses, _, current_total_value = scenario_loss_distribution(
            snapshot=snapshot,
            state=state,
            shock_window=historical_shocks,
            linear_shares=linear_shares,
            price_columns=price_columns,
        )

        var_t, es_t, tail_count = empirical_var_es(scenario_losses, alpha)
        realised_loss_t, realised_pnl_t = realised_next_day_pnl(
            snapshot=snapshot,
            next_snapshot=next_snapshot,
            state=state,
            linear_shares=linear_shares,
            price_columns=price_columns,
        )
        realised_pnl[forecast_date] = realised_pnl_t

        records.append(
            {
                "snapshot_date": snapshot_date.strftime("%Y-%m-%d"),
                "current_portfolio_value": current_total_value,
                "VaR_HistSim": max(var_t, 0.0),
                "ES_HistSim": max(es_t, 0.0),
                "tail_scenarios": tail_count,
                "quantile_method": "empirical_order_statistic",
                "historical_window_mean_loss": float(scenario_losses.mean()),
                "historical_window_vol_loss": float(scenario_losses.std(ddof=1)),
                "realised_loss": realised_loss_t,
                "realised_pnl": realised_pnl_t,
                "exception": int(realised_pnl_t < -var_t),
            }
        )

    results = pd.DataFrame(records, index=shock_frame.index[window:])
    realised_pnl_series = pd.Series(realised_pnl, name="pnl_realised_market_repricing").sort_index()

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

    realised_loss_signed = -results["realised_pnl"]
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
    ax.plot(
        results.index,
        realised_loss_signed,
        color="#2e86ab",
        linewidth=0.8,
        alpha=0.9,
        label="Realised loss (+ = loss, - = gain)",
    )
    ax.scatter(exceptions, realised_loss_signed.loc[exceptions], color="#c0392b", s=18, zorder=5, label="Exceptions")

    ax.set_title("Historical Simulation VaR Evolution | Full Repricing", loc="left", fontweight="bold")
    ax.set_ylabel("USD loss (+) / gain (-)")
    ax.set_xlabel("Date")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    add_crisis_annotations(ax, results.index)
    ax.legend(loc="upper left", ncol=4, fontsize=8)
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
            "tail_scenarios": results.loc[common, "tail_scenarios"],
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
                "backtest_basis": "actual next-day market levels repriced from current snapshot",
                "linear_book_convention": "static inception shares (no daily rebalancing)",
                "straddle_rate_proxy": "DGS10",
                "swap_rate_proxy": "DGS10",
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
    price_columns = [column for column in market_levels.columns if column in {"SPY", "IEF", "GLD", "EURUSD"}]
    weights = aligned_weights(price_columns)
    linear_shares = build_linear_shares(market_levels[price_columns], weights, V0)

    results, realised_pnl = compute_historical_sim_var(
        market_levels=market_levels,
        shock_frame=shock_frame,
        straddle_state=straddle_state,
        price_columns=price_columns,
        linear_shares=linear_shares,
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
