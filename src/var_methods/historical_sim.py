"""
historical_sim.py
=================
Core Historical Simulation implementation for the project.

This file is the actual HistSim model engine. It estimates a 1-day 99%
Historical Simulation VaR for the current portfolio by using real historical
risk-factor moves as scenarios and by repricing the current portfolio under
each scenario. The second HistSim file, `historical_sim_analysis.py`, is a
diagnostic layer built on top of this output; this file is the baseline model
that produces the VaR, ES, realised P&L, backtest table, and HistSim figures.

High-level idea
---------------
Historical Simulation does not assume normally distributed returns. Instead,
it asks what would happen to today's portfolio if recent historical market
moves happened again tomorrow. In this project, the historical shocks are joint
daily moves in:

* SPY, IEF, GLD, and EURUSD prices,
* VIX as the volatility proxy for the SPY straddle, and
* DGS10 as the interest-rate proxy for the IRS and option discount rate.

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

Important conventions
---------------------
* The linear book is held as fixed inception share quantities, so it is not
  implicitly rebalanced back to equal weights every day.
* Price and VIX shocks are applied as log-return shocks.
* DGS10 shocks are applied as absolute yield-level changes.
* The SPY straddle is fully repriced under each scenario, including the
  one-day reduction in tenor and the roll convention when the option expires.
* At 99% confidence with a 500-day window, the empirical 1% tail contains
  ceil(500 * 1%) = 5 scenarios. VaR is the least severe loss inside that
  5-scenario tail; ES is the average loss across those 5 scenarios.
"""

from __future__ import annotations

import sys
from decimal import Decimal, ROUND_CEILING
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
from backtesting.plot_backtest import _add_crisis_annotations, plot_all
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
from portfolio_pricing import (
    build_straddle_state,
    price_irs,
    price_straddle_position,
)


# =============================================================================
# SETTINGS
# =============================================================================

WINDOW = 500
ALPHA = 0.99
TRADING_DAYS = 252
LINEAR_ASSETS = {"SPY", "IEF", "GLD", "EURUSD"}
COMPONENT_NAMES = ("linear", "irs", "straddle")

PROCESSED_DIR = REPO_ROOT / CONFIG_PROCESSED_DIR
RAW_DIR = REPO_ROOT / CONFIG_RAW_DIR
OUTPUT_FIGS = REPO_ROOT / "outputs" / "figures"
OUTPUT_TABLES = REPO_ROOT / "outputs" / "tables"


# =============================================================================
# DATA PREPARATION
# =============================================================================

def aligned_weights(price_columns: list[str]) -> pd.Series:
    """
    Build the linear-asset weight vector in the same order as the price data.

    The project configuration stores the FX exposure as `EURUSD=X`, while the
    processed price matrix uses `EURUSD`. This function normalises that naming
    difference, reindexes the weights to the actual price columns, and raises a
    clear error if a required weight is missing.

    The output is used only to create fixed inception share quantities. After
    that point the Historical Simulation workflow works with shares, not daily
    target weights.
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

    Each linear position is sized at the first available price:

        shares_i = portfolio_value * weight_i / initial_price_i

    This removes a hidden daily-rebalancing assumption. If the model used
    weights directly every day, it would behave as if the portfolio were reset
    to the original weights each day. Fixed share quantities are closer to a
    trading-book position set: the market value changes over time because
    prices move, not because the book is continuously rebalanced.
    """
    initial_prices = prices.iloc[0].reindex(weights.index).to_numpy(dtype=float)
    return portfolio_value * weights.to_numpy(dtype=float) / initial_prices


def load_market_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load aligned market levels and derive the historical shock matrix.

    The function creates two different objects on purpose:

    1. `market_levels` contains actual observed levels on each date. These
       levels are needed to value the current portfolio snapshot and to compute
       realised next-day P&L.
    2. `shock_frame` contains one-day historical shocks. These shocks are the
       scenarios used by Historical Simulation. For a forecast date, only the
       previous `WINDOW` rows of `shock_frame` are used.

    The mixed shock convention is intentional. Price-like variables and VIX
    are shocked multiplicatively via log returns. DGS10 is a yield level, so it
    is shocked additively with an absolute daily change.

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
    Value the current portfolio snapshot before any historical shock is applied.

    This is the baseline mark-to-market value against which every scenario is
    compared. Scenario P&L is:

        shocked_portfolio_value - current_portfolio_value

    Scenario loss is the negative of that P&L. Returning the component values
    as well makes the function useful for diagnostics and consistency checks.
    """
    components = current_component_values(snapshot, state, linear_shares, price_columns)
    total_value = components["linear_value"] + components["irs_value"] + components["straddle_value"]
    return total_value, components


def current_component_values(
    snapshot: pd.Series,
    state: pd.Series,
    linear_shares: np.ndarray,
    price_columns: list[str],
) -> dict[str, float]:
    """
    Value each portfolio component at the current market snapshot.

    Components:

    * Linear book: fixed shares multiplied by current prices.
    * IRS: current DGS10 level is used as the floating market-rate proxy.
    * SPY straddle: current SPY, strike, remaining tenor, DGS10, and VIX are
      passed into the option-pricing helper.

    These component values are current mark-to-market values, not risk
    measures. They become risk measures only after historical shocks are applied
    and the resulting scenario loss distribution is built.
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

    return {
        "linear_value": linear_value,
        "irs_value": float(irs_value),
        "straddle_value": straddle_value,
    }


def scenario_loss_distribution(
    snapshot: pd.Series,
    state: pd.Series,
    shock_window: pd.DataFrame,
    linear_shares: np.ndarray,
    price_columns: list[str],
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Reprice the current snapshot under all historical shocks in the window.

    This is the central Historical Simulation function. It keeps today's
    portfolio state fixed, takes each historical row in `shock_window`, applies
    that row as a hypothetical one-day market move, and reprices the whole
    portfolio under that hypothetical tomorrow.

    Scenario construction:

    * Linear prices are shocked with exp(historical log return).
    * DGS10 is shifted by the historical absolute daily yield change.
    * VIX is shocked with exp(historical VIX log return).
    * The straddle tenor is reduced by one trading day. If the current option
      expires over the one-day horizon, the scenario values the newly opened
      ATM straddle so the scenario convention matches the realised P&L
      convention.

    The output is a full empirical distribution of 500 one-day losses for the
    current snapshot when the baseline `WINDOW = 500` is used.

    Returns
    -------
    losses : np.ndarray
        Scenario losses (positive values = losses).
    pnl : np.ndarray
        Scenario P&L (positive values = gains).
    current_total_value : float
        Current mark-to-market value of the portfolio snapshot.
    """
    scenario_components, current_components = scenario_component_values(
        snapshot=snapshot,
        state=state,
        shock_window=shock_window,
        linear_shares=linear_shares,
        price_columns=price_columns,
    )

    current_total_value = float(sum(current_components.values()))
    scenario_total_values = sum(scenario_components.values())
    pnl = scenario_total_values - current_total_value
    losses = current_total_value - scenario_total_values

    return losses, pnl, current_total_value


def scenario_component_values(
    snapshot: pd.Series,
    state: pd.Series,
    shock_window: pd.DataFrame,
    linear_shares: np.ndarray,
    price_columns: list[str],
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """
    Reprice each portfolio component under the same historical shock window.

    This helper is the single place where scenario market levels are built from
    historical shocks. Both total portfolio VaR and component diagnostics call
    it, so conventions for price shocks, DGS10 shocks, VIX shocks, and straddle
    roll handling cannot drift between the two code paths.

    Returns
    -------
    scenario_components : dict[str, np.ndarray]
        Scenario values for linear book, IRS, and straddle.
    current_components : dict[str, float]
        Current mark-to-market values for the same components.
    """
    current_components = current_component_values(snapshot, state, linear_shares, price_columns)

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
    roll_today = float(state["tenor_years"]) <= (1.0 / TRADING_DAYS + 1e-12)
    if roll_today:
        # The one-day horizon crosses the option roll: scenarios value the
        # newly opened ATM straddle, matching the realised P&L convention.
        scenario_straddle_prices = price_straddle_position(
            scenario_spy,
            scenario_spy,
            STRADDLE_DAYS / TRADING_DAYS,
            shocked_rates,
            scenario_vol,
        )
    else:
        scenario_straddle_prices = price_straddle_position(
            scenario_spy,
            float(state["strike_spy"]),
            horizon_tenor,
            shocked_rates,
            scenario_vol,
        )
    scenario_straddle_values = np.asarray(scenario_straddle_prices, dtype=float) * STRADDLE_SHARES

    scenario_components = {
        "linear": np.asarray(scenario_linear_values, dtype=float),
        "irs": np.asarray(scenario_irs_values, dtype=float),
        "straddle": np.asarray(scenario_straddle_values, dtype=float),
    }
    return scenario_components, current_components


def scenario_component_pnl_distribution(
    snapshot: pd.Series,
    state: pd.Series,
    shock_window: pd.DataFrame,
    linear_shares: np.ndarray,
    price_columns: list[str],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, float]]:
    """
    Reprice each portfolio component under all historical shocks.

    This function mirrors `scenario_loss_distribution`, but keeps the linear,
    IRS, and straddle scenario P&L separate. It is used for diagnostics, for
    example to check which component has large standalone tail losses or which
    component drove an extreme realised-loss day.

    Important: component-level VaRs computed from these arrays are standalone
    diagnostics. They are not additive contributions to total portfolio VaR,
    because the worst 1% of losses for each component can occur on different
    historical shock dates.

    Returns
    -------
    component_pnl : dict[str, np.ndarray]
        Scenario P&L by component.
    component_losses : dict[str, np.ndarray]
        Scenario losses by component.
    current_components : dict[str, float]
        Current mark-to-market values by component.
    """
    component_values, current_components = scenario_component_values(
        snapshot=snapshot,
        state=state,
        shock_window=shock_window,
        linear_shares=linear_shares,
        price_columns=price_columns,
    )
    current_value_map = {
        "linear": current_components["linear_value"],
        "irs": current_components["irs_value"],
        "straddle": current_components["straddle_value"],
    }
    component_pnl = {
        name: values - current_value_map[name]
        for name, values in component_values.items()
    }
    component_losses = {
        name: current_value_map[name] - values
        for name, values in component_values.items()
    }
    return component_pnl, component_losses, current_components


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

    In other words, Historical Simulation scenarios produce the forecast loss
    distribution, while the actual next-day market move produces the realised
    P&L used for backtesting. Keeping those two paths separate makes the
    backtest easier to defend: forecasts come from historical scenarios, but
    exceptions are based on what actually happened the next trading day.
    """
    current_total_value, _ = current_portfolio_value(snapshot, state, linear_shares, price_columns)

    next_prices = next_snapshot[price_columns].to_numpy(dtype=float)
    next_linear_value = float(np.dot(linear_shares, next_prices))

    next_rate = max(float(next_snapshot["DGS10"]) / 100.0, 0.0)
    next_irs_value, _ = price_irs(IRS_NOTIONAL, IRS_FIXED_RATE, next_rate)

    roll_today = float(state["tenor_years"]) <= (1.0 / TRADING_DAYS + 1e-12)
    if roll_today:
        next_straddle_price = price_straddle_position(
            float(next_snapshot["SPY"]),
            float(next_snapshot["SPY"]),
            STRADDLE_DAYS / TRADING_DAYS,
            next_rate,
            max(float(next_snapshot["VIX"]) / 100.0, 1e-6),
        )
    else:
        horizon_tenor = max(float(state["tenor_years"]) - 1.0 / TRADING_DAYS, 0.0)
        next_straddle_price = price_straddle_position(
            float(next_snapshot["SPY"]),
            float(state["strike_spy"]),
            horizon_tenor,
            next_rate,
            max(float(next_snapshot["VIX"]) / 100.0, 1e-6),
        )
    next_straddle_value = float(next_straddle_price) * STRADDLE_SHARES

    next_total_value = next_linear_value + float(next_irs_value) + next_straddle_value
    pnl = next_total_value - current_total_value
    loss = -pnl
    return float(loss), float(pnl)


def realised_next_day_component_pnl(
    snapshot: pd.Series,
    next_snapshot: pd.Series,
    state: pd.Series,
    linear_shares: np.ndarray,
    price_columns: list[str],
) -> dict[str, float]:
    """
    Reprice each component against actual next-day market levels.

    The result is a realised component P&L map for the same one-day horizon
    used in the total backtest. It supports the extreme-day diagnostics in
    `historical_sim_analysis.py`, where the project asks whether the largest
    losses were mainly driven by the linear book, the IRS, or the straddle.
    """
    current_components = current_component_values(snapshot, state, linear_shares, price_columns)

    next_prices = next_snapshot[price_columns].to_numpy(dtype=float)
    next_linear_value = float(np.dot(linear_shares, next_prices))

    next_rate = max(float(next_snapshot["DGS10"]) / 100.0, 0.0)
    next_irs_value, _ = price_irs(IRS_NOTIONAL, IRS_FIXED_RATE, next_rate)

    roll_today = float(state["tenor_years"]) <= (1.0 / TRADING_DAYS + 1e-12)
    if roll_today:
        next_straddle_price = price_straddle_position(
            float(next_snapshot["SPY"]),
            float(next_snapshot["SPY"]),
            STRADDLE_DAYS / TRADING_DAYS,
            next_rate,
            max(float(next_snapshot["VIX"]) / 100.0, 1e-6),
        )
    else:
        horizon_tenor = max(float(state["tenor_years"]) - 1.0 / TRADING_DAYS, 0.0)
        next_straddle_price = price_straddle_position(
            float(next_snapshot["SPY"]),
            float(state["strike_spy"]),
            horizon_tenor,
            next_rate,
            max(float(next_snapshot["VIX"]) / 100.0, 1e-6),
        )
    next_straddle_value = float(next_straddle_price) * STRADDLE_SHARES

    return {
        "linear": next_linear_value - current_components["linear_value"],
        "irs": float(next_irs_value) - current_components["irs_value"],
        "straddle": next_straddle_value - current_components["straddle_value"],
    }


# =============================================================================
# HISTORICAL SIMULATION
# =============================================================================

def empirical_var_es(losses: np.ndarray, alpha: float) -> tuple[float, float, int]:
    """
    Compute empirical Historical Simulation VaR and ES from scenario losses.

    For a 99% VaR, the relevant tail probability is 1%. With a 500-day window,
    that gives ceil(500 * 1%) = 5 tail observations. The losses are sorted from
    smallest to largest, the worst `tail_count` losses are selected, and:

    * VaR is the first value inside that worst-loss tail, i.e. the threshold.
    * ES is the average value of the worst-loss tail.

    This exact order-statistic convention is transparent and easy to explain,
    but it also shows a limitation of Historical Simulation: at 99% confidence
    and 500 observations, the entire tail estimate is based on only 5 scenarios.
    """
    ordered = np.sort(np.asarray(losses, dtype=float))
    tail_probability = Decimal("1") - Decimal(str(alpha))
    tail_count_decimal = (Decimal(len(ordered)) * tail_probability).to_integral_value(rounding=ROUND_CEILING)
    tail_count = max(1, int(tail_count_decimal))
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

    The loop is indexed so that each forecast date uses only information that
    would have been known before that forecast date:

    * `snapshot_date` is the market state at the end of the previous trading
      day.
    * `forecast_date` is the next trading day whose realised P&L is tested.
    * `historical_shocks = shock_frame.iloc[t - window:t]` contains the rolling
      historical window ending before the forecast date.

    For every forecast date, the function stores the HistSim VaR, ES, number of
    tail scenarios, current portfolio value, realised next-day loss/P&L, and
    whether the realised P&L breached VaR.
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
        # shock_frame starts one row after market_levels, so this slice uses
        # only historical shocks observed strictly before forecast_date.
        historical_shocks = shock_frame.iloc[t - window:t]

        scenario_losses, _, current_total_value = scenario_loss_distribution(
            snapshot=snapshot,
            state=state,
            shock_window=historical_shocks,
            linear_shares=linear_shares,
            price_columns=price_columns,
        )

        var_t, es_t, tail_count = empirical_var_es(scenario_losses, alpha)
        var_positive = max(var_t, 0.0)
        es_positive = max(es_t, 0.0)
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
                "VaR_HistSim": var_positive,
                "ES_HistSim": es_positive,
                "tail_scenarios": tail_count,
                "quantile_method": "empirical_order_statistic",
                "historical_window_mean_loss": float(scenario_losses.mean()),
                "historical_window_vol_loss": float(scenario_losses.std(ddof=1)),
                "realised_loss": realised_loss_t,
                "realised_pnl": realised_pnl_t,
                "exception": int(realised_pnl_t < -var_positive),
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

    The plot is intended for presentation use: it shows realised losses,
    VaR/ES, exception points, and crisis annotations in one time-series view.
    It helps explain the main HistSim result visually: average calibration can
    be good even when exceptions are visibly clustered around stress periods.
    """
    OUTPUT_FIGS.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13, 5))

    realised_losses = results["realised_loss"].clip(lower=0.0)
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
        realised_losses,
        color="#2e86ab",
        linewidth=0.8,
        alpha=0.9,
        label="Realised loss",
    )
    ax.scatter(exceptions, realised_losses.loc[exceptions], color="#c0392b", s=18, zorder=5, label="Exceptions")

    max_loss_date = results["realised_loss"].idxmax()
    max_loss_value = float(results.loc[max_loss_date, "realised_loss"])
    ax.annotate(
        f"Max realised loss\n{pd.Timestamp(max_loss_date).date()}",
        xy=(max_loss_date, max_loss_value),
        xytext=(10, 16),
        textcoords="offset points",
        fontsize=8,
        color="#444444",
        arrowprops={"arrowstyle": "->", "color": "#666666", "lw": 0.8},
    )

    ax.set_title("Historical Simulation VaR Evolution | Full Repricing", loc="left", fontweight="bold")
    ax.set_ylabel("USD loss")
    ax.set_xlabel("Date")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    _add_crisis_annotations(ax, results.index)
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

    Files written by this function:

    * `data/processed/var_historical_sim.csv`: full daily HistSim time series.
    * `outputs/tables/backtest_historical_sim.csv`: daily backtest detail.
    * `outputs/tables/backtest_summary_historical_sim.csv`: one-row statistical
      summary with Kupiec, Christoffersen independence, and conditional
      coverage results.

    The summary also records modelling conventions such as the 500-day window,
    full-repricing implementation, static linear shares, and DGS10/VIX proxy
    choices. These fields are useful when comparing HistSim with the other VaR
    methods in the project.
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
                "binomial_acceptance_lower_inclusive": backtest_result.binomial_acceptance_lower_inclusive,
                "binomial_acceptance_upper_inclusive": backtest_result.binomial_acceptance_upper_inclusive,
                "binomial_pvalue": backtest_result.binomial_pvalue,
                "binomial_reject_5pct": backtest_result.binomial_reject,
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

    Execution order:

    1. Load aligned market levels, risk-factor shocks, and straddle state.
    2. Convert configured weights into fixed linear share quantities.
    3. Compute the rolling HistSim VaR/ES and realised next-day P&L.
    4. Run the backtest against the realised P&L series.
    5. Save plots, daily detail tables, and summary tables.
    """
    market_levels, shock_frame, straddle_state = load_market_data()
    price_columns = [column for column in market_levels.columns if column in LINEAR_ASSETS]
    missing = LINEAR_ASSETS - set(price_columns)
    if missing:
        raise ValueError(f"Missing linear assets in market data: {sorted(missing)}")

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
