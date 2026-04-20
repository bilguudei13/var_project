from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
VAR_METHODS_DIR = REPO_ROOT / "src" / "var_methods"
SRC_DATA_DIR = REPO_ROOT / "src" / "data"

for path in (REPO_ROOT, VAR_METHODS_DIR, SRC_DATA_DIR):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from src.var_methods.historical_sim import (
    empirical_var_es,
    realised_next_day_component_pnl,
    realised_next_day_pnl,
    scenario_component_pnl_distribution,
    scenario_loss_distribution,
)
from src.var_methods.historical_sim_analysis import (
    build_extreme_day_table,
    compute_component_hs_table,
    compute_exception_severity,
    compute_window_sensitivity,
    filter_subperiod,
    summarize_tail_scenario_usage,
)


def _toy_hist_sim_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], np.ndarray]:
    dates = pd.date_range("2024-01-01", periods=6, freq="B")
    market_levels = pd.DataFrame(
        {
            "EURUSD": [1.00, 0.99, 1.01, 0.95, 0.70, 0.70],
            "GLD": [100.0] * 6,
            "IEF": [100.0] * 6,
            "SPY": [100.0] * 6,
            "VIX": [1e-6] * 6,
            "DGS10": [0.0] * 6,
        },
        index=dates,
    )
    shock_frame = pd.DataFrame(
        {
            "EURUSD_ret": np.log(market_levels["EURUSD"] / market_levels["EURUSD"].shift(1)).iloc[1:].to_numpy(),
            "GLD_ret": [0.0] * 5,
            "IEF_ret": [0.0] * 5,
            "SPY_ret": [0.0] * 5,
            "VIX_ret": [0.0] * 5,
            "DGS10_change": [0.0] * 5,
        },
        index=dates[1:],
    )
    straddle_state = pd.DataFrame(
        {
            "strike_spy": [100.0] * 6,
            "tenor_years": [0.0] * 6,
            "days_held": [0] * 6,
            "rolled_today": [False] * 6,
        },
        index=dates,
    )
    return market_levels, shock_frame, straddle_state, ["EURUSD", "GLD", "IEF", "SPY"], np.array([100.0, 0.0, 0.0, 0.0])


def test_exception_severity_uses_only_exception_days():
    realised_loss = pd.Series([8.0, 12.0, 16.0], index=pd.date_range("2024-01-01", periods=3, freq="B"))
    var_series = pd.Series([10.0, 10.0, 10.0], index=realised_loss.index)
    exceptions = pd.Series([0, 1, 1], index=realised_loss.index)

    severity = compute_exception_severity(realised_loss, var_series, exceptions)

    assert np.isclose(severity, 4.0)


def test_filter_subperiod_respects_index_bounds():
    index = pd.date_range("2020-01-01", periods=6, freq="D")
    frame = pd.DataFrame({"value": np.arange(6)}, index=index)

    filtered = filter_subperiod(frame, "2020-01-03", "2020-01-05")

    assert filtered.index.min() == pd.Timestamp("2020-01-03")
    assert filtered.index.max() == pd.Timestamp("2020-01-05")
    assert len(filtered) == 3


def test_tail_scenario_summary_counts_repeated_extremes():
    tail_usage_df = pd.DataFrame(
        {
            "forecast_date": ["2024-01-10", "2024-01-11", "2024-01-12"],
            "snapshot_date": ["2024-01-09", "2024-01-10", "2024-01-11"],
            "historical_shock_date": ["2008-10-10", "2008-10-10", "2020-03-16"],
            "tail_rank": [1, 2, 1],
            "scenario_loss": [30.0, 20.0, 40.0],
            "realised_loss": [10.0, 12.0, 25.0],
        }
    )

    summary = summarize_tail_scenario_usage(tail_usage_df, top_n=5)

    first = summary.iloc[0]
    assert first["historical_shock_date"] == "2008-10-10"
    assert first["times_in_tail"] == 2
    assert first["times_as_worst_scenario"] == 1


def test_component_hs_table_reproduces_linear_component_var_on_toy_data():
    market_levels, shock_frame, straddle_state, price_columns, linear_shares = _toy_hist_sim_inputs()

    component_df, _ = compute_component_hs_table(
        market_levels=market_levels,
        shock_frame=shock_frame,
        straddle_state=straddle_state,
        price_columns=price_columns,
        linear_shares=linear_shares,
        window=2,
        alpha=0.50,
    )

    snapshot = market_levels.loc[market_levels.index[2]]
    state = straddle_state.loc[straddle_state.index[2]]
    component_pnl, component_losses, _ = scenario_component_pnl_distribution(
        snapshot=snapshot,
        state=state,
        shock_window=shock_frame.iloc[0:2],
        linear_shares=linear_shares,
        price_columns=price_columns,
    )
    expected_var, expected_es, _ = empirical_var_es(component_losses["linear"], 0.50)

    first_linear = component_df.loc[component_df["component"] == "linear"].iloc[0]
    assert np.isclose(first_linear["VaR_component"], max(expected_var, 0.0))
    assert np.isclose(first_linear["ES_component"], max(expected_es, 0.0))


def test_component_pnl_and_losses_sum_to_total_portfolio_results():
    market_levels, shock_frame, straddle_state, price_columns, linear_shares = _toy_hist_sim_inputs()
    snapshot = market_levels.loc[market_levels.index[2]]
    next_snapshot = market_levels.loc[market_levels.index[3]]
    state = straddle_state.loc[straddle_state.index[2]]
    shock_window = shock_frame.iloc[0:2]

    component_pnl, component_losses, _ = scenario_component_pnl_distribution(
        snapshot=snapshot,
        state=state,
        shock_window=shock_window,
        linear_shares=linear_shares,
        price_columns=price_columns,
    )
    total_losses, total_pnl, _ = scenario_loss_distribution(
        snapshot=snapshot,
        state=state,
        shock_window=shock_window,
        linear_shares=linear_shares,
        price_columns=price_columns,
    )

    scenario_loss_sum = (
        component_losses["linear"] + component_losses["irs"] + component_losses["straddle"]
    )
    scenario_pnl_sum = component_pnl["linear"] + component_pnl["irs"] + component_pnl["straddle"]
    assert np.allclose(total_losses, scenario_loss_sum)
    assert np.allclose(total_pnl, scenario_pnl_sum)

    realised_loss, realised_pnl = realised_next_day_pnl(
        snapshot=snapshot,
        next_snapshot=next_snapshot,
        state=state,
        linear_shares=linear_shares,
        price_columns=price_columns,
    )
    realised_components = realised_next_day_component_pnl(
        snapshot=snapshot,
        next_snapshot=next_snapshot,
        state=state,
        linear_shares=linear_shares,
        price_columns=price_columns,
    )
    realised_component_pnl_sum = sum(realised_components.values())
    realised_component_loss_sum = -realised_component_pnl_sum
    assert np.isclose(realised_pnl, realised_component_pnl_sum)
    assert np.isclose(realised_loss, realised_component_loss_sum)


def test_window_sensitivity_distinguishes_different_lookback_windows():
    market_levels, shock_frame, straddle_state, price_columns, linear_shares = _toy_hist_sim_inputs()

    summary = compute_window_sensitivity(
        market_levels=market_levels,
        shock_frame=shock_frame,
        straddle_state=straddle_state,
        price_columns=price_columns,
        linear_shares=linear_shares,
        windows=[2, 3],
        alpha=0.50,
    )

    assert list(summary["window"]) == [2, 3]
    assert summary["mean_var"].nunique() == 2


def test_build_extreme_day_table_ranks_largest_driver():
    index = pd.to_datetime(["2024-01-10", "2024-01-11"])
    results = pd.DataFrame(
        {
            "snapshot_date": ["2024-01-09", "2024-01-10"],
            "current_portfolio_value": [1_000_000.0, 1_010_000.0],
            "VaR_HistSim": [20.0, 20.0],
            "ES_HistSim": [30.0, 30.0],
            "realised_loss": [50.0, 40.0],
            "realised_pnl": [-50.0, -40.0],
            "exception": [1, 1],
        },
        index=index,
    )
    component_df = pd.DataFrame(
        {
            "forecast_date": [index[0], index[0], index[0], index[1], index[1], index[1]],
            "component": ["linear", "irs", "straddle"] * 2,
            "realised_loss": [35.0, 5.0, 10.0, 8.0, 22.0, 10.0],
        }
    ).set_index("forecast_date")

    extreme_days = build_extreme_day_table(results, component_df, top_n=2)

    assert extreme_days.iloc[0]["largest_driver"] == "linear"
    assert extreme_days.iloc[1]["largest_driver"] == "irs"
