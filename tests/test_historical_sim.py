from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DATA_DIR = REPO_ROOT / "src" / "data"

for path in (REPO_ROOT, SRC_DATA_DIR):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from src.data.compute_pnl import compute_total_pnl
from backtesting.backtest import run_backtest
from src.data.portfolio_pricing import (
    build_straddle_state,
    price_irs,
    price_straddle_position,
    swap_annuity,
)
from src.var_methods.historical_sim import (
    STRADDLE_DAYS,
    STRADDLE_SHARES,
    TRADING_DAYS,
    V0,
    build_linear_shares,
    compute_historical_sim_var,
    empirical_var_es,
    realised_next_day_pnl,
    scenario_loss_distribution,
)


def test_empirical_var_es_uses_five_tail_scenarios_for_500_day_99pct_window():
    losses = np.arange(1.0, 501.0)

    var_value, es_value, tail_count = empirical_var_es(losses, 0.99)

    assert tail_count == 5
    assert var_value == 496.0
    assert es_value == 498.0


def test_price_irs_matches_discount_factor_annuity_definition():
    rate = 0.04
    notional = 1_000_000
    fixed_rate = 0.03

    value, dv01 = price_irs(notional, fixed_rate, rate, maturity=10)
    expected_annuity = sum(1.0 / ((1.0 + rate) ** t) for t in range(1, 11))

    assert np.isclose(swap_annuity(rate, maturity=10), expected_annuity)
    assert np.isclose(dv01, notional * expected_annuity * 0.0001)
    assert np.isclose(value, notional * (rate - fixed_rate) * expected_annuity)


def test_price_irs_supports_vector_inputs():
    rates = np.array([0.02, 0.04, 0.06])
    values, dv01 = price_irs(1_000_000, 0.03, rates, maturity=10)

    assert values.shape == rates.shape
    assert dv01.shape == rates.shape
    # Higher rates lower the fixed-leg annuity, so the dollar value of 1bp falls.
    assert np.all(np.diff(dv01) < 0)


def test_compute_total_pnl_uses_static_inception_shares():
    prices = pd.DataFrame(
        {
            "A": [10.0, 12.0, 20.0],
            "B": [20.0, 18.0, 20.0],
        },
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
    )
    instrument_pnl = pd.DataFrame(
        {
            "pnl_irs": [0.0, 0.0],
            "pnl_straddle": [0.0, 0.0],
        },
        index=prices.index[1:],
    )
    weights = {"A": 0.5, "B": 0.5}

    total = compute_total_pnl(prices, instrument_pnl, weights, portfolio_value=100.0)

    # Initial shares: 5 of A and 2.5 of B
    # Day 2 pnl: 5*(12-10) + 2.5*(18-20) = 10 - 5 = 5
    # Day 3 pnl: 5*(20-12) + 2.5*(20-18) = 40 + 5 = 45
    expected = pd.Series([5.0, 45.0], index=prices.index[1:], name="pnl_linear")

    pd.testing.assert_series_equal(total["pnl_linear"], expected)


def test_roll_day_realised_pnl_reopens_new_atm_straddle():
    price_columns = ["EURUSD", "GLD", "IEF", "SPY"]
    weights = pd.Series(
        {"EURUSD": 0.25, "GLD": 0.25, "IEF": 0.25, "SPY": 0.25},
        dtype=float,
    )
    initial_prices = pd.DataFrame([{"EURUSD": 1.20, "GLD": 100.0, "IEF": 95.0, "SPY": 500.0}])
    linear_shares = build_linear_shares(initial_prices, weights, V0)

    snapshot = pd.Series(
        {
            "EURUSD": 1.20,
            "GLD": 100.0,
            "IEF": 95.0,
            "SPY": 500.0,
            "VIX": 20.0,
            "DGS10": 4.0,
        }
    )
    next_snapshot = pd.Series(
        {
            "EURUSD": 1.20,
            "GLD": 100.0,
            "IEF": 95.0,
            "SPY": 500.0,
            "VIX": 20.0,
            "DGS10": 4.0,
        }
    )
    state = pd.Series(
        {
            "strike_spy": 500.0,
            "tenor_years": 1.0 / TRADING_DAYS,
            "days_held": STRADDLE_DAYS - 1,
            "rolled_today": False,
        }
    )

    loss, pnl = realised_next_day_pnl(
        snapshot=snapshot,
        next_snapshot=next_snapshot,
        state=state,
        linear_shares=linear_shares,
        price_columns=price_columns,
    )

    new_straddle_value = float(
        price_straddle_position(
            500.0,
            500.0,
            STRADDLE_DAYS / TRADING_DAYS,
            0.04,
            0.20,
        )
    ) * STRADDLE_SHARES
    current_one_day_value = float(
        price_straddle_position(
            500.0,
            500.0,
            1.0 / TRADING_DAYS,
            0.04,
            0.20,
        )
    ) * STRADDLE_SHARES

    # On a roll day with unchanged market levels, the portfolio should record
    # the repricing jump from the expiring straddle into the newly opened ATM
    # straddle rather than collapsing the position to zero value.
    assert loss < 0.0
    assert np.isclose(-pnl, loss)
    assert np.isclose(pnl, new_straddle_value - current_one_day_value, rtol=1e-6)


def test_roll_day_scenario_distribution_reopens_new_atm_straddle():
    price_columns = ["EURUSD", "GLD", "IEF", "SPY"]
    weights = pd.Series(
        {"EURUSD": 0.25, "GLD": 0.25, "IEF": 0.25, "SPY": 0.25},
        dtype=float,
    )
    initial_prices = pd.DataFrame([{"EURUSD": 1.20, "GLD": 100.0, "IEF": 95.0, "SPY": 500.0}])
    linear_shares = build_linear_shares(initial_prices, weights, V0)

    snapshot = pd.Series(
        {
            "EURUSD": 1.20,
            "GLD": 100.0,
            "IEF": 95.0,
            "SPY": 500.0,
            "VIX": 20.0,
            "DGS10": 4.0,
        }
    )
    state = pd.Series(
        {
            "strike_spy": 500.0,
            "tenor_years": 1.0 / TRADING_DAYS,
            "days_held": STRADDLE_DAYS - 1,
            "rolled_today": False,
        }
    )

    shock_window = pd.DataFrame(
        {
            "EURUSD_ret": [0.0],
            "GLD_ret": [0.0],
            "IEF_ret": [0.0],
            "SPY_ret": [0.0],
            "VIX_ret": [0.0],
            "DGS10_change": [0.0],
        },
        index=pd.to_datetime(["2024-01-02"]),
    )

    losses, pnl, current_total_value = scenario_loss_distribution(
        snapshot=snapshot,
        state=state,
        shock_window=shock_window,
        linear_shares=linear_shares,
        price_columns=price_columns,
    )

    current_one_day_value = float(
        price_straddle_position(
            500.0,
            500.0,
            1.0 / TRADING_DAYS,
            0.04,
            0.20,
        )
    ) * STRADDLE_SHARES
    new_straddle_value = float(
        price_straddle_position(
            500.0,
            500.0,
            STRADDLE_DAYS / TRADING_DAYS,
            0.04,
            0.20,
        )
    ) * STRADDLE_SHARES

    assert current_total_value > 0.0
    assert losses.shape == (1,)
    assert pnl.shape == (1,)
    assert np.isclose(pnl[0], new_straddle_value - current_one_day_value, rtol=1e-6)
    assert np.isclose(losses[0], -(new_straddle_value - current_one_day_value), rtol=1e-6)


def test_build_straddle_state_rolls_on_day_30():
    dates = pd.date_range("2024-01-01", periods=35, freq="B")
    spy = pd.Series(np.linspace(100.0, 134.0, len(dates)), index=dates)

    state = build_straddle_state(spy, straddle_days=STRADDLE_DAYS, trading_days=TRADING_DAYS)

    assert bool(state.iloc[29]["rolled_today"]) is False
    assert bool(state.iloc[30]["rolled_today"]) is True
    assert state.iloc[30]["days_held"] == 0
    assert state.iloc[30]["strike_spy"] == spy.iloc[30]


def test_run_backtest_smoke_has_known_kupiec_result():
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    pnl = pd.Series(0.0, index=dates)
    pnl.iloc[50] = -2.0
    var = pd.Series(1.0, index=dates)

    result = run_backtest(pnl=pnl, var=var, confidence=0.99, method_name="HistSimTest")

    assert result.T == 100
    assert result.N == 1
    assert result.expected_N == pytest.approx(1.0)
    assert result.lr_uc == pytest.approx(0.0)
    assert result.reject_uc is False


def test_compute_historical_sim_var_main_loop_flags_known_exception():
    dates = pd.date_range("2024-01-01", periods=510, freq="B")
    price_columns = ["EURUSD", "GLD", "IEF", "SPY"]
    market_levels = pd.DataFrame(
        {
            "EURUSD": 1.20,
            "GLD": 100.0,
            "IEF": 95.0,
            "SPY": 100.0,
            "VIX": 20.0,
            "DGS10": 4.0,
        },
        index=dates,
    )
    market_levels.loc[dates[501], "SPY"] = 80.0

    shock_frame = pd.DataFrame(
        {
            "EURUSD_ret": 0.0,
            "GLD_ret": 0.0,
            "IEF_ret": 0.0,
            "SPY_ret": 0.0,
            "VIX_ret": 0.0,
            "DGS10_change": 0.0,
        },
        index=dates[1:],
    )
    straddle_state = pd.DataFrame(
        {
            "strike_spy": 100.0,
            "tenor_years": STRADDLE_DAYS / TRADING_DAYS,
            "days_held": 0,
            "rolled_today": False,
        },
        index=dates,
    )
    weights = pd.Series(
        {"EURUSD": 0.25, "GLD": 0.25, "IEF": 0.25, "SPY": 0.25},
        dtype=float,
    )
    linear_shares = build_linear_shares(market_levels[price_columns], weights, V0)

    results, realised_pnl = compute_historical_sim_var(
        market_levels=market_levels,
        shock_frame=shock_frame,
        straddle_state=straddle_state,
        price_columns=price_columns,
        linear_shares=linear_shares,
        window=500,
        alpha=0.99,
    )

    assert len(results) == len(shock_frame) - 500
    assert len(realised_pnl) == len(results)
    assert {"VaR_HistSim", "ES_HistSim", "tail_scenarios", "exception"}.issubset(results.columns)
    assert results.iloc[0]["tail_scenarios"] == 5
    assert results.iloc[0]["exception"] == 1
