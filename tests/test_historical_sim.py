from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DATA_DIR = REPO_ROOT / "src" / "data"

for path in (REPO_ROOT, SRC_DATA_DIR):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from src.data.compute_pnl import compute_total_pnl
from src.data.portfolio_pricing import price_irs, price_straddle_position, swap_annuity
from src.var_methods.historical_sim import (
    STRADDLE_DAYS,
    TRADING_DAYS,
    V0,
    build_linear_shares,
    empirical_var_es,
    realised_next_day_pnl,
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
    ) * 2000.0
    current_one_day_value = float(
        price_straddle_position(
            500.0,
            500.0,
            1.0 / TRADING_DAYS,
            0.04,
            0.20,
        )
    ) * 2000.0

    # On a roll day with unchanged market levels, the portfolio should record
    # the repricing jump from the expiring straddle into the newly opened ATM
    # straddle rather than collapsing the position to zero value.
    assert loss < 0.0
    assert np.isclose(-pnl, loss)
    assert np.isclose(pnl, new_straddle_value - current_one_day_value, rtol=1e-6)
