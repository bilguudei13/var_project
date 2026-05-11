#!/usr/bin/env python3
"""
validate_mapping.py

Checks that the portfolio mapping used by the Monte Carlo model matches the
reference P&L construction from compute_pnl.py.

What it validates:
1) Linear book P&L
2) IRS P&L
3) Straddle P&L
4) Total P&L
5) Optional smoke test: vectorized MC pricing wrappers vs production pricers

This script is deterministic and does not require running the MC simulator.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------
# Paths / imports
# ---------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DATA_DIR = REPO_ROOT / "src" / "data"
if str(SRC_DATA_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DATA_DIR))

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import (
    WEIGHTS_DICT,
    V0,
    IRS_NOTIONAL,
    IRS_FIXED_RATE,
    STRADDLE_DAYS,
    STRADDLE_SHARES,
    RAW_DIR,
    PROCESSED_DIR,
)

from compute_pnl import compute_instrument_pnl, compute_total_pnl
from portfolio_pricing import build_straddle_state, price_irs, price_straddle_position

# Optional: smoke-test the MC wrappers if the audit file is available.
try:
    from mc_garch_t_copula import price_irs_vec, price_straddle_vec
except Exception:
    price_irs_vec = None
    price_straddle_vec = None


RAW_PATH = REPO_ROOT / RAW_DIR
PROCESSED_PATH = REPO_ROOT / PROCESSED_DIR
OUTPUT_TABLES = REPO_ROOT / "outputs" / "tables"


LINEAR_TICKERS = ["EURUSD", "GLD", "IEF", "SPY"]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def load_data():
    prices = pd.read_csv(RAW_PATH / "prices.csv", index_col=0, parse_dates=True)
    vix = pd.read_csv(RAW_PATH / "vix.csv", index_col=0, parse_dates=True).squeeze()
    dgs10 = pd.read_csv(RAW_PATH / "dgs10.csv", index_col=0, parse_dates=True).squeeze()

    common = prices.index.intersection(vix.index).intersection(dgs10.index).sort_values()
    prices = prices.loc[common].copy()
    vix = vix.loc[common].copy()
    dgs10 = dgs10.loc[common].copy()

    print(f"Data loaded : {len(common)} days  ({common[0].date()} → {common[-1].date()})")
    return prices, vix, dgs10


def build_linear_shares_from_prices(prices: pd.DataFrame) -> pd.Series:
    """
    Mirror the compute_pnl.py convention:
    shares_j = V0 * w_j / initial_price_j
    """
    price_frame = prices.rename(columns={"EURUSD=X": "EURUSD"}).sort_index()[LINEAR_TICKERS]

    weights_aligned = {
        ("EURUSD" if k == "EURUSD=X" else k): float(v)
        for k, v in WEIGHTS_DICT.items()
    }
    w = pd.Series(weights_aligned)[LINEAR_TICKERS].astype(float)

    initial_prices = price_frame.iloc[0]
    linear_shares = V0 * w / initial_prices
    return linear_shares


def safe_scalar(x) -> float:
    return float(np.asarray(x).squeeze())


def smoke_test_pricers():
    """
    Optional sanity check:
    compare vectorized MC wrappers to production pricers on one example input.
    """
    if price_irs_vec is None or price_straddle_vec is None:
        print("Optional wrapper smoke test skipped (MC wrappers not importable).")
        return

    print("=" * 65)
    print("SMOKE TEST — pricing wrappers")
    print("=" * 65)

    # IRS wrapper vs production pricer
    test_rate = 0.023
    irs_prod, _ = price_irs(IRS_NOTIONAL, IRS_FIXED_RATE, test_rate)
    irs_vec = safe_scalar(price_irs_vec(IRS_NOTIONAL, IRS_FIXED_RATE, test_rate))
    irs_diff = abs(irs_prod - irs_vec)

    # Straddle wrapper vs production pricer
    S = 100.0
    K = 100.0
    T = 0.5
    r = 0.02
    sigma = 0.25
    strad_prod = safe_scalar(price_straddle_position(S, K, T, r, sigma))
    strad_vec = safe_scalar(price_straddle_vec(S, K, T, r, sigma))
    strad_diff = abs(strad_prod - strad_vec)

    print(f"IRS     : prod={irs_prod:,.8f}  vec={irs_vec:,.8f}  diff={irs_diff:.3e}")
    print(f"Straddle: prod={strad_prod:,.8f}  vec={strad_vec:,.8f}  diff={strad_diff:.3e}")

    if irs_diff > 1e-8:
        print("  WARNING: IRS wrapper does not match production pricer closely.")
    if strad_diff > 1e-8:
        print("  WARNING: Straddle wrapper does not match production pricer closely.")
    print()


def validate_mapping(prices: pd.DataFrame, vix: pd.Series, dgs10: pd.Series) -> pd.DataFrame:
    """
    Compare manual recomputation of the P&L components against compute_pnl.py.
    """
    print("=" * 65)
    print("VALIDATION — P&L / risk-mapping consistency")
    print("=" * 65)

    # Reference P&L from compute_pnl.py
    instrument_ref = compute_instrument_pnl(prices, vix, dgs10)
    total_ref = compute_total_pnl(prices, instrument_ref, WEIGHTS_DICT, V0)

    # Manual recomputation setup
    price_frame = prices.rename(columns={"EURUSD=X": "EURUSD"}).sort_index()[LINEAR_TICKERS]
    common = price_frame.index.intersection(vix.index).intersection(dgs10.index).sort_values()
    price_frame = price_frame.loc[common]
    vix = vix.loc[common]
    dgs10 = dgs10.loc[common]

    linear_shares = build_linear_shares_from_prices(prices)

    # Straddle state from the same helper used by compute_pnl.py
    straddle_state = build_straddle_state(price_frame["SPY"], straddle_days=STRADDLE_DAYS)

    rows = []
    for t in range(1, len(common)):
        dt = common[t]

        # --- Linear ---
        pnl_linear_manual = ((price_frame.iloc[t] - price_frame.iloc[t - 1]) * linear_shares).sum()

        # --- IRS ---
        v_today, _ = price_irs(IRS_NOTIONAL, IRS_FIXED_RATE, float(dgs10.iloc[t]) / 100.0)
        v_prev, _ = price_irs(IRS_NOTIONAL, IRS_FIXED_RATE, float(dgs10.iloc[t - 1]) / 100.0)
        pnl_irs_manual = v_today - v_prev

        # --- Straddle ---
        prev_state = straddle_state.iloc[t - 1]
        today_state = straddle_state.iloc[t]

        rf_today = max(float(dgs10.iloc[t]) / 100.0, 0.0)
        rf_prev = max(float(dgs10.iloc[t - 1]) / 100.0, 0.0)

        value_today = safe_scalar(
            price_straddle_position(
                float(price_frame["SPY"].iloc[t]),
                float(today_state["strike_spy"]),
                float(today_state["tenor_years"]),
                rf_today,
                float(vix.iloc[t]) / 100.0,
            )
        ) * STRADDLE_SHARES

        value_prev = safe_scalar(
            price_straddle_position(
                float(price_frame["SPY"].iloc[t - 1]),
                float(prev_state["strike_spy"]),
                float(prev_state["tenor_years"]),
                rf_prev,
                float(vix.iloc[t - 1]) / 100.0,
            )
        ) * STRADDLE_SHARES

        pnl_straddle_manual = value_today - value_prev

        # --- Total ---
        pnl_total_manual = pnl_linear_manual + pnl_irs_manual + pnl_straddle_manual

        # --- Reference values from compute_pnl.py ---
        ref_linear = float(total_ref.loc[dt, "pnl_linear"])
        ref_irs = float(instrument_ref.loc[dt, "pnl_irs"])
        ref_straddle = float(instrument_ref.loc[dt, "pnl_straddle"])
        ref_total = float(total_ref.loc[dt, "pnl_total"])

        rows.append({
            "date": dt,
            "manual_linear": pnl_linear_manual,
            "ref_linear": ref_linear,
            "abs_err_linear": abs(pnl_linear_manual - ref_linear),

            "manual_irs": pnl_irs_manual,
            "ref_irs": ref_irs,
            "abs_err_irs": abs(pnl_irs_manual - ref_irs),

            "manual_straddle": pnl_straddle_manual,
            "ref_straddle": ref_straddle,
            "abs_err_straddle": abs(pnl_straddle_manual - ref_straddle),

            "manual_total": pnl_total_manual,
            "ref_total": ref_total,
            "abs_err_total": abs(pnl_total_manual - ref_total),
        })

    diag = pd.DataFrame(rows).set_index("date")

    # Summary
    def summarize(col: str) -> tuple[float, float]:
        return float(diag[col].max()), float(diag[col].mean())

    max_lin, mean_lin = summarize("abs_err_linear")
    max_irs, mean_irs = summarize("abs_err_irs")
    max_str, mean_str = summarize("abs_err_straddle")
    max_tot, mean_tot = summarize("abs_err_total")

    print("\nMax / mean absolute error vs compute_pnl.py")
    print(f"Linear   : max={max_lin:.6e}  mean={mean_lin:.6e}")
    print(f"IRS      : max={max_irs:.6e}  mean={mean_irs:.6e}")
    print(f"Straddle : max={max_str:.6e}  mean={mean_str:.6e}")
    print(f"Total    : max={max_tot:.6e}  mean={mean_tot:.6e}")

    worst = diag.sort_values("abs_err_total", ascending=False).head(10)
    print("\nWorst 10 dates by total P&L mismatch:")
    print(
        worst[[
            "manual_total", "ref_total", "abs_err_total",
            "manual_linear", "ref_linear", "abs_err_linear",
            "manual_irs", "ref_irs", "abs_err_irs",
            "manual_straddle", "ref_straddle", "abs_err_straddle",
        ]].to_string(float_format="{:.6f}".format)
    )

    # Quick pass/fail flags
    tol = 1e-8
    ok = (
        max_lin < tol and
        max_irs < tol and
        max_str < tol and
        max_tot < tol
    )

    print("\nOverall mapping check:", "PASS ✓" if ok else "FAIL ✗")
    print("=" * 65)

    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
    diag_path = OUTPUT_TABLES / "validate_pricing_functions_diagnostics.csv"
    diag.to_csv(diag_path)
    print(f"Diagnostics saved -> {diag_path.relative_to(REPO_ROOT)}")

    return diag


def main() -> None:
    prices, vix, dgs10 = load_data()
    smoke_test_pricers()
    validate_mapping(prices, vix, dgs10)


if __name__ == "__main__":
    main()