"""
historical_sim_vol_adjusted.py
==============================

1-day 99% Volatility-Adjusted Historical Simulation VaR.

Idea:
- use the same historical shocks as plain Historical Simulation
- but rescale each old shock by:

    current volatility / historical scenario-day volatility

This helps old calm-period shocks behave more like today if today's market is
much more volatile.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from historical_sim import (
    ALPHA,
    WINDOW,
    COMPONENT_NAMES,
    OUTPUT_FIGS,
    OUTPUT_TABLES,
    PROCESSED_DIR,
    LINEAR_ASSETS,
    aligned_weights,
    build_linear_shares,
    empirical_var_es,
    load_market_data,
    plot_var_evolution,
    realised_next_day_pnl,
    save_results,
    scenario_loss_distribution,
)
from config import V0
from backtesting.backtest import run_backtest
from backtesting.plot_backtest import plot_all


VOL_WINDOW = 60
EPS = 1e-8


def estimate_factor_volatility(shock_frame: pd.DataFrame, vol_window: int = VOL_WINDOW) -> pd.DataFrame:
    """
    Estimate rolling volatility for each shock column.

    Price/VIX shocks are already in return space.
    DGS10_change stays in absolute-change space.
    """
    vol = shock_frame.rolling(vol_window).std(ddof=1)
    vol = vol.replace(0.0, np.nan).ffill().bfill()
    return vol


def adjust_shocks_for_volatility(
    shock_window: pd.DataFrame,
    current_vol: pd.Series,
    scenario_vols: pd.DataFrame,
) -> pd.DataFrame:
    """
    Rescale each historical shock to current volatility level.

    adjusted_shock = raw_shock * current_vol / scenario_day_vol
    """
    ratio = current_vol / scenario_vols
    ratio = ratio.replace([np.inf, -np.inf], np.nan).fillna(1.0)

    adjusted = shock_window * ratio
    return adjusted


def compute_vol_adjusted_hs_var(
    market_levels: pd.DataFrame,
    shock_frame: pd.DataFrame,
    straddle_state: pd.DataFrame,
    price_columns: list[str],
    linear_shares: np.ndarray,
    window: int,
    alpha: float,
    vol_window: int = VOL_WINDOW,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Same structure as plain HistSim, but use volatility-adjusted shocks.
    """
    if len(shock_frame) <= max(window, vol_window):
        raise ValueError(
            f"Not enough observations: need more than {max(window, vol_window)} rows."
        )

    shock_vol = estimate_factor_volatility(shock_frame, vol_window=vol_window)

    records: list[dict[str, float | int | str]] = []
    realised_pnl: dict[pd.Timestamp, float] = {}

    common_dates = market_levels.index

    print("\n" + "=" * 70)
    print("Volatility-Adjusted Historical Simulation VaR")
    print("=" * 70)
    print(f"HS window         : {window}")
    print(f"Vol window        : {vol_window}")
    print(f"Confidence        : {alpha:.0%}")
    print(f"Forecast points   : {len(shock_frame) - window}")

    for t in range(window, len(shock_frame)):
        snapshot_date = common_dates[t]
        forecast_date = shock_frame.index[t]

        snapshot = market_levels.loc[snapshot_date]
        next_snapshot = market_levels.loc[forecast_date]
        state = straddle_state.loc[snapshot_date]

        historical_shocks = shock_frame.iloc[t - window:t].copy()
        historical_vols = shock_vol.iloc[t - window:t].copy()
        current_vol = shock_vol.iloc[t - 1].copy()

        adjusted_shocks = adjust_shocks_for_volatility(
            shock_window=historical_shocks,
            current_vol=current_vol,
            scenario_vols=historical_vols,
        )

        scenario_losses, _, current_total_value = scenario_loss_distribution(
            snapshot=snapshot,
            state=state,
            shock_window=adjusted_shocks,
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

        avg_ratio = float((current_vol / historical_vols.mean()).replace([np.inf, -np.inf], np.nan).mean())

        records.append(
            {
                "snapshot_date": snapshot_date.strftime("%Y-%m-%d"),
                "current_portfolio_value": current_total_value,
                "VaR_HS_VolAdj": var_positive,
                "ES_HS_VolAdj": es_positive,
                "tail_scenarios": tail_count,
                "quantile_method": "empirical_order_statistic",
                "historical_window_mean_loss": float(scenario_losses.mean()),
                "historical_window_vol_loss": float(scenario_losses.std(ddof=1)),
                "avg_vol_ratio": avg_ratio,
                "realised_loss": realised_loss_t,
                "realised_pnl": realised_pnl_t,
                "exception": int(realised_pnl_t < -var_positive),
            }
        )

    results = pd.DataFrame(records, index=shock_frame.index[window:])
    realised_pnl_series = pd.Series(realised_pnl, name="pnl_realised_market_repricing").sort_index()

    print(f"Mean VaR        : ${results['VaR_HS_VolAdj'].mean():,.2f}")
    print(f"Mean ES         : ${results['ES_HS_VolAdj'].mean():,.2f}")
    print(f"Min VaR         : ${results['VaR_HS_VolAdj'].min():,.2f}")
    print(f"Max VaR         : ${results['VaR_HS_VolAdj'].max():,.2f}")

    return results, realised_pnl_series


def save_vol_adjusted_results(results: pd.DataFrame, realised_pnl: pd.Series, backtest_result) -> None:
    """
    Save outputs with different filenames so we do not overwrite plain HistSim.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)

    var_path = PROCESSED_DIR / "var_historical_sim_vol_adjusted.csv"
    results.to_csv(var_path)
    print(f"Saved VaR series      -> {var_path}")

    common = realised_pnl.index.intersection(results.index)
    detail = pd.DataFrame(
        {
            "pnl": realised_pnl.loc[common],
            "actual_loss": -realised_pnl.loc[common],
            "VaR_HS_VolAdj": results.loc[common, "VaR_HS_VolAdj"],
            "ES_HS_VolAdj": results.loc[common, "ES_HS_VolAdj"],
            "tail_scenarios": results.loc[common, "tail_scenarios"],
            "snapshot_date": results.loc[common, "snapshot_date"],
            "current_portfolio_value": results.loc[common, "current_portfolio_value"],
            "avg_vol_ratio": results.loc[common, "avg_vol_ratio"],
            "exception": (realised_pnl.loc[common] < -results.loc[common, "VaR_HS_VolAdj"]).astype(int),
        },
        index=common,
    )

    detail_path = OUTPUT_TABLES / "backtest_historical_sim_vol_adjusted.csv"
    detail.to_csv(detail_path)
    print(f"Saved backtest table  -> {detail_path}")

    summary = pd.DataFrame(
        [
            {
                "method": backtest_result.method_name,
                "confidence": backtest_result.confidence,
                "window": WINDOW,
                "vol_window": VOL_WINDOW,
                "implementation": "volatility-adjusted historical simulation with full repricing",
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
    summary_path = OUTPUT_TABLES / "backtest_summary_historical_sim_vol_adjusted.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary table   -> {summary_path}")


def main() -> None:
    market_levels, shock_frame, straddle_state = load_market_data()
    price_columns = [column for column in market_levels.columns if column in LINEAR_ASSETS]

    weights = aligned_weights(price_columns)
    linear_shares = build_linear_shares(market_levels[price_columns], weights, V0)

    results, realised_pnl = compute_vol_adjusted_hs_var(
        market_levels=market_levels,
        shock_frame=shock_frame,
        straddle_state=straddle_state,
        price_columns=price_columns,
        linear_shares=linear_shares,
        window=WINDOW,
        alpha=ALPHA,
        vol_window=VOL_WINDOW,
    )

    backtest_result = run_backtest(
        pnl=realised_pnl,
        var=results["VaR_HS_VolAdj"],
        confidence=ALPHA,
        method_name="HistSim_VolAdj",
    )

    print("\n" + str(backtest_result))

    plot_all(backtest_result, pnl=realised_pnl, var=results["VaR_HS_VolAdj"], save=True)
    save_vol_adjusted_results(results, realised_pnl, backtest_result)

    print("\n" + "=" * 70)
    print("Volatility-Adjusted Historical Simulation workflow complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()