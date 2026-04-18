"""
historical_sim_analysis.py
==========================
Baseline-only diagnostics for the Historical Simulation workstream.

This module deliberately treats the full-repricing Historical Simulation model
as the single official HistSim specification and focuses on diagnostics rather
than additional HS variants.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src" / "data"):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from backtesting.backtest import run_backtest
from historical_sim import (
    ALPHA,
    COMPONENT_NAMES,
    LINEAR_ASSETS,
    WINDOW,
    aligned_weights,
    build_linear_shares,
    compute_historical_sim_var,
    empirical_var_es,
    load_market_data,
    realised_next_day_component_pnl,
    scenario_component_pnl_distribution,
)
from config import V0


OUTPUT_TABLES = REPO_ROOT / "outputs" / "tables"
OUTPUT_FIGS = REPO_ROOT / "outputs" / "figures"
REPORT_DIR = REPO_ROOT / "report"
BASELINE_VAR_PATH = REPO_ROOT / "data" / "processed" / "var_historical_sim.csv"

SUBPERIODS = [
    ("Full Sample", None, None),
    ("GFC", "2008-09-15", "2009-06-30"),
    ("Euro Debt / Post-Crisis", "2011-07-01", "2012-01-31"),
    ("COVID", "2020-02-20", "2020-05-31"),
    ("Inflation / Rates Shock", "2022-01-01", "2022-12-31"),
]
WINDOW_GRID = [250, 500, 750]
EXTREME_DAY_COUNT = 15
TAIL_SUMMARY_COUNT = 25
COMPONENT_LABELS = {
    "linear": "Linear Book",
    "irs": "IRS",
    "straddle": "Straddle",
}


def format_pvalue(value: float) -> str:
    """
    Format p-values for report text without hiding very small values as 0.0000.
    """
    if pd.isna(value):
        return "nan"
    if value < 1e-4:
        return "< 0.0001"
    return f"{value:.4f}"


def compute_exception_severity(
    realised_loss: pd.Series,
    var_series: pd.Series,
    exceptions: pd.Series,
) -> float:
    """
    Average exceedance size conditional on an exception.
    """
    mask = exceptions.astype(bool)
    if not mask.any():
        return float("nan")
    exceedance = realised_loss.loc[mask] - var_series.loc[mask]
    return float(exceedance.mean())


def filter_subperiod(results: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    """
    Slice a results DataFrame by date index.
    """
    if start is None or end is None:
        return results.copy()
    return results.loc[pd.Timestamp(start):pd.Timestamp(end)].copy()


def summarize_subperiods(
    results: pd.DataFrame,
    method_name: str,
    var_column: str,
    es_column: str,
    confidence: float,
) -> pd.DataFrame:
    """
    Build the standard backtesting summary across key crisis subperiods.
    """
    rows: list[dict[str, float | int | str]] = []

    for label, start, end in SUBPERIODS:
        sub = filter_subperiod(results, start, end)
        if sub.empty:
            continue

        backtest_result = run_backtest(
            pnl=sub["realised_pnl"],
            var=sub[var_column],
            confidence=confidence,
            method_name=method_name,
        )
        exceptions = (sub["realised_pnl"] < -sub[var_column]).astype(int)
        rows.append(
            {
                "method": method_name,
                "subperiod": label,
                "start_date": sub.index.min().date().isoformat(),
                "end_date": sub.index.max().date().isoformat(),
                "observations": backtest_result.T,
                "expected_exceptions": backtest_result.expected_N,
                "observed_exceptions": backtest_result.N,
                "exception_rate": backtest_result.exception_rate,
                "kupiec_pvalue": backtest_result.pvalue_uc,
                "christoffersen_ind_pvalue": backtest_result.pvalue_ind,
                "christoffersen_cc_pvalue": backtest_result.pvalue_cc,
                "exception_severity": compute_exception_severity(
                    sub["realised_loss"],
                    sub[var_column],
                    exceptions,
                ),
                "mean_var": float(sub[var_column].mean()),
                "mean_es": float(sub[es_column].mean()),
            }
        )

    return pd.DataFrame(rows)


def compute_component_hs_table(
    market_levels: pd.DataFrame,
    shock_frame: pd.DataFrame,
    straddle_state: pd.DataFrame,
    price_columns: list[str],
    linear_shares: np.ndarray,
    window: int,
    alpha: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute component-wise HS time series and aggregated tail-usage records.
    """
    component_records: list[dict[str, float | int | str]] = []
    tail_usage_records: list[dict[str, float | int | str]] = []
    common_dates = market_levels.index

    for t in range(window, len(shock_frame)):
        snapshot_date = common_dates[t]
        forecast_date = shock_frame.index[t]
        snapshot = market_levels.loc[snapshot_date]
        next_snapshot = market_levels.loc[forecast_date]
        state = straddle_state.loc[snapshot_date]
        historical_shocks = shock_frame.iloc[t - window:t]

        component_pnl, component_losses, _ = scenario_component_pnl_distribution(
            snapshot=snapshot,
            state=state,
            shock_window=historical_shocks,
            linear_shares=linear_shares,
            price_columns=price_columns,
        )
        realised_component_pnl = realised_next_day_component_pnl(
            snapshot=snapshot,
            next_snapshot=next_snapshot,
            state=state,
            linear_shares=linear_shares,
            price_columns=price_columns,
        )

        scenario_losses = (
            np.asarray(component_losses["linear"], dtype=float)
            + np.asarray(component_losses["irs"], dtype=float)
            + np.asarray(component_losses["straddle"], dtype=float)
        )
        _, _, tail_count = empirical_var_es(scenario_losses, alpha)
        sorted_indices = np.argsort(scenario_losses)
        tail_indices = sorted_indices[-tail_count:]
        tail_ranked = tail_indices[np.argsort(scenario_losses[tail_indices])[::-1]]
        realised_loss = -float(realised_component_pnl["linear"] + realised_component_pnl["irs"] + realised_component_pnl["straddle"])

        for rank, scenario_idx in enumerate(tail_ranked, start=1):
            tail_usage_records.append(
                {
                    "forecast_date": forecast_date.date().isoformat(),
                    "snapshot_date": snapshot_date.date().isoformat(),
                    "historical_shock_date": historical_shocks.index[scenario_idx].date().isoformat(),
                    "tail_rank": rank,
                    "scenario_loss": float(scenario_losses[scenario_idx]),
                    "realised_loss": realised_loss,
                }
            )

        for component_name in COMPONENT_NAMES:
            component_var, component_es, component_tail_count = empirical_var_es(component_losses[component_name], alpha)
            component_var = max(component_var, 0.0)
            component_es = max(component_es, 0.0)
            realised_pnl = float(realised_component_pnl[component_name])
            realised_component_loss = -realised_pnl
            component_records.append(
                {
                    "forecast_date": forecast_date.date().isoformat(),
                    "snapshot_date": snapshot_date.date().isoformat(),
                    "component": component_name,
                    "component_label": COMPONENT_LABELS[component_name],
                    "VaR_component": component_var,
                    "ES_component": component_es,
                    "tail_scenarios": component_tail_count,
                    "realised_pnl": realised_pnl,
                    "realised_loss": realised_component_loss,
                    "exception": int(realised_pnl < -component_var),
                }
            )

    component_df = pd.DataFrame(component_records)
    if not component_df.empty:
        component_df["forecast_date"] = pd.to_datetime(component_df["forecast_date"])
        component_df["snapshot_date"] = pd.to_datetime(component_df["snapshot_date"])
        component_df = component_df.set_index("forecast_date").sort_index()

    tail_usage_df = pd.DataFrame(tail_usage_records)
    return component_df, tail_usage_df


def summarize_component_hs(component_df: pd.DataFrame, confidence: float) -> pd.DataFrame:
    """
    Summarize component-wise HS risk metrics and backtesting statistics.
    """
    rows: list[dict[str, float | int | str]] = []

    for component_name in COMPONENT_NAMES:
        sub = component_df.loc[component_df["component"] == component_name].copy()
        backtest_result = run_backtest(
            pnl=sub["realised_pnl"],
            var=sub["VaR_component"],
            confidence=confidence,
            method_name=COMPONENT_LABELS[component_name],
        )
        exceptions = (sub["realised_pnl"] < -sub["VaR_component"]).astype(int)
        rows.append(
            {
                "component": component_name,
                "component_label": COMPONENT_LABELS[component_name],
                "observations": backtest_result.T,
                "expected_exceptions": backtest_result.expected_N,
                "observed_exceptions": backtest_result.N,
                "exception_rate": backtest_result.exception_rate,
                "kupiec_pvalue": backtest_result.pvalue_uc,
                "christoffersen_ind_pvalue": backtest_result.pvalue_ind,
                "christoffersen_cc_pvalue": backtest_result.pvalue_cc,
                "exception_severity": compute_exception_severity(
                    sub["realised_loss"],
                    sub["VaR_component"],
                    exceptions,
                ),
                "mean_var": float(sub["VaR_component"].mean()),
                "mean_es": float(sub["ES_component"].mean()),
            }
        )

    return pd.DataFrame(rows)


def summarize_tail_scenario_usage(tail_usage_df: pd.DataFrame, top_n: int = TAIL_SUMMARY_COUNT) -> pd.DataFrame:
    """
    Aggregate recurring tail-scenario usage across rolling windows.
    """
    if tail_usage_df.empty:
        return pd.DataFrame(
            columns=[
                "historical_shock_date",
                "times_in_tail",
                "times_as_worst_scenario",
                "mean_tail_rank",
                "max_scenario_loss",
                "mean_scenario_loss",
                "mean_realised_loss_when_used",
            ]
        )

    summary = (
        tail_usage_df.groupby("historical_shock_date")
        .agg(
            times_in_tail=("historical_shock_date", "size"),
            times_as_worst_scenario=("tail_rank", lambda x: int((x == 1).sum())),
            mean_tail_rank=("tail_rank", "mean"),
            max_scenario_loss=("scenario_loss", "max"),
            mean_scenario_loss=("scenario_loss", "mean"),
            mean_realised_loss_when_used=("realised_loss", "mean"),
        )
        .reset_index()
        .sort_values(["times_in_tail", "times_as_worst_scenario", "max_scenario_loss"], ascending=[False, False, False])
        .head(top_n)
    )
    return summary


def build_extreme_day_table(results: pd.DataFrame, component_df: pd.DataFrame, top_n: int = EXTREME_DAY_COUNT) -> pd.DataFrame:
    """
    Create a table of the worst realised-loss days with component contributions.
    """
    component_wide = (
        component_df.reset_index()
        .pivot_table(index="forecast_date", columns="component", values="realised_loss", aggfunc="first")
        .rename(
            columns={
                "linear": "linear_realised_loss",
                "irs": "irs_realised_loss",
                "straddle": "straddle_realised_loss",
            }
        )
    )
    merged = results.join(component_wide, how="left")
    component_columns = ["linear_realised_loss", "irs_realised_loss", "straddle_realised_loss"]

    def largest_driver(row: pd.Series) -> str:
        available = {column: float(row[column]) for column in component_columns}
        winner = max(available, key=available.get)
        return winner.replace("_realised_loss", "")

    extreme_days = merged.sort_values("realised_loss", ascending=False).head(top_n).copy()
    extreme_days["largest_driver"] = extreme_days.apply(largest_driver, axis=1)
    return extreme_days[
        [
            "snapshot_date",
            "current_portfolio_value",
            "VaR_HistSim",
            "ES_HistSim",
            "realised_loss",
            "realised_pnl",
            "exception",
            "linear_realised_loss",
            "irs_realised_loss",
            "straddle_realised_loss",
            "largest_driver",
        ]
    ]


def compute_window_sensitivity(
    market_levels: pd.DataFrame,
    shock_frame: pd.DataFrame,
    straddle_state: pd.DataFrame,
    price_columns: list[str],
    linear_shares: np.ndarray,
    windows: list[int],
    alpha: float,
) -> pd.DataFrame:
    """
    Re-run the same baseline HS logic over alternative rolling windows.
    """
    rows: list[dict[str, float | int]] = []

    for window in windows:
        results, realised_pnl = compute_historical_sim_var(
            market_levels=market_levels,
            shock_frame=shock_frame,
            straddle_state=straddle_state,
            price_columns=price_columns,
            linear_shares=linear_shares,
            window=window,
            alpha=alpha,
        )
        backtest_result = run_backtest(
            pnl=realised_pnl,
            var=results["VaR_HistSim"],
            confidence=alpha,
            method_name=f"HistSim_{window}",
        )
        exceptions = (results["realised_pnl"] < -results["VaR_HistSim"]).astype(int)
        rows.append(
            {
                "window": window,
                "observations": backtest_result.T,
                "expected_exceptions": backtest_result.expected_N,
                "observed_exceptions": backtest_result.N,
                "exception_rate": backtest_result.exception_rate,
                "kupiec_pvalue": backtest_result.pvalue_uc,
                "christoffersen_ind_pvalue": backtest_result.pvalue_ind,
                "christoffersen_cc_pvalue": backtest_result.pvalue_cc,
                "exception_severity": compute_exception_severity(
                    results["realised_loss"],
                    results["VaR_HistSim"],
                    exceptions,
                ),
                "mean_var": float(results["VaR_HistSim"].mean()),
                "mean_es": float(results["ES_HistSim"].mean()),
            }
        )

    return pd.DataFrame(rows).sort_values("window").reset_index(drop=True)


def plot_subperiod_diagnostics(subperiod_df: pd.DataFrame) -> None:
    """
    Plot exception rates by subperiod with the theoretical breach rate line.
    """
    OUTPUT_FIGS.mkdir(parents=True, exist_ok=True)
    ordered = subperiod_df.loc[subperiod_df["subperiod"] != "Full Sample"].copy()

    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(ordered))
    ax.bar(x, ordered["exception_rate"], color="#d35400", alpha=0.85)
    ax.axhline(1.0 - ALPHA, color="#7f8c8d", linestyle="--", linewidth=1.3, label="Theoretical 1% rate")

    for i, row in enumerate(ordered.itertuples(index=False)):
        ax.text(i, row.exception_rate + 0.003, f"{row.observed_exceptions}/{row.observations}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(ordered["subperiod"], rotation=10, ha="right")
    ax.set_ylabel("Exception rate")
    ax.set_title("Historical Simulation Subperiod Diagnostics", loc="left", fontweight="bold")
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(xmax=1, decimals=1))
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.legend(loc="upper right")
    fig.tight_layout()

    path = OUTPUT_FIGS / "23_histsim_subperiod_diagnostics.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure        -> {path.relative_to(REPO_ROOT)}")


def plot_component_drivers(extreme_days: pd.DataFrame) -> None:
    """
    Plot the component realised losses on the worst total-loss days.
    """
    OUTPUT_FIGS.mkdir(parents=True, exist_ok=True)
    plot_days = extreme_days.head(10).copy()
    labels = [pd.Timestamp(idx).strftime("%Y-%m-%d") for idx in plot_days.index]
    x = np.arange(len(plot_days))
    width = 0.26

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width, plot_days["linear_realised_loss"], width=width, color="#2e86ab", label="Linear Book")
    ax.bar(x, plot_days["irs_realised_loss"], width=width, color="#d35400", label="IRS")
    ax.bar(x + width, plot_days["straddle_realised_loss"], width=width, color="#8e5a2b", label="Straddle")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("USD realised loss")
    ax.set_title("Extreme-Day Drivers | Historical Simulation", loc="left", fontweight="bold")
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.legend(loc="upper right")
    fig.tight_layout()

    path = OUTPUT_FIGS / "24_histsim_component_extreme_day_drivers.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure        -> {path.relative_to(REPO_ROOT)}")


def write_interpretation_markdown(
    full_sample: pd.Series,
    subperiod_df: pd.DataFrame,
    window_df: pd.DataFrame,
    component_summary: pd.DataFrame,
    tail_summary: pd.DataFrame,
    extreme_days: pd.DataFrame,
) -> None:
    """
    Write a report-ready HistSim interpretation focused on one baseline model.
    """
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    covid = subperiod_df.loc[subperiod_df["subperiod"] == "COVID"].iloc[0]
    gfc = subperiod_df.loc[subperiod_df["subperiod"] == "GFC"].iloc[0]
    dominant_component = component_summary.sort_values("mean_var", ascending=False).iloc[0]
    most_reused_tail = tail_summary.iloc[0] if not tail_summary.empty else None
    shortest_window = window_df.loc[window_df["window"] == min(window_df["window"])].iloc[0]
    longest_window = window_df.loc[window_df["window"] == max(window_df["window"])].iloc[0]
    baseline_window = window_df.loc[window_df["window"] == WINDOW].iloc[0]
    dominant_realised_driver = None if extreme_days.empty else extreme_days["largest_driver"].value_counts().idxmax()
    dominant_realised_driver_label = (
        COMPONENT_LABELS[dominant_realised_driver]
        if dominant_realised_driver in COMPONENT_LABELS
        else None
    )

    tail_sentence = (
        f"The most persistent historical tail shock in the rolling windows is {most_reused_tail['historical_shock_date']}, "
        f"which appears {int(most_reused_tail['times_in_tail'])} times among the active tail scenarios. "
        f"Under a 500-day window this is the structural maximum, meaning the shock remained in the tail on every day it was eligible."
        if most_reused_tail is not None
        else "Tail-scenario recurrence could not be summarised because no tail diagnostics were available."
    )
    realised_driver_sentence = (
        f"In the extreme realised-loss table, the most frequent largest driver is the {dominant_realised_driver_label}."
        if dominant_realised_driver_label is not None
        else "The extreme-day driver table did not contain enough observations to rank realised-loss drivers."
    )

    content = f"""# Historical Simulation Interpretation

## One baseline model, not a HistSim family

The Historical Simulation contribution in this repository is intentionally presented as **one official baseline model**: a 1-day 99% full-repricing Historical Simulation VaR. This is more scientifically defensible for the project than introducing several small HistSim variants, because the emphasis moves to correctness, backtesting, extreme-loss interpretation, and model limitations.

## Why Kupiec can pass while Christoffersen fails

Over the full sample, the baseline HistSim model records {int(full_sample["observed_exceptions"])} exceptions against {full_sample["expected_exceptions"]:.1f} expected exceptions, with a Kupiec p-value of {format_pvalue(float(full_sample["kupiec_pvalue"]))}. This indicates that the unconditional breach frequency is close to the theoretical 1% target.

At the same time, the Christoffersen independence p-value is {format_pvalue(float(full_sample["christoffersen_ind_pvalue"]))}, so independence is rejected. The central interpretation is that Historical Simulation reuses a fixed rolling historical window and therefore adjusts only gradually when volatility regimes change. The model can therefore match the average breach frequency over a long sample while still producing clustered exceptions in fast-moving stress regimes.

## Why the 500-day 99% setup is stepwise

With a 500-day rolling window at the 99% level, the VaR estimate is anchored by only 5 active tail scenarios in each window. This makes the baseline model transparent but also discrete and regime-sensitive: a small number of recurring historical shocks can dominate the VaR estimate for long stretches of time. {tail_sentence}

## Crisis windows and stress interpretation

The GFC window is the clearest example of local stress for the baseline model: {int(gfc["observed_exceptions"])} exceptions over {int(gfc["observations"])} observations, with a Kupiec p-value of {format_pvalue(float(gfc["kupiec_pvalue"]))}. The COVID window is also highly informative: {int(covid["observed_exceptions"])} exceptions over {int(covid["observations"])} observations and mean VaR of ${covid["mean_var"]:,.0f}. This shows that short crisis samples can display sharply elevated breach rates and large realised losses even when independence is not rejected within the subperiod itself.

## Which part of the book drives the HS losses

The component-wise HS analysis shows that the largest **standalone average component VaR** comes from the {dominant_component["component_label"]}, with mean component VaR of ${dominant_component["mean_var"]:,.0f}. This should be read as a standalone risk diagnostic rather than a portfolio-VaR decomposition: the component VaRs do not add up to total portfolio VaR because each component can have a different set of worst tail scenarios in the 500-day window. {realised_driver_sentence}

## Window robustness

The window-sensitivity check keeps the same Historical Simulation methodology but varies the historical lookback. The shortest tested window ({int(shortest_window["window"])}) produces mean VaR of ${shortest_window["mean_var"]:,.0f}, while the longest tested window ({int(longest_window["window"])}) produces mean VaR of ${longest_window["mean_var"]:,.0f}. This confirms that the choice of historical window materially affects responsiveness, exception behaviour, and tail severity even when the underlying HS model is unchanged. In this dataset, the baseline 500-day specification is also the only tested window that does **not** reject Kupiec unconditional coverage: the 250-day window has p-value {format_pvalue(float(shortest_window["kupiec_pvalue"]))}, the 500-day window has p-value {format_pvalue(float(baseline_window["kupiec_pvalue"]))}, and the 750-day window has p-value {format_pvalue(float(longest_window["kupiec_pvalue"]))}. This comparison should still be read as a robustness check rather than a perfectly controlled horse race, because the alternative windows are evaluated on different effective forecast samples after their own burn-in periods.
"""

    path = REPORT_DIR / "historical_sim_interpretation.md"
    path.write_text(content, encoding="utf-8")
    print(f"Saved interpretation -> {path.relative_to(REPO_ROOT)}")


def main() -> None:
    """
    Run the baseline-only HistSim diagnostics workflow.
    """
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUTPUT_FIGS.mkdir(parents=True, exist_ok=True)

    baseline_results = pd.read_csv(BASELINE_VAR_PATH, index_col=0, parse_dates=True)
    market_levels, shock_frame, straddle_state = load_market_data()
    price_columns = [column for column in market_levels.columns if column in LINEAR_ASSETS]
    weights = aligned_weights(price_columns)
    linear_shares = build_linear_shares(market_levels[price_columns], weights, V0)

    subperiod_df = summarize_subperiods(
        baseline_results,
        method_name="HistSim",
        var_column="VaR_HistSim",
        es_column="ES_HistSim",
        confidence=ALPHA,
    )
    subperiod_path = OUTPUT_TABLES / "histsim_subperiod_diagnostics.csv"
    subperiod_df.to_csv(subperiod_path, index=False)
    print(f"Saved table         -> {subperiod_path.relative_to(REPO_ROOT)}")

    component_df, tail_usage_df = compute_component_hs_table(
        market_levels=market_levels,
        shock_frame=shock_frame,
        straddle_state=straddle_state,
        price_columns=price_columns,
        linear_shares=linear_shares,
        window=WINDOW,
        alpha=ALPHA,
    )
    component_summary = summarize_component_hs(component_df, confidence=ALPHA)
    component_path = OUTPUT_TABLES / "histsim_component_summary.csv"
    component_summary.to_csv(component_path, index=False)
    print(f"Saved table         -> {component_path.relative_to(REPO_ROOT)}")

    extreme_days = build_extreme_day_table(baseline_results, component_df, top_n=EXTREME_DAY_COUNT)
    extreme_path = OUTPUT_TABLES / "histsim_extreme_days.csv"
    extreme_days.to_csv(extreme_path, index_label="forecast_date")
    print(f"Saved table         -> {extreme_path.relative_to(REPO_ROOT)}")

    tail_summary = summarize_tail_scenario_usage(tail_usage_df, top_n=TAIL_SUMMARY_COUNT)
    tail_path = OUTPUT_TABLES / "histsim_tail_scenario_diagnostics.csv"
    tail_summary.to_csv(tail_path, index=False)
    print(f"Saved table         -> {tail_path.relative_to(REPO_ROOT)}")

    window_df = compute_window_sensitivity(
        market_levels=market_levels,
        shock_frame=shock_frame,
        straddle_state=straddle_state,
        price_columns=price_columns,
        linear_shares=linear_shares,
        windows=WINDOW_GRID,
        alpha=ALPHA,
    )
    window_path = OUTPUT_TABLES / "histsim_window_sensitivity.csv"
    window_df.to_csv(window_path, index=False)
    print(f"Saved table         -> {window_path.relative_to(REPO_ROOT)}")

    plot_subperiod_diagnostics(subperiod_df)
    plot_component_drivers(extreme_days)
    full_sample = subperiod_df.loc[subperiod_df["subperiod"] == "Full Sample"].iloc[0]
    write_interpretation_markdown(
        full_sample,
        subperiod_df,
        window_df,
        component_summary,
        tail_summary,
        extreme_days,
    )


if __name__ == "__main__":
    main()
