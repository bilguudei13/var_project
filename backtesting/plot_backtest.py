"""
backtesting/plot_backtest.py
============================
Standardised backtesting visualisations for the Market Risk Modelling project.

Produces four figure types that share a single style block at the top of this
file.  Filenames are built deterministically from BacktestResult.method_name
so team members never choose filenames manually.

  Fig 11 — Exception timeline          one file per method
  Fig 12 — Exception rate comparison   one file, all methods combined
  Fig 13 — LR statistics dot plot      one file, all methods combined
  Fig 14 — Transition matrix heatmap   one file per method

Usage
-----
Single-method figures (each team member calls after run_backtest):

    from backtesting.plot_backtest import plot_exception_timeline, plot_transition_matrix
    plot_exception_timeline(result, pnl=pnl_series, var=var_series)
    plot_transition_matrix(result)

Comparison figures (called once, after all methods are collected):

    from backtesting.plot_backtest import plot_exception_rate_comparison, plot_lr_statistics
    plot_exception_rate_comparison([res_dn, res_hs, res_mc, res_garch, res_evt])
    plot_lr_statistics([res_dn, res_hs, res_mc, res_garch, res_evt])

Or generate both single-method figures at once:

    from backtesting.plot_backtest import plot_all
    plot_all(result, pnl=pnl_series, var=var_series)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats

from .backtest import BacktestResult

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = _REPO_ROOT / "outputs" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Style  — edit this single block to retheme all backtesting figures
# ---------------------------------------------------------------------------

STYLE: dict = {
    "figure.facecolor":         "white",
    "figure.dpi":               150,
    "axes.facecolor":           "white",
    "axes.edgecolor":           "#cccccc",
    "axes.linewidth":           0.8,
    "axes.grid":                True,
    "axes.spines.top":          False,
    "axes.spines.right":        False,
    "axes.titlelocation":       "left",
    "axes.titlesize":           11,
    "axes.titleweight":         "bold",
    "axes.titlepad":            10,
    "axes.labelsize":           9,
    "axes.labelcolor":          "#333333",
    "grid.color":               "#ebebeb",
    "grid.linewidth":           0.6,
    "grid.linestyle":           "--",
    "xtick.labelsize":          8,
    "ytick.labelsize":          8,
    "xtick.color":              "#555555",
    "ytick.color":              "#555555",
    "xtick.major.pad":          4,
    "font.family":              "sans-serif",
    "font.size":                9,
    "legend.framealpha":        0.95,
    "legend.edgecolor":         "#cccccc",
    "legend.fontsize":          8,
    "legend.handlelength":      1.5,
}

# Colour tokens
C = {
    "pnl":          "#2E86AB",
    "var_fill":     "#FDE8CC",
    "var_line":     "#E07B39",
    "exception":    "#C0392B",
    "pass":         "#27AE60",
    "fail":         "#C0392B",
    "neutral":      "#95A5A6",
    "crit":         "#8E44AD",
    "heatmap_low":  "#F7FBFF",
    "heatmap_high": "#2171B5",
}

_CV1 = stats.chi2.ppf(0.95, df=1)
_CV2 = stats.chi2.ppf(0.95, df=2)

_WIDE  = (13, 4.5)
_TALL  = (13, 5.5)
_SMALL = (6,  5)

CRISIS_PERIODS = [
    ("2008-09-15", "2009-06-30", "GFC"),
    ("2011-07-01", "2012-01-31", "Euro Debt"),
    ("2020-02-20", "2020-05-31", "COVID"),
    ("2022-01-01", "2022-12-31", "Inflation Shock"),
]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _apply_style() -> None:
    plt.rcParams.update(STYLE)


def _fig_path(number: int, description: str, method_name: Optional[str] = None) -> Path:
    slug = ""
    if method_name:
        slug = "_" + method_name.strip().replace(" ", "_").replace("+", "plus")
    return FIGURE_DIR / f"{number:02d}_{description}{slug}.png"


def _pass_fail_color(rejected: bool) -> str:
    return C["fail"] if rejected else C["pass"]


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=STYLE["figure.dpi"], bbox_inches="tight")
    print(f"  Saved -> {path.relative_to(_REPO_ROOT)}")


def _subtitle(method_name: str, confidence: float, T: int, N: int, expected: float) -> str:
    return (
        f"{method_name}  |  {confidence:.0%} VaR  |  "
        f"T = {T:,}    Expected N = {expected:.0f}    Observed N = {N}"
    )


def _add_crisis_annotations(ax: plt.Axes, index: pd.Index) -> None:
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


# ---------------------------------------------------------------------------
# Figure 11 — Exception timeline
# ---------------------------------------------------------------------------

def plot_exception_timeline(
    result: BacktestResult,
    pnl: pd.Series,
    var: pd.Series,
    save: bool = True,
) -> tuple:
    """
    Daily P&L line with -VaR boundary and exception markers.

    Parameters
    ----------
    result : BacktestResult  from run_backtest()
    pnl    : pd.Series       daily P&L (positive = gain)
    var    : pd.Series       daily VaR (positive loss threshold)
    save   : bool
    """
    _apply_style()

    common = pnl.index.intersection(var.index)
    pnl_a  = pnl.loc[common]
    var_a  = var.loc[common]
    exc_idx = pnl_a.index[pnl_a < -var_a]

    fig, ax = plt.subplots(figsize=_WIDE)

    ax.plot(var_a.index, -var_a,
            color=C["var_line"], linewidth=1.2, linestyle="--",
            label=f"-VaR ({result.confidence:.0%})", zorder=2)

    ax.plot(pnl_a.index, pnl_a,
            color=C["pnl"], linewidth=0.9, alpha=0.9,
            label="Daily P&L", zorder=3)

    ax.scatter(exc_idx, pnl_a.loc[exc_idx],
               color=C["exception"], s=22, zorder=5,
               label=f"Exceptions  N = {result.N}")

    ax.axhline(0, color="#bbbbbb", linewidth=0.6, zorder=0)

    ax.set_title(_subtitle(
        result.method_name, result.confidence,
        result.T, result.N, result.expected_N
    ))
    ax.set_xlabel("Date")
    ax.set_ylabel("P&L (USD)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:,.0f}")
    )
    _add_crisis_annotations(ax, common)
    ax.legend(loc="upper left", ncol=3)
    fig.tight_layout()

    if save:
        _save(fig, _fig_path(11, "exceptions_timeline", result.method_name))

    return fig, ax


# ---------------------------------------------------------------------------
# Figure 12 — Exception rate comparison (all methods)
# ---------------------------------------------------------------------------

def plot_exception_rate_comparison(
    results: list,
    save: bool = True,
) -> tuple:
    """
    Horizontal bar chart comparing empirical exception rates across all methods.
    Green = Kupiec not rejected.  Red = rejected.

    Parameters
    ----------
    results : list of BacktestResult
    save    : bool
    """
    _apply_style()

    methods = [r.method_name for r in results]
    rates   = [r.exception_rate for r in results]
    colors  = [_pass_fail_color(r.reject_uc) for r in results]
    p_theor = 1.0 - results[0].confidence

    y = np.arange(len(methods))

    fig, ax = plt.subplots(figsize=(9, max(4, 0.7 * len(methods) + 2)))

    bars = ax.barh(y, rates, color=colors, alpha=0.82, height=0.55, zorder=2)

    ax.axvline(p_theor, color=C["crit"], linewidth=1.4, linestyle="--",
               label=f"Theoretical rate  {p_theor:.1%}", zorder=3)

    for bar, r in zip(bars, results):
        ax.text(
            bar.get_width() + max(rates) * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{r.N} / {r.T:,}",
            va="center", ha="left", fontsize=8, color="#444444"
        )

    ax.set_yticks(y)
    ax.set_yticklabels(methods, fontsize=9)
    ax.set_xlabel("Exception rate")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
    ax.set_title("Exception Rate by Method")
    ax.set_xlim(0, max(rates) * 1.35)

    pass_patch = mpatches.Patch(color=C["pass"], alpha=0.82, label="Kupiec: not rejected")
    fail_patch = mpatches.Patch(color=C["fail"], alpha=0.82, label="Kupiec: rejected")
    crit_line  = plt.Line2D([0], [0], color=C["crit"], linestyle="--",
                             linewidth=1.4, label=f"Theoretical  {p_theor:.1%}")
    ax.legend(handles=[pass_patch, fail_patch, crit_line], loc="lower right")

    ax.invert_yaxis()
    fig.tight_layout()

    if save:
        _save(fig, _fig_path(12, "exception_rate_comparison"))

    return fig, ax


# ---------------------------------------------------------------------------
# Figure 13 — LR statistics dot plot (all methods)
# ---------------------------------------------------------------------------

def plot_lr_statistics(
    results: list,
    save: bool = True,
) -> tuple:
    """
    Three-panel dot plot: LR_UC, LR_IND, LR_CC for every method with
    chi-squared critical value lines.  Green = not rejected, red = rejected.

    Parameters
    ----------
    results : list of BacktestResult
    save    : bool
    """
    _apply_style()

    methods = [r.method_name for r in results]
    y       = np.arange(len(methods))

    panels = [
        ([r.lr_uc  for r in results], [r.reject_uc  for r in results],
         _CV1, "LR$_{UC}$",  "Kupiec  (UC)",           f"chi2(1) cv = {_CV1:.2f}"),
        ([r.lr_ind for r in results], [r.reject_ind for r in results],
         _CV1, "LR$_{IND}$", "Christoffersen  (IND)",  f"chi2(1) cv = {_CV1:.2f}"),
        ([r.lr_cc  for r in results], [r.reject_cc  for r in results],
         _CV2, "LR$_{CC}$",  "Christoffersen  (CC)",   f"chi2(2) cv = {_CV2:.2f}"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=_TALL, sharey=True)

    for ax, (values, rejected, cv, xlabel, title, cv_label) in zip(axes, panels):
        dot_colors = [_pass_fail_color(r) for r in rejected]

        ax.axvline(0, color="#cccccc", linewidth=0.6, zorder=0)
        ax.axvline(cv, color=C["crit"], linewidth=1.3, linestyle="--",
                   label=cv_label, zorder=1)

        ax.scatter(values, y, color=dot_colors, s=65, zorder=3,
                   edgecolors="white", linewidths=0.5)

        x_offset = max(values) * 0.04 if max(values) > 0 else 0.1
        for val, yi, rej in zip(values, y, rejected):
            ax.text(val + x_offset, yi,
                    f"{val:.2f}", va="center", ha="left",
                    fontsize=7.5, color=_pass_fail_color(rej))

        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.set_xlim(left=-0.5)
        ax.legend(fontsize=7.5, loc="lower right")
        ax.grid(axis="x")
        ax.grid(axis="y", alpha=0.35)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(methods, fontsize=9)
    axes[0].invert_yaxis()

    fig.suptitle(
        "Likelihood Ratio Test Statistics  |  5% Significance",
        fontsize=11, fontweight="bold", x=0.02, ha="left", y=1.01
    )
    fig.tight_layout()

    if save:
        _save(fig, _fig_path(13, "lr_statistics"))

    return fig, axes


# ---------------------------------------------------------------------------
# Figure 14 — Transition matrix heatmap (per method)
# ---------------------------------------------------------------------------

def plot_transition_matrix(
    result: BacktestResult,
    save: bool = True,
) -> tuple:
    """
    2x2 heatmap of first-order Markov transition probabilities used in the
    Christoffersen independence test.

    Rows = state at t  (0 = no breach, 1 = breach)
    Cols = state at t+1

    Parameters
    ----------
    result : BacktestResult
    save   : bool
    """
    _apply_style()

    counts = np.array(
        [[result.n00, result.n01],
         [result.n10, result.n11]],
        dtype=float
    )
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    probs = counts / row_sums

    fig, ax = plt.subplots(figsize=_SMALL)

    im = ax.imshow(probs, cmap="Blues", vmin=0, vmax=1, aspect="auto")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Transition probability", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    for i in range(2):
        for j in range(2):
            text_color = "white" if probs[i, j] > 0.6 else "#222222"
            ax.text(j, i,
                    f"{probs[i, j]:.3f}\n(n = {int(counts[i, j]):,})",
                    ha="center", va="center",
                    fontsize=9, color=text_color, fontweight="bold")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No breach (t+1)", "Breach (t+1)"], fontsize=8)
    ax.set_yticklabels(["No breach (t)", "Breach (t)"], fontsize=8)
    ax.set_xlabel("Next day")
    ax.set_ylabel("Current day")
    ax.set_title(
        f"Transition Matrix  |  {result.method_name}\n"
        f"LR_IND = {result.lr_ind:.3f}    "
        f"p = {result.pvalue_ind:.4f}    "
        f"Reject H0: {result.reject_ind}",
        fontsize=9
    )

    if result.reject_ind:
        for spine in ax.spines.values():
            spine.set_edgecolor(C["fail"])
            spine.set_linewidth(1.8)
            spine.set_visible(True)

    fig.tight_layout()

    if save:
        _save(fig, _fig_path(14, "transition_matrix", result.method_name))

    return fig, ax


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def plot_all(
    result: BacktestResult,
    pnl: pd.Series,
    var: pd.Series,
    save: bool = True,
) -> None:
    """
    Generate Figures 11 and 14 for a single method in one call.
    Run plot_exception_rate_comparison() and plot_lr_statistics() separately
    once all methods have been backtested and collected into a list.
    """
    plot_exception_timeline(result, pnl, var, save=save)
    plot_transition_matrix(result, save=save)
