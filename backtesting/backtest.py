"""
backtesting/backtest.py
=======================
Centralised VaR backtesting utilities for the Market Risk Modelling project.

Implements:
  - Kupiec (1995) Proportion of Failures (POF) test  — unconditional coverage
  - Christoffersen (1998) Independence test           — clustering of exceptions
  - Christoffersen (1998) Conditional Coverage test   — UC + Independence jointly

Usage
-----
Import and call run_backtest() with any VaR series and the corresponding P&L
series.  The function returns a BacktestResult dataclass that every team member
can print or inspect uniformly.

    from backtesting.backtest import run_backtest

    result = run_backtest(
        pnl=pnl_series,       # pd.Series of daily P&L (positive = gain)
        var=var_series,        # pd.Series of daily VaR (positive number, i.e. loss threshold)
        confidence=0.99,       # VaR confidence level
        method_name="HistSim"  # label for display / report
    )
    print(result)

Theoretical references
----------------------
All test statistics and critical values follow Dr. Sebastian Irle's lecture
notes.

  - Exception indicator definition: Irle Lecture Notes, Ch. 8, p. 184
      I_t = 1  if  PnL_t < -VaR_t  (a breach / "overshoot")
      I_t = 0  otherwise

  - Kupiec LR_UC: Irle Lecture Notes, Ch. 8, p. 184
      Under H0: N ~ B(T, p) where p = 1 - confidence level
      LR_UC = -2 * ln[ p^N (1-p)^(T-N) / p_hat^N (1-p_hat)^(T-N) ]
      LR_UC ~ chi2(1) under H0
      Reference: P. Kupiec (1995), "Techniques for verifying the accuracy
      of risk measurement models", Journal of Derivatives 3(2), 73-84.

  - Christoffersen LR_IND and LR_CC: derived from transition matrix of I_t
      Reference: P.F. Christoffersen (1998), "Evaluating interval forecasts",
      International Economic Review 39(4), 841-862.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    """Stores all outputs from a single VaR backtest run."""

    method_name: str
    confidence: float
    T: int                   # total observations in backtest window
    N: int                   # observed number of exceptions
    expected_N: float        # expected exceptions under H0 = T * (1 - confidence)
    exception_rate: float    # empirical exception rate = N / T

    # Kupiec POF test (unconditional coverage)
    lr_uc: float
    pvalue_uc: float
    reject_uc: bool          # reject H0 at 5% significance?

    # Christoffersen independence test
    lr_ind: float
    pvalue_ind: float
    reject_ind: bool

    # Christoffersen conditional coverage (joint) test
    lr_cc: float
    pvalue_cc: float
    reject_cc: bool

    # Transition counts used for the independence test
    n00: int = 0
    n01: int = 0
    n10: int = 0
    n11: int = 0

    exceptions_index: Optional[pd.Index] = field(default=None, repr=False)

    def __str__(self) -> str:
        p = 1.0 - self.confidence
        sep = "─" * 58
        lines = [
            sep,
            f"  Backtest Results  ·  {self.method_name}",
            sep,
            f"  Confidence level   : {self.confidence:.0%}  (p = {p:.4f})",
            f"  Observations (T)   : {self.T}",
            f"  Expected exceptions: {self.expected_N:.1f}",
            f"  Observed exceptions: {self.N}  "
            f"(exception rate = {self.exception_rate:.4f})",
            "",
            "  ── Kupiec POF (unconditional coverage) ──────────────",
            f"  LR_UC   = {self.lr_uc:.4f}  |  p-value = {self.pvalue_uc:.4f}  "
            f"|  Reject H0 @ 5%: {self.reject_uc}",
            "",
            "  ── Christoffersen Independence ───────────────────────",
            f"  Transitions  00={self.n00}  01={self.n01}  "
            f"10={self.n10}  11={self.n11}",
            f"  LR_IND  = {self.lr_ind:.4f}  |  p-value = {self.pvalue_ind:.4f}  "
            f"|  Reject H0 @ 5%: {self.reject_ind}",
            "",
            "  ── Christoffersen Conditional Coverage (CC) ──────────",
            f"  LR_CC   = {self.lr_cc:.4f}  |  p-value = {self.pvalue_cc:.4f}  "
            f"|  Reject H0 @ 5%: {self.reject_cc}",
            sep,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core test functions
# ---------------------------------------------------------------------------

def compute_exceptions(
    pnl: pd.Series,
    var: pd.Series,
) -> pd.Series:
    """
    Build the binary exception indicator series I_t.

    Parameters
    ----------
    pnl : pd.Series
        Daily portfolio P&L (positive = gain, negative = loss).
    var : pd.Series
        Daily VaR estimate expressed as a *positive* loss threshold.
        A breach occurs when PnL < -VaR, i.e. the realised loss exceeds
        the forecast VaR.

    Returns
    -------
    pd.Series of int (0 or 1), aligned on the intersection of both indices.

    Notes
    -----
    Definition follows Irle Lecture Notes, Ch. 8 "Backtesting", p. 184:
        I_t = 1  if  PnL_t < -VaR_t  (VaR breach)
        I_t = 0  otherwise
    """
    # Align on shared dates
    common = pnl.index.intersection(var.index)
    if len(common) == 0:
        raise ValueError(
            "pnl and var share no common index entries. "
            "Check that both series cover the same date range."
        )
    pnl_aligned = pnl.loc[common]
    var_aligned  = var.loc[common]

    exceptions = (pnl_aligned < -var_aligned).astype(int)
    exceptions.name = "exception"
    return exceptions


def kupiec_test(
    exceptions: pd.Series,
    confidence: float,
) -> tuple[float, float]:
    """
    Kupiec (1995) Proportion of Failures (POF) test.

    Tests H0: the true exception probability equals p = 1 - confidence.
    The test statistic LR_UC follows a chi-squared(1) distribution under H0.

    Formula (Irle Lecture Notes, Ch. 8, p. 184):
        p_hat = N / T
        LR_UC = -2 * ln[ p^N * (1-p)^(T-N) / p_hat^N * (1-p_hat)^(T-N) ]
              ~ chi2(1) under H0

    Parameters
    ----------
    exceptions : pd.Series of {0, 1}
        Exception indicator series from compute_exceptions().
    confidence : float
        VaR confidence level (e.g., 0.99 for 99% VaR).

    Returns
    -------
    lr_uc : float   — likelihood ratio statistic
    p_value : float — p-value under chi2(1)
    """
    p = 1.0 - confidence          # theoretical exception probability
    T = len(exceptions)
    N = int(exceptions.sum())
    p_hat = N / T                 # empirical exception probability

    # Edge cases: p_hat at 0 or 1 means the log-likelihood is degenerate.
    # We clamp to avoid log(0); the resulting LR will still be very large.
    eps = 1e-10
    p_hat_clamped = np.clip(p_hat, eps, 1.0 - eps)

    log_L0 = N * np.log(p) + (T - N) * np.log(1.0 - p)
    log_L1 = N * np.log(p_hat_clamped) + (T - N) * np.log(1.0 - p_hat_clamped)

    lr_uc = -2.0 * (log_L0 - log_L1)
    lr_uc = max(lr_uc, 0.0)      # numerical guard; should already be >= 0
    p_value = 1.0 - stats.chi2.cdf(lr_uc, df=1)

    return lr_uc, p_value


def christoffersen_test(
    exceptions: pd.Series,
) -> tuple[float, float, int, int, int, int]:
    """
    Christoffersen (1998) Independence test.

    Tests H0: VaR breaches are independently distributed over time.
    Rejects when exceptions cluster (e.g., during crises), indicating
    that the model fails to adapt to changing volatility regimes.

    The test builds a first-order Markov transition matrix over the
    exception indicator series I_t and compares a constrained (i.i.d.)
    likelihood to the unconstrained one.

    LR_IND ~ chi2(1) under H0.

    Reference: Christoffersen (1998), IER 39(4), 841–862.

    Parameters
    ----------
    exceptions : pd.Series of {0, 1}

    Returns
    -------
    lr_ind : float
    p_value : float
    n00, n01, n10, n11 : int  — transition counts
    """
    I = exceptions.values.astype(int)

    # Transition counts
    # n_ij = number of times state i is followed by state j
    n00 = int(((I[:-1] == 0) & (I[1:] == 0)).sum())
    n01 = int(((I[:-1] == 0) & (I[1:] == 1)).sum())
    n10 = int(((I[:-1] == 1) & (I[1:] == 0)).sum())
    n11 = int(((I[:-1] == 1) & (I[1:] == 1)).sum())

    # Transition probabilities
    eps = 1e-10

    denom_0 = n00 + n01
    pi_01 = n01 / denom_0 if denom_0 > 0 else eps   # P(exception | no exception yesterday)

    denom_1 = n10 + n11
    pi_11 = n11 / denom_1 if denom_1 > 0 else eps   # P(exception | exception yesterday)

    # Unconditional probability under unconstrained model
    N_total = n01 + n11
    T_m1    = len(I) - 1             # T-1 transitions
    pi      = N_total / T_m1 if T_m1 > 0 else eps

    # Clamp to avoid log(0)
    pi_01 = np.clip(pi_01, eps, 1.0 - eps)
    pi_11 = np.clip(pi_11, eps, 1.0 - eps)
    pi    = np.clip(pi,    eps, 1.0 - eps)

    # Log-likelihood under H0 (i.i.d., single parameter pi)
    log_L0 = (
        (n00 + n10) * np.log(1.0 - pi)
        + (n01 + n11) * np.log(pi)
    )

    # Log-likelihood under H1 (first-order Markov, two parameters)
    log_L1 = (
        n00 * np.log(1.0 - pi_01)
        + n01 * np.log(pi_01)
        + n10 * np.log(1.0 - pi_11)
        + n11 * np.log(pi_11)
    )

    lr_ind = -2.0 * (log_L0 - log_L1)
    lr_ind = max(lr_ind, 0.0)
    p_value = 1.0 - stats.chi2.cdf(lr_ind, df=1)

    return lr_ind, p_value, n00, n01, n10, n11


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_backtest(
    pnl: pd.Series,
    var: pd.Series,
    confidence: float = 0.99,
    method_name: str = "Unknown",
    significance: float = 0.05,
) -> BacktestResult:
    """
    Run Kupiec + Christoffersen backtests for a given VaR series.

    This is the single function all team members should call after computing
    their VaR time series. The function aligns the P&L and VaR series on
    their shared index, computes the exception indicator, and returns a
    BacktestResult object that can be printed or passed to the report writer.

    Parameters
    ----------
    pnl : pd.Series
        Daily realised portfolio P&L indexed by date.
        *Convention*: positive values = gains, negative values = losses.
    var : pd.Series
        Daily VaR estimate indexed by date.
        *Convention*: expressed as a positive number (the loss threshold).
        A breach occurs when PnL_t < -VaR_t.
    confidence : float, default 0.99
        VaR confidence level (0.99 = 99% VaR).
    method_name : str
        Short label identifying the VaR model (e.g., "DeltaNormal",
        "HistSim", "MonteCarlo", "GARCH", "EVT", "GARCH+EVT").
    significance : float, default 0.05
        Significance level used to determine the reject_* flags.

    Returns
    -------
    BacktestResult

    Examples
    --------
    >>> result = run_backtest(pnl, var_histsim, confidence=0.99,
    ...                       method_name="HistSim")
    >>> print(result)
    >>> result.reject_uc   # True / False
    >>> result.lr_cc        # the CC test statistic for the report
    """
    # 1. Build exception indicator (Irle Lecture Notes, Ch. 8, p. 184)
    exceptions = compute_exceptions(pnl, var)

    T   = len(exceptions)
    N   = int(exceptions.sum())
    p   = 1.0 - confidence

    if N == 0:
        warnings.warn(
            f"[{method_name}] Zero exceptions recorded over {T} observations. "
            "Kupiec and Christoffersen statistics may be degenerate."
        )

    # 2. Kupiec POF test — unconditional coverage
    lr_uc, pv_uc = kupiec_test(exceptions, confidence)

    # 3. Christoffersen independence test
    lr_ind, pv_ind, n00, n01, n10, n11 = christoffersen_test(exceptions)

    # 4. Christoffersen conditional coverage (additive by construction)
    #    LR_CC = LR_UC + LR_IND ~ chi2(2) under H0
    lr_cc = lr_uc + lr_ind
    pv_cc = 1.0 - stats.chi2.cdf(lr_cc, df=2)

    return BacktestResult(
        method_name=method_name,
        confidence=confidence,
        T=T,
        N=N,
        expected_N=round(T * p, 1),
        exception_rate=N / T,
        lr_uc=lr_uc,
        pvalue_uc=pv_uc,
        reject_uc=pv_uc < significance,
        lr_ind=lr_ind,
        pvalue_ind=pv_ind,
        reject_ind=pv_ind < significance,
        lr_cc=lr_cc,
        pvalue_cc=pv_cc,
        reject_cc=pv_cc < significance,
        n00=n00,
        n01=n01,
        n10=n10,
        n11=n11,
        exceptions_index=exceptions[exceptions == 1].index,
    )


# ---------------------------------------------------------------------------
# Convenience: compare multiple methods at once
# ---------------------------------------------------------------------------

def compare_methods(results: list[BacktestResult]) -> pd.DataFrame:
    """
    Build a summary DataFrame comparing backtest results across VaR methods.

    Parameters
    ----------
    results : list of BacktestResult
        Output of run_backtest() for each method.

    Returns
    -------
    pd.DataFrame with one row per method.

    Example
    -------
    >>> summary = compare_methods([res_dn, res_hs, res_mc, res_garch])
    >>> print(summary.to_string())
    """
    rows = []
    for r in results:
        rows.append({
            "Method":        r.method_name,
            "T":             r.T,
            "Expected N":    r.expected_N,
            "Observed N":    r.N,
            "Exc. Rate":     f"{r.exception_rate:.4f}",
            "LR_UC":         f"{r.lr_uc:.3f}",
            "p(UC)":         f"{r.pvalue_uc:.4f}",
            "Reject UC":     r.reject_uc,
            "LR_IND":        f"{r.lr_ind:.3f}",
            "p(IND)":        f"{r.pvalue_ind:.4f}",
            "Reject IND":    r.reject_ind,
            "LR_CC":         f"{r.lr_cc:.3f}",
            "p(CC)":         f"{r.pvalue_cc:.4f}",
            "Reject CC":     r.reject_cc,
        })
    return pd.DataFrame(rows).set_index("Method")