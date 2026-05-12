"""
Backtest-basis consistency checks for the Monte Carlo VaR pipeline.

The MC backtest is built so that its `actual_loss` column reflects the
canonical realised P&L series (`-pnl_total` from
`data/processed/total_portfolio_pnl.csv`). Before the May 2026 basis fix,
`mc_gaussian.py` scaled the simulated linear P&L by the initial portfolio
notional V0 while the realised side used fixed share counts at current
prices. The result was a MC mean VaR roughly half of HistSim and
~320 exceptions / 7.5 % at 99 % over the 4,019-day window.

These tests guard the two failure modes that would have caught that bug:

  1. The MC backtest's realised-loss column must agree with the canonical
     `-pnl_total` series on the overlapping dates.
  2. The MC mean VaR must be on the same order of magnitude as the HistSim
     mean VaR. The old V0 basis produced a ~0.5x ratio; we require >= 0.6
     to leave headroom for genuine method differences.

The tests skip if the productive output CSVs are absent (e.g. fresh clone
that has not run the MC pipeline yet) so they do not block unrelated PRs.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
MC_BACKTEST = REPO_ROOT / "outputs" / "tables" / "backtest_mc.csv"
HS_BACKTEST = REPO_ROOT / "outputs" / "tables" / "backtest_historical_sim.csv"
TOTAL_PNL = REPO_ROOT / "data" / "processed" / "total_portfolio_pnl.csv"


def _read_indexed(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, parse_dates=True)


def _require(*paths: Path) -> None:
    missing = [str(p.relative_to(REPO_ROOT)) for p in paths if not p.exists()]
    if missing:
        pytest.skip(f"required output(s) not present: {missing}")


def test_mc_backtest_realised_loss_matches_canonical_pnl_series():
    """The MC backtest's `actual_loss` must equal `-pnl_total` on overlapping dates.

    Tolerance is $1, well below any realistic numerical-rounding floor in
    pandas CSV round-trips but tight enough to flag a basis swap.
    """
    _require(MC_BACKTEST, TOTAL_PNL)

    mc = _read_indexed(MC_BACKTEST)
    total_pnl = _read_indexed(TOTAL_PNL)["pnl_total"]

    common = mc.index.intersection(total_pnl.index)
    assert len(common) > 0, "MC backtest and total_portfolio_pnl share no dates"

    canonical_loss = -total_pnl.loc[common]
    mc_loss = mc.loc[common, "actual_loss"]

    diff = (mc_loss - canonical_loss).abs()
    assert diff.max() < 1.0, (
        f"MC backtest actual_loss diverges from canonical -pnl_total: "
        f"max |diff| = ${diff.max():,.2f}. The MC backtest is reading the "
        f"realised series from a different source than total_portfolio_pnl.csv."
    )


def test_mc_var_scale_is_plausible_vs_histsim():
    """Mean MC VaR must be on the same order as mean HistSim VaR.

    The pre-fix V0-basis bug produced mean MC VaR / mean HS VaR ~= 0.49
    on the same realised series. We require the ratio to be >= 0.6 to
    catch that failure mode with margin while still allowing for genuine
    method differences (Gaussian MC is narrower-tailed than HistSim, so
    some gap is expected — just not a 2x gap).
    """
    _require(MC_BACKTEST, HS_BACKTEST)

    mc = _read_indexed(MC_BACKTEST)
    hs = _read_indexed(HS_BACKTEST)

    common = mc.index.intersection(hs.index)
    assert len(common) > 250, "MC and HistSim backtests share too few dates to compare"

    mean_mc = mc.loc[common, "VaR_MC"].mean()
    mean_hs = hs.loc[common, "VaR_HistSim"].mean()
    ratio = mean_mc / mean_hs

    assert ratio >= 0.6, (
        f"Mean MC VaR (${mean_mc:,.0f}) is implausibly small relative to mean "
        f"HistSim VaR (${mean_hs:,.0f}) — ratio {ratio:.2f}. This is the same "
        f"scale failure the V0-vs-fixed-shares basis bug produced; check that "
        f"the MC linear leg uses fixed share counts at current prices."
    )


def test_mc_var_scale_is_plausible_vs_realised_max_loss():
    """Mean MC VaR must not be vanishingly small vs the realised loss series.

    Sanity floor: the max realised loss should not exceed 5x the mean MC
    VaR. The old V0 bug produced max realised loss / mean MC VaR ~= 5.2
    on the 4,269-day window; the fixed code is comfortably under 4. We
    use 5 as a coarse upper bound so the test catches a re-introduction
    of the V0 basis without being overfit to the current numbers.
    """
    _require(MC_BACKTEST)

    mc = _read_indexed(MC_BACKTEST)
    mean_var = mc["VaR_MC"].mean()
    max_loss = mc["actual_loss"].max()

    assert mean_var > 0, "MC VaR mean is non-positive"
    assert max_loss / mean_var < 5.0, (
        f"Realised max loss ${max_loss:,.0f} is {max_loss / mean_var:.2f}x the "
        f"mean MC VaR ${mean_var:,.0f}. The MC VaR series appears too small "
        f"for the realised loss scale (basis mismatch suspected)."
    )
