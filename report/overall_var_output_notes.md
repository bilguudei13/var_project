# Overall VaR Output Notes

## Common Setup

- All methods target 1-day 99% VaR.
- Backtest sample length in available outputs: 4,269 observations for the
  500-day-window methods (HistSim, EVT, GARCH-EVT, Copula); 4,019 for
  Monte Carlo, which uses a 750-day rolling estimation window.
- Expected exceptions at 99%: 42.7 (4,269-day window) and 40.2 (MC).
- Key question for professor: should all methods be backtested against exactly the same realised P&L series?

## Backtest Overview From Current Outputs

### HistSim

- Observed exceptions: 42
- Exception rate: 0.98%
- Lecture-style exact binomial acceptance range: 30 to 56 exceptions, inclusive.
- Lecture-style rejection rule: reject if observed exceptions are below 30 or above 56.
- Exact binomial p-value: 1.0000, computed in `backtesting/backtest.py` and exported to `backtest_summary_historical_sim.csv`.
- Lecture-style threshold conclusion: 42 is inside the inclusive 30-56 range, so HistSim is not rejected on exception count.
- Kupiec p-value: 0.9152
- Christoffersen IND p-value: < 0.0001
- Conditional coverage p-value: 0.0002
- Interpretation: unconditional coverage is excellent under both the lecture binomial threshold and Kupiec LR, but exceptions cluster.

### EVT

- Observed exceptions: 65
- Exception rate: 1.52%
- Kupiec p-value: 0.0014
- Christoffersen IND p-value: 0.0197
- Conditional coverage p-value: 0.0004
- Interpretation: rejects unconditional coverage on the new harmonised P&L basis. EVT catches the tail shape but its mean VaR (~$28.9k) leaves it short of the realised loss series; exceptions also cluster.

### GARCH-EVT

- Observed exceptions: 47
- Exception rate: 1.10%
- Kupiec p-value: 0.5142
- Christoffersen IND p-value: 0.5471
- Conditional coverage p-value: 0.6743
- Interpretation: strongest backtest result in the current outputs. It passes unconditional coverage, independence, and conditional coverage.

### Monte Carlo

- Observed exceptions: 64
- Exception rate: 1.59%  (4,019-day window, expected ≈ 40.2)
- Kupiec LR_UC: 12.08  |  p-value: 0.0005
- Christoffersen IND LR: 25.65  |  p-value: < 0.0001
- Conditional coverage LR_CC: 37.73  |  p-value: < 0.0001
- Interpretation: the MC linear-leg basis mismatch documented before May 2026 has been fixed. The simulated linear P&L is now built from fixed inception share counts re-marked at the current rolling prices (`shares_j = V0 · w_j / P_{0,j}`, then `DV_i = sum_j shares_j · P_{t-1,j} · (exp(sim_i[j]) − 1)`), which matches `compute_pnl.py` exactly. On the joined backtest CSV the MC realised-loss column now equals `-pnl_total` to the cent, and the mean MC VaR is $27.8k versus mean HistSim VaR $33.6k (ratio 0.83), well within plausible range. The remaining over-exception is genuinely model-driven: Gaussian MC understates fat tails (no Student-t / EVT tail shape) and ignores volatility clustering (constant 750-day Σ), so Kupiec and both Christoffersen tests still reject. This is what a Gaussian full-revaluation MC is expected to fail on; t-copula and GARCH-t-copula variants in `outputs/tables/backtest_mc_t_copula.csv` and `backtest_mc_garch_t_copula.csv` address those shortcomings. Guard tests for the basis fix live in `tests/test_mc_backtest_basis.py`.

### Copula (legacy Python wrapper output)

- Observed exceptions: 188
- Exception rate: 4.40%
- Kupiec p-value: < 0.0001
- Christoffersen IND p-value: 0.0002
- Conditional coverage p-value: < 0.0001
- Interpretation: large over-exception count on the documented P&L basis. This CSV is retained as a legacy output from the deleted Python wrapper, not as the canonical copula implementation. The copula mean VaR (~$19.3k) is well below the realised loss series (max ≈ $85.5k); the gap is consistent with the wrapper's documented simplification that nonlinear (IRS + straddle) P&L was independently bootstrapped from the linear copula scenarios, which understates tail co-movement. New copula work should use the R pipeline in `src/var_methods/GARCH-Model.R`.

## Comparability Status (post-PR #28, post May-2026 MC basis fix)

- EVT, GARCH-EVT, MC, and legacy Copula use canonical `-pnl_total` as `actual_loss` on their backtest dates. HistSim shares the same headline realised-loss range (max ≈ $85,514), but reconstructs portfolio state internally and differs from canonical `-pnl_total` on 156 / 4,269 dates by up to about $5.3k, mostly around straddle roll-over mechanics.
- HistSim / EVT / GARCH-EVT / legacy Copula use a 4,269-day backtest; MC uses 4,019 days because its rolling estimation window is 750 vs 500. On the overlapping 4,019 dates, MC `actual_loss` equals `-pnl_total` to the cent.
- Cross-method comparison of exception counts is now materially apples-to-apples on the realised side, with the HistSim residual documented above, and on the simulated side after this branch's MC basis fix.
- The simulated linear-leg basis mismatch noted in earlier revisions has been resolved; see the Monte Carlo section above.

## Best Discussion Points With Professor

1. Is GARCH-EVT the preferred statistical result because it now passes Kupiec, Christoffersen IND, and conditional coverage?
2. Is HistSim still defensible as baseline because it passes Kupiec (and the lecture binomial threshold) but fails independence in an economically interpretable way?
3. Now that the Monte Carlo linear-leg basis is aligned with `compute_pnl.py`, the residual over-exception (64 / 4,019 ≈ 1.59 %) is a Gaussian-tail and volatility-clustering finding, not a basis bug. Is the Gaussian full-reval MC useful as a *diagnostic* benchmark — to show how much fat tails and time-varying Σ matter — even though it fails Kupiec on its own?
4. Does the legacy Copula over-exception flag the documented simplification (independent bootstrap of nonlinear P&L)? Should the team commit to the R pipeline's full joint treatment for the final write-up, or interpret the current Python-wrapper output as a known-undershoot baseline?
5. Given that EVT / GARCH-EVT / MC / legacy Copula are on canonical `-pnl_total` and HistSim has only the documented state-reconstruction residual, is the cross-method ranking acceptable for the report? (The previous larger basis mismatch was resolved as part of PR #28.)

## Short Overall Interpretation

- GARCH-EVT performs best statistically — it now passes unconditional coverage, independence, and conditional coverage on the harmonised P&L series.
- HistSim is well calibrated on average (42 exceptions vs 42.7 expected) but produces clustered exceptions, so it fails independence.
- EVT rejects unconditional coverage on the harmonised series; its tail-shape fit is sound but the level is too low for the realised loss distribution.
- Legacy Copula over-exceeds heavily on the documented series, consistent with the independent-bootstrap simplification for nonlinear P&L documented in the README.
- Monte Carlo over-exceeds at 1.59% (64 in 4,019 days) on the post-fix basis; this is now a genuine Gaussian-tail / volatility-clustering finding rather than the V0 basis bug — the linear leg uses fixed inception share counts at current prices, matching `compute_pnl.py`. See the Monte Carlo section.
- EVT, GARCH-EVT, MC, and legacy Copula use canonical `-pnl_total`; HistSim has a small documented reconstruction residual. The remaining major issues are method-internal rather than the old realised-loss basis mismatch.
