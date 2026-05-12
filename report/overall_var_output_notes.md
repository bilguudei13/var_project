# Overall VaR Output Notes

## Common Setup

- All methods target 1-day 99% VaR.
- Backtest sample length in available outputs: 4,269 observations.
- Expected exceptions at 99%: 42.7.
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

- Observed exceptions: 320
- Exception rate: 7.50%
- Kupiec p-value: ~0
- Christoffersen IND p-value: ~0
- Conditional coverage p-value: ~0
- Interpretation: this output looks problematic for a 99% VaR. Root cause appears to be a linear-book basis mismatch: `src/var_methods/monte_carlo.py` scales the simulated linear P&L by the initial portfolio notional `V0 = $1,000,000`, while the realised P&L in `total_portfolio_pnl.csv` is built from fixed share counts and therefore scales with the drifted current portfolio value. On the joined backtest CSVs, MC and HistSim share an essentially identical realised loss series (mean |diff| ≈ $34), yet MC mean VaR is $16.5k vs HistSim $33.1k — roughly a 2× gap that tracks the SPY appreciation over 2007–2024. **Treat MC as a diagnostic output, not a fully comparable method ranking, until the linear leg is rescaled with the rolling current portfolio value and the dependent backtest CSV and figure 07 are regenerated.**

### Copula

- Observed exceptions: 188
- Exception rate: 4.40%
- Kupiec p-value: < 0.0001
- Christoffersen IND p-value: 0.0002
- Conditional coverage p-value: < 0.0001
- Interpretation: large over-exception count on the harmonised P&L basis. The copula mean VaR (~$19.3k) is well below the realised loss series (max ≈ $85.5k); the gap is consistent with the documented assumption that nonlinear (IRS + straddle) P&L is independently bootstrapped from the linear copula scenarios, which understates tail co-movement. See the known-limitations section in the README.

## Comparability Status (post-PR #28)

- The realised loss series is now identical across the five method backtest files (max ≈ $85,514, min ≈ −$74,863, n = 4,269 observations).
- Cross-method comparison of exception counts is therefore apples-to-apples; the previous EVT/GARCH-EVT/Copula vs HistSim/MC basis mismatch was resolved as part of PR #28.
- The remaining caveat is the Monte Carlo basis mismatch on the simulation side (initial V0 vs current portfolio value) — see the Monte Carlo section above.

## Best Discussion Points With Professor

1. Is GARCH-EVT the preferred statistical result because it now passes Kupiec, Christoffersen IND, and conditional coverage?
2. Is HistSim still defensible as baseline because it passes Kupiec (and the lecture binomial threshold) but fails independence in an economically interpretable way?
3. Does the Monte Carlo over-exception flag a modelling issue (linear leg fixed to initial V0)? Should MC be re-run with rolling current portfolio value before it is used for ranking?
4. Does the Copula over-exception flag the documented simplification (independent bootstrap of nonlinear P&L)? Should the team commit to the full joint treatment for the final write-up, or interpret the current copula output as a known-undershoot baseline?
5. Now that all five methods share the same realised P&L series, is the cross-method ranking acceptable for the report? (The previous basis mismatch was resolved as part of PR #28.)

## Short Overall Interpretation

- GARCH-EVT performs best statistically — it now passes unconditional coverage, independence, and conditional coverage on the harmonised P&L series.
- HistSim is well calibrated on average (42 exceptions vs 42.7 expected) but produces clustered exceptions, so it fails independence.
- EVT rejects unconditional coverage on the harmonised series; its tail-shape fit is sound but the level is too low for the realised loss distribution.
- Copula over-exceeds heavily on the harmonised series, consistent with the independent-bootstrap simplification for nonlinear P&L documented in the README.
- Monte Carlo still over-exceeds at 7.50%; root cause is the linear-leg basis mismatch (constant initial V0 in MC vs drifted current portfolio value on the realised side) — see the Monte Carlo section.
- All five backtests now use the same realised P&L series, so the cross-method comparison is apples-to-apples on the *target* side; the remaining issues are method-internal.
