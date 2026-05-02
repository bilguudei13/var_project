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
- Kupiec p-value: 0.9152
- Christoffersen IND p-value: < 0.0001
- Conditional coverage p-value: 0.0002
- Interpretation: unconditional coverage is excellent, but exceptions cluster.

### EVT

- Observed exceptions: 56
- Exception rate: 1.31%
- Kupiec p-value: 0.0507
- Christoffersen IND p-value: 0.0064
- Conditional coverage p-value: 0.0036
- Interpretation: borderline unconditional coverage, independence fails. EVT catches tails better than normal methods but still clusters.

### GARCH-EVT

- Observed exceptions: 41
- Exception rate: 0.96%
- Kupiec p-value: 0.7936
- Christoffersen IND p-value: 0.4132
- Conditional coverage p-value: 0.6914
- Interpretation: strongest backtest result in the current outputs. It passes unconditional coverage and independence.

### Monte Carlo

- Observed exceptions: 320
- Exception rate: 7.50%
- Kupiec p-value: ~0
- Christoffersen IND p-value: ~0
- Conditional coverage p-value: ~0
- Interpretation: this output looks problematic for a 99% VaR. Either the MC VaR is too low, the backtest P&L basis differs, or there is an implementation/calibration issue.

### Copula

- Observed exceptions: 51
- Exception rate: 1.19%
- Kupiec p-value: 0.2149
- Christoffersen IND p-value: 0.0003
- Conditional coverage p-value: 0.0007
- Interpretation: unconditional coverage is acceptable, but exceptions are not independent.

## Important Comparability Issue

- The realised loss series is not identical across all method backtest files.
- HistSim and Monte Carlo use a realised loss series with max loss around $85.5k.
- EVT, GARCH-EVT, and Copula use a realised loss series with max loss around $52.9k.
- This means method comparisons may not be fully apples-to-apples unless the team intentionally used different backtesting P&L bases.
- This is the most important thing to clarify with the professor/team before final interpretation.

## Best Discussion Points With Professor

1. Should every VaR method be backtested against the same realised portfolio P&L?
2. Is GARCH-EVT the preferred statistical result because it passes both Kupiec and Christoffersen?
3. Is HistSim still defensible as baseline because it passes Kupiec but fails independence in an economically interpretable way?
4. Does the Monte Carlo output indicate a modelling issue, or is it based on a different portfolio/P&L convention?
5. Should the final report compare methods directly, or first state differences in backtesting basis?

## Short Overall Interpretation

- GARCH-EVT performs best statistically.
- HistSim is well calibrated on average but produces clustered exceptions.
- EVT and Copula are acceptable on unconditional coverage but fail independence.
- Monte Carlo needs review because 320 exceptions is far too many for a 99% VaR.
- Before ranking methods, the team should verify that all backtests use the same realised P&L definition.
