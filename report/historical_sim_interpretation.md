# Historical Simulation Interpretation

## One baseline model, not a HistSim family

The Historical Simulation contribution in this repository is intentionally presented as **one official baseline model**: a 1-day 99% full-repricing Historical Simulation VaR. This is more scientifically defensible for the project than introducing several small HistSim variants, because the emphasis moves to correctness, backtesting, extreme-loss interpretation, and model limitations.

## Why Kupiec can pass while Christoffersen fails

Over the full sample, the baseline HistSim model records 42 exceptions against 42.7 expected exceptions, with a Kupiec p-value of 0.9152. This indicates that the unconditional breach frequency is close to the theoretical 1% target.

At the same time, the Christoffersen independence p-value is < 0.0001, so independence is rejected. The central interpretation is that Historical Simulation reuses a fixed rolling historical window and therefore adjusts only gradually when volatility regimes change. The model can therefore match the average breach frequency over a long sample while still producing clustered exceptions in fast-moving stress regimes.

## Why the 500-day 99% setup is stepwise

With a 500-day rolling window at the 99% level, the VaR estimate is anchored by only 5 active tail scenarios in each window. This makes the baseline model transparent but also discrete and regime-sensitive: a small number of recurring historical shocks can dominate the VaR estimate for long stretches of time. The most persistent historical tail shock in the rolling windows is 2008-12-01, which appears 500 times among the active tail scenarios. Under a 500-day window this is the structural maximum, meaning the shock remained in the tail on every day it was eligible.

## Crisis windows and stress interpretation

The GFC window is the clearest example of local stress for the baseline model: 7 exceptions over 200 observations, with a Kupiec p-value of 0.0056. The COVID window is also highly informative: 6 exceptions over 70 observations and mean VaR of $40,071. This shows that short crisis samples can display sharply elevated breach rates and large realised losses even when independence is not rejected within the subperiod itself.

## Which part of the book drives the HS losses

The component-wise HS analysis shows that the largest **standalone average component VaR** comes from the Linear Book, with mean component VaR of $32,550. This should be read as a standalone risk diagnostic rather than a portfolio-VaR decomposition: the component VaRs do not add up to total portfolio VaR because each component can have a different set of worst tail scenarios in the 500-day window. In the extreme realised-loss table, the most frequent largest driver is the Linear Book.

## Window robustness

The window-sensitivity check keeps the same Historical Simulation methodology but varies the historical lookback. The shortest tested window (250) produces mean VaR of $29,268, while the longest tested window (750) produces mean VaR of $33,593. This confirms that the choice of historical window materially affects responsiveness, exception behaviour, and tail severity even when the underlying HS model is unchanged. In this dataset, the baseline 500-day specification is also the only tested window that does **not** reject Kupiec unconditional coverage: the 250-day window has p-value 0.0006, the 500-day window has p-value 0.9152, and the 750-day window has p-value 0.0096. This comparison should still be read as a robustness check rather than a perfectly controlled horse race, because the alternative windows are evaluated on different effective forecast samples after their own burn-in periods.
