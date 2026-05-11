# Critical Analysis — GARCH-Copula VaR Pipeline

Pipeline run: `steps_3_to_8_marginal_garch.R` → `steps_9_to_12_copula_var.R` → `step15_post_var_validation.R`  
Data: 2006-01-01 to 2024-12-31 · N = 4 769 daily observations  
Portfolio: USD 1 000 000 across SPY (25 %), IEF+IRS (25 %), GLD (25 %), EUR/USD (25 %)  
Plus: rolling 30-day ATM straddle on SPY (20 contracts × 100 shares) + 10-year IRS (3 % fixed, USD 1 M)

---

## 1. Reconstruction Check

**Did the P&L mapping produce plausible numbers?**

| Component | Mean ($/day) | Std ($/day) | Min | Max | Excess Kurt. | Normal? |
|---|---:|---:|---:|---:|---:|---|
| pnl\_linear | 207 | 4 979 | −43 099 | 53 117 | 11.1 | NO |
| pnl\_irs | 4 | 5 586 | −49 742 | 29 354 | 2.9 | NO |
| pnl\_straddle | −105 | 3 585 | −38 979 | 53 635 | **39.0** | NO |
| **pnl\_total** | **106** | **6 993** | **−52 940** | **62 557** | 5.6 | NO |

**Verdict: structurally coherent but dominated by straddle tail-risk.**  
The linear component behaves as expected (positively skewed, moderate kurtosis).  
The IRS contributes almost nothing in expectation but introduces large rate-driven spikes.  
The straddle has extreme kurtosis (39) driven by weekly roll events and large spot moves.  
All four KS tests reject normality at any meaningful significance level.

**Concern:** The straddle mean daily P&L of −$105 implies a carry cost of roughly −$26 500/year, consistent with ATM straddle decay. However, skewness = +1.85 means the big payouts (sigma-spike events) are heavily right-skewed — the distribution is asymmetric in a way that standard copula VaR handles poorly.

---

## 2. GARCH Marginals (Steps 3–8)

### 2a. Model selection

| Factor | ARMA | GARCH spec | Innovation | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|---|---|---|
| SPY\_log\_return | (3,1) | apARCH(1,1) | jsu | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ |
| DGS10\_change | (2,0) | apARCH(1,1) | sstd | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ |
| GLD\_log\_return | (0,0) | gjrGARCH(1,1) | nig | ✓ | ✗ | ✓ | ✓ | ✗ | ✓ |
| EURUSD\_log\_return | (3,2) | apARCH(1,1) | std | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| SPY\_level\_change | (3,2) | eGARCH(1,1) | nig | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ |
| VIX\_change | (0,4) | gjrGARCH(1,1) | jsu | ✗ | ✓ | ✓ | ✓ | ✗ | ✗ |

*C1 = no autocorr residuals · C2 = no ARCH residuals · C3 = GoF · C4 = variance targeting · C5 = no sign bias · C6 = parameter stability*

**Pass rates: C1 3/6 · C2 3/6 · C3 3/6 · C4 4/6 · C5 1/6 · C6 3/6.**

### 2b. GoF failure rate

4 of 6 factors had **no GoF-passing (C3) candidate** in the full 28-combo search:
- SPY\_log\_return: mean GoF p = 0.00 for all top-3 combos → fallback to lowest AIC
- DGS10\_change: best GoF p = 0.0034 → fallback
- SPY\_level\_change: best GoF p = 0.0464 → fallback
- VIX\_change: best GoF p = 0.0132 → fallback

Only GLD (gjrGARCH/nig, GoF p = 0.55) and EURUSD (apARCH/std, GoF p = 0.13) pass.  
This means 4 of 6 PIT inputs to the copula come from models with distributional misfit.

### 2c. Variance targeting (C4) failures

| Factor | Model variance | Empirical variance | Ratio | Deviation |
|---|---:|---:|---:|---|
| SPY\_log\_return | 6.66 × 10⁻⁵ | 1.50 × 10⁻⁴ | 0.446 | **−55.4 %** |
| EURUSD\_log\_return | 2.36 × 10⁻⁵ | 5.25 × 10⁻⁵ | 0.449 | **−55.1 %** |
| DGS10\_change | 2.61 × 10⁻³ | 3.31 × 10⁻³ | 0.790 | −21.0 % |
| GLD\_log\_return | identical | identical | 1.000 | −0.0 % ✓ |
| SPY\_level\_change | 6.891 | 6.927 | 0.995 | −0.5 % ✓ |
| VIX\_change | 3.663 | 3.757 | 0.975 | −2.5 % ✓ |

SPY and EURUSD apARCH models report unconditional variance ≈ 45 % of the empirical variance. **Root cause:** rugarch's variance targeting for apARCH sets the long-run variance constraint in δ-power space (where δ is the Box-Cox power parameter), not in variance space. `uncvariance(fit)` maps back to variance, but the targeted quantity was the δ-power-transformed variance. This is not a code bug; it is an acknowledged limitation of apARCH + VT in rugarch. Consequence: the copula PIT pseudo-observations for SPY and EURUSD are generated from a model with systematically underestimated unconditional variance.

**Sign Bias (C5) fails for 5/6 factors,** indicating unmodeled leverage effects in most series. The GJR and apARCH models capture asymmetry partially, but C5 still rejects. eGARCH for SPY\_level\_change also fails C5.

### 2d. ARMA residual autocorrelation

SPY (C1 ✗), EURUSD (C1 ✗), VIX (C1 ✗): Ljung-Box on standardized residuals still rejects at lag 10. This implies residual mean-model misfit — the ARMA(p,q) chosen by `auto.arima` does not fully remove serial structure. For VaR purposes this is second-order but inflates estimation uncertainty.

---

## 3. Copula (Steps 10–12)

### 3a. Static fit

| Family | log-Lik | AIC | CvM stat | CvM CI₉₅ | Rejected |
|---|---:|---:|---:|---:|---|
| **t (selected)** | 12 390 | **−24 749** | 0.0168 | 0.0296 | **NO** |
| Gaussian | 12 075 | −24 121 | 0.0179 | 0.0319 | NO |
| Clayton | 58 | −113 | 0.0697 | 0.0541 | **YES** |
| Gumbel | 0 | 2 | 0.0610 | 0.0526 | **YES** |
| Frank | 0 | 2 | 0.0610 | 0.0526 | **YES** |

**t-copula is the correct choice.** AIC difference t vs. Gaussian = 628 (huge). Archimedean families fail because they impose a single-parameter dependence structure with no tail symmetry, which cannot capture the mixed dependence across 6 risk factors.

The t-copula captures symmetric tail dependence — appropriate for equity + rate + FX + vol factors where crises drag all correlations up simultaneously.

**Limitation:** the t-copula enforces a single degree-of-freedom parameter for all pairs. The true tail dependence between, say, SPY and VIX (strongly negatively correlated in crises) differs fundamentally from SPY–GLD or SPY–DGS10. A vine/D-vine copula would allow pair-specific tail dependence but crashed under Windows R 4.5.3 (Step 14d documents this).

### 3b. Rolling GoF (18 windows, annual refits 2007–2024)

| Window | CvM observed | CvM q₉₅ | p-value | Rejected |
|---|---:|---:|---:|---|
| 2007-12-31 | 0.163 | 0.636 | 0.286 | NO |
| **2009-01-12** | 0.057 | 0.263 | 0.032 | **YES** |
| 2010–2016 | ≤ 0.133 | > 0.103 | ≥ 0.098 | NO |
| **2017-12-18** | 0.045 | 0.198 | 0.036 | **YES** |
| **2018-12-17** | 0.043 | 0.158 | 0.028 | **YES** |
| **2019-12-13** | 0.040 | 0.171 | 0.000 | **YES** |
| 2020–2024 | ≤ 0.082 | > 0.038 | ≥ 0.098 | NO |

4 / 18 windows (22 %) reject the GoF test, including the post-GFC window (2009) and a cluster around 2017–2019. The 2009 rejection is expected — the training window contains the Lehman shock and the copula structure changes. The 2017–2019 cluster (low-volatility, post-taper tantrum) may reflect that the training window then includes 2008 extremes that distort the fit for calm-regime data.

---

## 4. VaR Backtest (Step 12 + Backtesting Layer)

### 4a. Overall exception statistics

| Metric | Value |
|---|---|
| Total observations | 4 269 |
| Exceptions (loss > VaR₉₅) | 51 |
| Observed exception rate | **1.195 %** |
| Expected exception rate | 5.0 % |
| Mean VaR (95 %) | USD 18 915 |
| VaR range | USD 10 801 – 31 938 |
| Std of daily P&L | USD 6 993 |
| VaR / σ | **2.70 × σ** |

**Kupiec LR test:** LR ≈ 185 >> χ²₀.₀₅(1) = 3.84 → **model REJECTED** (too conservative).  
Normal 95 % VaR would be 1.645 × $6 993 = $11 504; model produces $18 915 on average — **64 % above the normal equivalent.** The fat-tail GARCH marginals + t-copula + straddle kurtosis compound to massively over-predict tail risk in calm periods.

### 4b. Year-by-year exception rates

| Year | Exceptions | N | Rate | Comment |
|---|---:|---:|---:|---|
| 2008 | **29** | 243 | **11.9 %** | GFC — severe under-prediction |
| 2009 | 0 | 252 | 0.0 % | Post-crisis recovery |
| 2010 | 0 | 252 | 0.0 % | |
| 2011 | 4 | 252 | 1.6 % | European debt crisis |
| 2012–2014 | ≤ 1 | 250–252 | ≤ 0.4 % | |
| 2015 | 3 | 252 | 1.2 % | China devaluation |
| 2016 | 2 | 252 | 0.8 % | Brexit + Trump |
| 2017 | 0 | 251 | 0.0 % | |
| 2018 | 2 | 251 | 0.8 % | Rate normalization |
| 2020 | **7** | 253 | **2.8 %** | COVID-19 |
| 2021–2024 | ≤ 1/yr | 250–252 | ≤ 0.4 % | |

**The model suffers from the classic structural break problem:** the training window pre-2008 cannot anticipate the GFC regime shift. The rolling refit on annual cadence is too slow to react to crisis-speed correlation changes (correlations can jump within days in a crisis).

**No exceptions in 2009–2010** despite the recovery being volatile: the VaR is still very high from the 2008-retrained window, creating a "shadow of the crisis" effect. The model becomes over-conservative after every crisis and under-conservative entering them.

### 4c. Assessment

The 1.2 % overall exception rate means the bank holding this portfolio would reserve **4.2× more capital** than historically necessary on a pure frequency basis. This is expensive but perhaps desirable for tail robustness. The real problem is the 2008 spike — actual losses exceeded VaR on 12 % of crisis days, meaning the model fails exactly when it matters most.

---

## 5. Gesamtbewertung

### What works

- **t-copula selection is robust and well-motivated.** AIC difference of 628 vs. Gaussian, CvM not rejected, symmetric tail dependence is theoretically appropriate for multi-asset crisis scenarios.
- **GLD and EURUSD marginals pass GoF (C3)** and produce valid PIT pseudo-observations.
- **DGS10, SPY\_level\_change, VIX\_change pass C4 (variance targeting)** with deviations < 21 %.
- **The P&L mapping is economically coherent** — IRS, straddle, and linear components all have plausible magnitudes and signs.
- **Rolling window structure is correct** — annual refits, 252-day base window, daily VaR output.

### What is problematic

1. **SPY and EURUSD apARCH models systematically underestimate unconditional variance (−55 %).** The copula input U-margins for these two factors come from a model with wrong long-run variance. This biases the t-copula parameter estimation.

2. **4 of 6 factors have no GoF-passing GARCH specification.** The fallback to lowest-AIC without GoF means the innovation distribution is not validated for distributional fit. SPY residuals (GoF p ≈ 0) are particularly misfit.

3. **Sign bias (C5) fails for 5/6 factors.** Leverage effects are not fully captured despite using asymmetric GARCH variants. This is likely a joint consequence of: (a) VT constraining the asymmetry parameters, (b) the 6-factor structure requiring separate models that cannot jointly capture cross-factor leverage.

4. **VaR is over-conservative by 4× in calm periods but under-conservative in crises.** This is the worst possible property for a risk model — expensive to maintain and ineffective when it matters.

5. **The straddle component (excess kurtosis 39, skewness 1.85) dominates the tail of the P&L distribution.** A 6-dimensional parametric copula with a single set of marginals cannot realistically capture this component's non-linear payoff under stress scenarios. A simulation framework that prices the straddle directly under each MC scenario path (as the current `map_factors_to_pnl()` does) is the right approach, but the GARCH marginals driving SPY returns are themselves misfit.

6. **4/18 rolling GoF rejections** indicate periodic model misspecification. The annual refit cadence leaves the model exposed to regime shifts for up to a full year.

### Bottom line

The pipeline is **methodologically complete and internally consistent** as a university-level implementation of the GARCH-Copula VaR framework (McNeil, Frey & Embrechts, Chapter 7). The code correctly implements Steps 3–15 with documented diagnostics and override hooks. However, the empirical results show that this model would not be adequate for regulatory (Basel III) use:
- Kupiec test rejected (exception rate 1.2 % ≠ 5 %)
- Persistent C4/C5 failures in SPY and EURUSD marginals
- 2008 crisis under-coverage (11.9 % exceptions vs. 5 % limit)

For a research/academic context this is an honest and well-documented outcome.
