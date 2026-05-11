# Code Structure & Test Map — GARCH-Copula VaR Pipeline

---

## 1. Files Overview

```
var_project/
├── data/
│   ├── raw/                         prices.csv · vix.csv · dgs10.csv
│   └── processed/                   generated outputs (git-ignored)
├── src/
│   ├── data/
│   │   ├── config.py                Portfolio constants (tickers, weights, V0, IRS, straddle)
│   │   ├── download_data.py         yfinance + FRED download → data/raw/
│   │   ├── compute_returns.py       Log returns, VIX returns, DGS10 changes → processed/
│   │   ├── compute_pnl.py           Instrument P&L via portfolio_pricing.py → processed/
│   │   └── portfolio_pricing.py     Pricing: price_irs(), price_straddle_position(),
│   │                                build_straddle_state(), swap_annuity()
│   └── var_methods/
│       ├── steps_3_to_8_marginal_garch.R   Phase B: GARCH marginals (Steps 3–8)
│       ├── steps_9_to_12_copula_var.R      Phase C+D: Copula + VaR (Steps 9–12, 14)
│       └── step15_post_var_validation.R    Phase E: Post-VaR validation (Layer A–D)
├── outputs/
│   ├── figures/                     All PNG plots (garch_step*, step8*, step9*, step10*, …)
│   └── tables/                      All CSV diagnostics (garch_summary.csv, …)
└── report/                          LaTeX/Markdown report (separate)
```

---

## 2. Pipeline Flow

```
download_data.py          →  data/raw/{prices,vix,dgs10}.csv
compute_returns.py        →  data/processed/risk_factors.csv
                              (6 columns: SPY_log_return, DGS10_change, GLD_log_return,
                               EURUSD_log_return, SPY_level_change, VIX_change)

source("steps_3_to_8_marginal_garch.R")
  Step 3:  ADF test per factor
  Step 4:  ARMA mean model selection (auto.arima, d=0)
  Step 5:  ARCH-LM + Ljung-Box on squared residuals
  Step 6:  4 GARCH types × 7 distributions → pick lowest AIC + GoF pass
  Step 7:  6-criteria validation of chosen model
  Step 8:  PIT → gamlss family fit on standardised residuals → U_t ∈ (0,1)
  Exports: results list, factor_names, factors_mat, factor_dates, FIG_DIR, TBL_DIR

source("steps_9_to_12_copula_var.R")   [requires steps_3_to_8 in workspace]
  SETUP:  reticulate bridge → portfolio_pricing.py (IRS, straddle pricing)
  Step  9:  pseudo-observations U_param = pobs(factors_mat)
  Step 10:  copula family selection (AIC + CvM GoF) → results$copula_fit
  Step 11:  MC simulation (N=50 000) → Zstar → GARCH σ forecast → portfolio P&L
  Step 12:  static VaR/ES + rolling VaR (annual refits, 252-day window)
  Step 14:  sensitivity / stress analysis + vine robustness (subprocess)
  Exports: results$var_rolling (attr: fits, window, step)

source("step15_post_var_validation.R")  [requires steps_9_to_12 in workspace]
  Layer A:  pooled in-sample re-validation (mirrors Step 7 criteria)
  Layer B:  per-window out-of-sample stability
  Layer C:  exception-based residual diagnostics
  Layer D:  variance targeting deviation plot
```

**Execution order is strict.** Steps 9–12 consume `results$garch_fit` from Steps 3–8. Step 15 consumes `results$var_rolling` with its attached `fits` attribute.

---

## 3. Test Map

Every statistical test in the pipeline, with its location, null hypothesis, desired outcome, and failure consequence.

| Step | Test | Location | H₀ | Want | Fail consequence |
|---|---|---|---|---|---|
| 3 | ADF (Augmented Dickey-Fuller) | `adf.test(y)` in loop | Unit root (non-stationary) | Reject H₀ (p < 0.05) | Factor may be non-stationary → invalid log-return framework |
| 4 | Ljung-Box on ARMA residuals | `Box.test(resid, lag=10)` | No autocorrelation | Do not reject (p > 0.05) | Serial structure remains in mean; biases GARCH residuals |
| 5 | ARCH-LM test | `FinTS::ArchTest(ri, lags=10)` | No ARCH effects | Reject H₀ (p < 0.05) | No justification for GARCH modelling |
| 5 | Ljung-Box on squared residuals | `Box.test(ri^2, lag=10)` | No autocorr in squared returns | Reject H₀ | Same: confirms volatility clustering |
| 6 | GoF (Pearson χ², rugarch) | `gof(fit, groups=c(20,30,40,50))` | Innovation distribution correct | Do not reject (mean p > 0.05) | Innovation distribution misfit → bad PIT pseudo-obs |
| 7 (C1) | Weighted Ljung-Box on Ẑ | `Weighted.Box.test(Zh, lag=10, fitdf=p+q)` | No residual autocorrelation | p > 0.05 | ARMA not adequate for mean |
| 7 (C2) | Weighted LB on Ẑ² + ARCH-LM | same on `Zh^2` | No residual ARCH | p > 0.05 | GARCH not capturing all volatility clustering |
| 7 (C3) | Pearson GoF | `gof(fit)` | Innovation distribution correct | mean p > 0.05 | Same as Step 6 |
| 7 (C4) | Variance targeting | `uncvariance(fit)` vs `var(y)` | Model var ≈ empirical var | ratio ∈ [0.75, 1.25] | Long-run risk mis-calibrated |
| 7 (C5) | Sign Bias test | `signbias(fit)` | No leverage effect | p > 0.05 | Unmodelled asymmetry; tails mis-shaped |
| 7 (C6) | Nyblom parameter stability | `nyblom(fit)` | Stable parameters over time | Joint stat < critical | Parameters drift → rolling estimates unreliable |
| 8 | KS test on PIT uniforms | `ks.test(U, "punif")` (implicit via gamlss) | PIT ~ Uniform(0,1) | Passes (p > 0.05) | Copula input margins are not uniform → copula estimation biased |
| 9 | Visual check | `pairs(U_param)` | Pseudo-obs near uniform margins | Histograms flat | Same as above |
| 10 | Cramér-von-Mises GoF (upper-tail) | `cvm_gof()` per family | Simulated copula matches empirical | stat < q₀.₉₅ of H₀ dist | Copula family rejected for this data |
| 12 | Rolling CvM GoF | `cvm_gof()` per window | Fit adequate in this window | p > 0.05 (not rejected) | Model misfit for this sub-period |
| 15A | All C1–C6 in-sample | `validate_one_fit()` | Same as Step 7 | Mirror of Step 7 results | Confirms Step 7 is consistent |
| 15B | C1–C6 per rolling window | `validate_one_fit()` on each window fit | Stable out-of-sample diagnostics | Consistent pass rates | Rolling instability → VaR unreliable across regimes |
| 15C | Exception clustering | Kupiec LR, Christoffersen independence | Exception rate = 5 %; no clustering | LR < χ²(1) critical | VaR model rejected by backtest |
| 15D | VT deviation | `uncvariance` vs window `var(y)` | |deviation| < 25 % per window | Flags only | Tracks apARCH VT bias across time |

---

## 4. Override Hooks

All overrides must be set **before** sourcing the relevant script. They are removed by setting to `NULL`.

### steps_3_to_8_marginal_garch.R

```r
# Force specific GARCH lag order (default: c(1,1) for all factors)
manual_garch_order <- list(SPY_log_return = c(1, 1), DGS10_change = c(2, 1))

# Restrict GARCH model type candidates (default: all four)
# Values: "sGARCH", "gjrGARCH", "eGARCH", "apARCH"
manual_garch_model <- list(SPY_log_return = "gjrGARCH")

# Force a specific innovation distribution (bypasses AIC+GoF selection)
# Values: "norm", "std", "sstd", "nig", "jsu", "snorm", "ged"
manual_garch_dist  <- list(SPY_log_return = "sstd", EURUSD_log_return = "sstd")

# Force PIT marginal distribution for copula input (bypasses gamlss selection)
# Values: gamlss family names, e.g. "SST", "JSU", "TF", "NO"
manual_pit_dist    <- list(SPY_log_return = "SST")
```

### steps_9_to_12_copula_var.R

```r
# Force copula family for static fit (bypasses AIC+CvM selection)
# Values: "t", "normal", "clayton", "gumbel", "frank"
manual_copula <- "t"

# Force rolling engine: "vine" (default if rvinecopulib available) or "single"
options(manual_copula_engine = "single")

# Override portfolio weights for MC scenario (K-vector, renormalised internally)
manual_weights <- c(0.4, 0.3, 0.2, 0.1)

# Override portfolio weights for stress scenario runs (Step 14a)
# Named list of numeric vectors, each length K
manual_weights_scenarios <- list(
  equity_heavy  = c(0.6, 0.1, 0.2, 0.1),
  bond_heavy    = c(0.1, 0.6, 0.2, 0.1)
)

# Hard-code rolling window size (default: chosen by AIC-stability diagnostic)
manual_window <- 500
```

---

## 5. Output Artifacts

### Figures (`outputs/figures/`)

| File | Produced by | Content |
|---|---|---|
| `garch_step3_adf_<fct>.png` | Step 3 | Return series + rolling mean/sd (ADF title) |
| `garch_step4_arma_<fct>.png` | Step 4 | ACF/PACF of returns + LB p-value |
| `garch_step5_arch_<fct>.png` | Step 5 | ACF of squared returns + LM test |
| `garch_step6a_densities_<fct>.png` | Step 6 | Density overlay (data vs best fit) |
| `garch_step6b_qq_<fct>.png` | Step 6 | QQ-plot of standardized residuals |
| `step6a_dist_bars_<fct>.png` | Step 6 | AIC heatmap (model × distribution) |
| `step6b_zhat_qq_<fct>.png` | Step 6 | QQ residuals with multiple model overlays |
| `garch_step7_<fct>.png` | Step 7 | Sign-bias plot + Nyblom trace |
| `step8a_pit_density_<fct>.png` | Step 8 | PIT uniform density check |
| `step8b_pit_qq_<fct>.png` | Step 8 | PIT QQ vs Uniform(0,1) |
| `step9a_uniform_histograms.png` | Step 9 | Marginal U-histograms (should be flat) |
| `step9b_pairwise_pseudos.png` | Step 9 | Scatterplot matrix of U_param |
| `step10a_copula_aic.png` | Step 10 | AIC bar chart per copula family |
| `step10b_cvm_nulldist_<fam>.png` | Step 10 | H₀ distribution + observed CvM stat |
| `step10c_simulated_vs_empirical.png` | Step 10 | Simulated vs empirical pairwise copula |
| `var_copula_03_var_results.png` | Step 12 | Rolling VaR₉₅ time series |

### Tables (`outputs/tables/`)

| File | Content |
|---|---|
| `pre_garch_tests.csv` | LB and ARCH-LM test statistics for all 6 factors (Step 5) |
| `step6_dist_comparison_<fct>.csv` | Full 28-row (model × dist) AIC/GoF table per factor |
| `garch_dist_selection_top3.csv` | Top-3 candidates per factor |
| `garch_summary.csv` | Final ARMA/GARCH/dist + C1–C6 pass/fail per factor |
| `garch_variance_check.csv` | C4 detail: empirical vs model variance, ratio, persistence |
| `step8_pit_comparison_<fct>.csv` | gamlss family AIC comparison per factor |
| `marginal_selection.csv` | Chosen PIT family per rolling window |
| `step10_copula_comparison.csv` | Copula family AIC + CvM per family |
| `copula_gof.csv` | Rolling CvM GoF per annual window |
| `backtest_copula.csv` | Daily VaR, actual loss, exception flag |
| `summary_total_pnl.csv` | P&L component statistics (mean, std, skew, kurt, KS) |
| `summary_instrument_pnl.csv` | IRS + straddle P&L statistics separately |

---

## 6. Six Validation Criteria (C1–C6) Detail

These are evaluated in **Step 7** (full-sample, stored in `results$garch_valid`) and again in **Step 15 Layer A/B** (out-of-sample windows). All six use the standardised residuals Ẑ_t = ε_t / σ_t from the fitted GARCH model.

**C1 — No autocorrelation in Ẑ**
- Test: Weighted Ljung-Box at lag 10 (fitdf = p_AR + q_MA)
- Pass: p > 0.05
- Interpretation: Mean model (ARMA) has removed all serial dependence
- rugarch function: `WeightedPortTest::Weighted.Box.test(Zh, lag=10, type="Ljung-Box", fitdf=p+q)`

**C2 — No remaining ARCH in Ẑ²**
- Tests: Weighted LB on Ẑ² (fitdf = α + β) AND `FinTS::ArchTest(Zh, lags=10)`
- Pass: p > 0.05 in both
- Interpretation: Variance model has captured all volatility clustering
- C2 = `min(LB_sq_p, ArchTest_p) > 0.05`

**C3 — Innovation distribution correct**
- Test: Pearson χ² goodness-of-fit with 4 group sizes {20, 30, 40, 50}
- Pass: mean p-value across 4 groups > 0.05
- Interpretation: The chosen distribution (jsu, nig, sstd, …) fits the standardised residuals
- rugarch function: `gof(fit, groups = c(20, 30, 40, 50))`

**C4 — Variance targeting consistency**
- Test: compare `uncvariance(fit)` (model unconditional variance) to `var(y)` (empirical variance)
- Pass: |ratio − 1| < 0.25 (i.e., deviation < 25 %)
- Interpretation: Long-run risk matches the observed data
- **Known issue:** rugarch's VT for apARCH targets δ-power-space variance, causing systematic underestimation of unconditional variance for SPY and EURUSD. This is not a code bug.

**C5 — No sign bias (leverage effect residual)**
- Test: `rugarch::signbias(fit)` — tests for asymmetric response to positive vs. negative innovations
- Pass: all three p-values (sign bias, positive sign bias, negative sign bias) > 0.05
- Interpretation: The asymmetric GARCH model (GJR/eGARCH/apARCH) has captured all leverage effects
- In practice: C5 frequently fails because real leverage effects are more complex than a single asymmetry parameter can capture

**C6 — Parameter stability (Nyblom)**
- Test: `rugarch::nyblom(fit)` — tests for parameter constancy over time
- Pass: Joint Nyblom statistic < critical value at 5 % level
- Interpretation: GARCH parameters are stable over the estimation period (no structural break)
- Note: In a 19-year sample including the GFC, COVID, and multiple rate cycles, C6 failure is expected for most factors

---

## 7. Reproducibility Instructions

### Full run from scratch

```r
# 0. Download data (one-time, requires internet)
# python src/data/download_data.py

# 1. Compute risk factors
# python src/data/compute_returns.py

# 2. GARCH marginals (takes ~5–10 min for 6 factors × 28 model/dist combos)
setwd("path/to/var_project")
source("src/var_methods/steps_3_to_8_marginal_garch.R")

# 3. Copula + VaR (takes ~3–5 min including rolling simulation)
source("src/var_methods/steps_9_to_12_copula_var.R")

# 4. Post-VaR validation
source("src/var_methods/step15_post_var_validation.R")
```

### Rerun with overrides (without re-running GARCH)

```r
# Keep existing 'results' in workspace from Step 2; only re-run copula layer
manual_copula <- "gaussian"                      # try Gaussian instead of t
source("src/var_methods/steps_9_to_12_copula_var.R")
```

### Speed up Step 6 (development)

```r
# Restrict to 2 model types and 3 distributions
manual_garch_model <- setNames(
  rep(list(c("sGARCH", "gjrGARCH")), 6), factor_names)
# NOTE: re-source Step 3–8 after setting this; the results list is rebuilt from scratch
```

### Key R packages required

| Package | Used for |
|---|---|
| rugarch | GARCH fitting, GoF, sign bias, Nyblom |
| gamlss, gamlss.dist | PIT marginal distribution fitting |
| copula | Copula fitting (`fitCopula`), simulation (`rCopula`) |
| reticulate | Python bridge (portfolio pricing) |
| rvinecopulib | Vine copula (Step 14d — optional, subprocess-isolated) |
| WeightedPortTest | Weighted Ljung-Box test |
| FinTS | ARCH-LM test |
| forecast | auto.arima |
| tseries | adf.test |

Python 3.13 + numpy + pandas + scipy required for `portfolio_pricing.py` (IRS and straddle pricing).  
Set `RETICULATE_PYTHON` env var or `reticulate::use_python()` before sourcing Step 9–12.
