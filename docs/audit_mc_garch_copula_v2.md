# Audit of `monte_carlo_garch_copula_v2_audit.py` against Lecture Notes

**Auditor:** Claude · **Date:** 2026-05-05
**Scope:** validate every step of the v2 Monte‑Carlo / GARCH / t‑copula pipeline against the Irle *Market Risk Modelling* lecture notes (228 pp).
**Reference file:** `src/var_methods/monte_carlo_garch_copula_v2_audit.py` (771 lines).

The audit is organised as (i) a per‑step lecture‑compliance matrix, (ii) findings ranked by severity, (iii) suggested remediations.

---

## 1. Lecture‑compliance matrix

| Pipeline step (v2)                                         | Lecture reference                                               | v2 code location                       | Status        |
|------------------------------------------------------------|-----------------------------------------------------------------|----------------------------------------|---------------|
| Decomposition `Y_t = µ + X_t`, `X_t = σ_t Z_t`             | §8.5, eq. (19)–(20), p. 159–160                                 | `fit_garch_marginals`, ln 168–193     | ✅ aligned    |
| GARCH(1,1): `σ²_t = ω + α X²_{t-1} + β σ²_{t-1}`           | §8.3, p. 141                                                    | ln 608–610                             | ✅ aligned    |
| Stationarity `α + β < 1`                                   | §8.3, Thm 4(ii), p. 143                                         | ln 184–187 (rescale if violated)       | ✅ enforced   |
| Standardised residuals `Ẑ_t = (Y_t − µ̂)/σ̂_t`              | §8.5, Thm 8.2, p. 162                                           | ln 192–193, 604                        | ✅ aligned    |
| One‑step‑ahead `Var(Y_{t+1}|F_t) = ω + α X_t² + β σ_t²`    | §8.5, p. 160                                                    | ln 608–611                             | ✅ aligned    |
| Rolling window of 500 obs.                                 | §8.6 Step 7, p. 182                                             | `WINDOW = 500` (ln 73)                 | ✅ exact match|
| Re‑estimate every 50 obs.                                  | §8.6 Step 7, p. 182                                             | `REFIT_EVERY = 50` (ln 76)             | ✅ exact match|
| Student‑t innovations (return distribution non‑normal)     | §8.6 Step 6, p. 177–180                                         | `dist='t'`, ln 169                     | ✅ aligned    |
| EWMA, `λ = 0.94`                                           | §8.2 (RiskMetrics decay), p. 135                                | `EWMA_LAMBDA = 0.94` (ln 93)           | ✅ aligned    |
| Pseudo‑observations via ranks/(n+1)                        | §7.2 (empirical copula), p. 124                                 | `compute_pseudo_observations`, ln 228  | ✅ aligned    |
| t‑copula `C_{ν,ρ}(u,v) = t_{ν,2}(t_ν^{-1}(u), t_ν^{-1}(v); ρ)` | §7.2 t‑copula, p. 113                                           | `simulate_t_copula`, ln 400–411        | ✅ aligned    |
| Cramér–von‑Mises GoF on copula `S_X = Σ (C_X(U_i)−C_n(U_i))²` | §7.2, p. 124                                                    | `copula_gof_cvm`, ln 308–393           | ✅ aligned    |
| Empirical quantile `VaR ≈ −Q̂_{1−α}`                       | Ch. 6, p. 80                                                    | `extract_var`, ln 486–488              | ✅ aligned    |
| Full revaluation (option position f(t,S,r,σ))              | Ch. 6, p. 81–82                                                 | `scenarios_to_pnl`, ln 462–479         | ✅ aligned    |
| Look‑ahead avoidance — instrument state from `t−1`         | implicit (general statistical hygiene)                          | ln 614–622                             | ✅ aligned    |
| Backtest: N ~ B(T, p)                                      | §8.6 Step 8, p. 184                                             | `backtest.py kupiec_test`              | ✅ aligned    |
| Kupiec POF + Christoffersen tests                          | **NOT in lecture** — beyond §8.6                                | `backtest.py`                          | ➕ extension  |
| EWMA‑driven dynamic correlation `R_t`                      | **NOT in lecture** — extension of static copula                 | `_ewma_corr_update`, ln 248–256        | ➕ extension  |

**Summary:** every numeric formula in v2 that has a lecture counterpart matches the lecture exactly. The two pieces marked "extension" (Kupiec/Christoffersen and EWMA‑driven `R_t`) go beyond the lecture; both are well‑motivated and well‑cited in the codebase.

---

## 2. Findings ranked by severity

### High – material impact on grade or correctness

**H1. Missing GARCH model‑validation tests (lecture §8.5, p. 165–169).**
The lecture devotes seven pages to the criteria a GARCH model must satisfy after calibration:

1. Independence of `Ẑ_t` (Ljung‑Box on standardised residuals).
2. Identical distribution / homoscedasticity (Ljung‑Box on `Ẑ_t²`, Engle ARCH‑LM).
3. Distribution match (Kolmogorov‑Smirnov or Adjusted Pearson GoF on `Ẑ_t`).
4. Modelled vs. empirical unconditional variance (`α₀/(1−α₁−β)` vs sample var).
5. News‑impact symmetry (Engle sign‑bias test).
6. Parameter stability (Nyblom test).

v2 implements **none** of these for the GARCH marginals. It does implement the analogous CvM GoF for the *copula* (good), but not for the six univariate GARCH fits. Per the lecture: *"All these tests are e.g. implemented in the R‑package rugarch"*. The Python equivalent is `arch.unitroot` + `statsmodels.stats.diagnostic` (Ljung‑Box, ARCH‑LM) and `scipy.stats.kstest`. Given the project rubric explicitly asks "Do you validate your assumptions properly, e.g. by means of suitable statistical tests?" (Lecture, Group projects: Grading IV), this is the most consequential gap.

**Suggested fix:** add a `validate_garch_marginals(std_resids, params)` helper that runs Ljung‑Box (`statsmodels.stats.diagnostic.acorr_ljungbox`), Engle's ARCH‑LM (`statsmodels.stats.diagnostic.het_arch`), and KS against the standardised Student‑t with the fitted ν per factor. Print summary on each refit, reject the fit if material violation.

**H2. Variance targeting not implemented (lecture §8.3 p. 148).**
The lecture explicitly discusses variance targeting as the canonical fix when modelled `α₀/(1−α₁−β)` differs materially from the sample variance: *"this step is taken if the modeled and observed unconditional variance deviate materially."* v2 takes whatever `arch_model` returns and never compares the implied unconditional variance to the empirical one — yet the v1‑BUG‑2 narrative (lines 18–24) is *exactly* about ω being stale. Frequent refitting (every 50 days) is one valid response, but variance targeting is the textbook one.

**Suggested fix:** after each refit, check `|σ²_uncond_model − σ²_sample| / σ²_sample`. If above some threshold (e.g. 25 %), set `ω := σ²_sample · (1 − α − β)` (the second variance‑targeting variant on p. 148) before propagating.

### Medium – correctness‑adjacent or stylistic

**M1. Q_ewma initialisation not consistent with the fitted copula (ln 572–580).**
`fit_t_copula` returns an MLE‑estimated correlation matrix `R_static`. Then the EWMA state is bootstrapped from `np.cov(std_resids.T)` (Pearson) — *not* from `R_static`. For low ν the t‑copula MLE differs from Pearson; using Pearson here introduces a small inconsistency between the static fit reported in the refit log and the dynamic `R_dynamic` that actually drives the simulation.

**Suggested fix:** initialise `Q_ewma = R_static` (already a valid correlation matrix). The on‑diagonal stays at 1, so subsequent EWMA updates are unbiased.

**M2. Empirical‑quantile interpolation vs. lecture definition.**
Lecture (Ch. 6, p. 80) defines the empirical quantile at level `(1−α)` as the **`⌊(1−α)·k+1⌋`‑smallest value**. v2 uses `np.percentile` (linear interpolation by default). For `M = 10 000` and α = 99 % the difference is at most one order statistic and immaterial in practice, but the exact lecture definition is `np.partition(pnl_sim, m)[m]` with `m = int((1-α)*M)`. Worth a comment or one‑line change for full lecture fidelity.

**M3. EWMA on a *correlation matrix* is a beyond‑lecture extension (ln 248–256).**
The lecture's EWMA chapter (§8.2) is purely univariate. The v2 extension to a DCC‑style EWMA on `Q_t = λ Q_{t-1} + (1-λ) z_{t-1} z_{t-1}'`, then re‑normalised to a correlation matrix, is *standard* in industry (Engle 2002 DCC), but it should be **flagged as an extension** in the report so the examiner knows you are aware. A two‑line comment at the EWMA helper plus a sentence in the report would close this.

**M4. Floor `max(-q, 0.0)` silently hides negative VaR (ln 488).**
Lecture Eq. (12): `VaRα,Δt = −Q_{1-α}(ΔV)`, with no requirement that the result be non‑negative. For a portfolio with positive drift the 1 %‑quantile of the simulated P&L can occasionally be positive, in which case true VaR is *negative* (the bottom 1 % of outcomes is still a gain). The floor at zero is a pragmatic convention but deviates from the formal lecture definition; it can also mask numerical pathologies (degenerate σ_forecast, all‑gain simulations).

**Suggested fix:** keep the floor for reporting but raise a warning if `q > 0`, and store the raw quantile in the diagnostics output.

### Low – cosmetic / minor

**L1. Dead import.** `RF_RATE` is imported (ln 65) but never used; the straddle uses the simulated DGS10 instead (correct, matches `compute_pnl.py`). Either drop the import or use it as the constant‑rate fallback.

**L2. CvM bootstrap inner seed is fixed at 42 (ln 347).** Acceptable because the parametric copula CDF approximation should be deterministic across bootstrap iterations, but worth documenting why.

**L3. `nu = max(nu, 2.1)` (ln 188).** Floors ν at 2.1 to keep variance finite. For factors that legitimately want ν ≈ 2.5 this is fine; for VIX_ret it has historically been borderline. Consider logging when the floor is hit.

**L4. Refit warm‑up loop discards the previous Q_ewma (ln 575–580).** With `WINDOW = 500` and `λ = 0.94` the contribution of the first residual is `λ^500 ≈ 5e-14`, so the steady‑state EWMA after the replay is effectively a function only of the last ~50 residuals. The asymmetry between the first and subsequent refits is therefore numerically negligible — but conceptually, you could simply keep `Q_ewma` and skip the warm‑up, saving a Python loop.

**L5. Profile MLE grid is integer‑only `[2..20]` (ln 77).** Refining around the best ν with a Brent search would give a smoother log‑likelihood profile in your report. Optional polish.

---

## 3. Lecture extensions worth highlighting in the report

Three things in v2 are *better* than what the lecture prescribes; mention them explicitly so the grader credits them under the rubric criterion *"How much state‑of‑the‑art and how innovative is your solution?"*:

1. **Kupiec + Christoffersen backtests** (`backtest.py`) — the lecture only asks for a binomial confidence interval (p. 184). v2 reports LR_UC, LR_IND and LR_CC with proper χ² p‑values.
2. **t‑copula instead of Gaussian copula** (lecture p. 88 calls Gaussian copula "industry standard"). t‑copula has non‑zero tail dependence — strictly more conservative for joint downside.
3. **DCC‑style time‑varying correlation `R_t`** via EWMA — the lecture only discusses static copulas.

---

## 4. Cross‑file consistency (verified against `compute_pnl.py`, `portfolio_pricing.py`, `backtest.py`, `config.py`)

* Linear‑P&L static inception shares (`shares = V0 · w / initial_prices`) — **matches** `compute_pnl.compute_total_pnl` (line 98 of `compute_pnl.py`). v2's claim on lines 433–438 is correct.
* IRS pricing — `_price_irs_prod` is `portfolio_pricing.price_irs`, value = `notional · (swap_rate − fixed_rate) · annuity` (par‑rate proxy). Identical formula on both sides.
* Straddle Greeks — `price_straddle_vec` (ln 100–105) is the call+put closed form; `portfolio_pricing.price_straddle` returns the same value plus Greeks for delta‑gamma analysis. **Numerically identical** for the price.
* Straddle state — `build_straddle_state` is shared, so `K`, `T_now` agree with the back‑test exactly.
* Backtest — `run_backtest(..., confidence=ALPHA, method_name="MC-GARCH-Copula-v2")` correctly uses the shared framework. The `BacktestResult` object exposes `lr_uc`, `lr_ind`, `lr_cc`, `pvalue_*`, `n00..n11`, all consumed by the `print` block on lines 760–767.

No cross‑file inconsistency was found.

---

## 5. Verdict

The v2 file is **structurally sound and largely faithful to the lecture**. Every formula that has a lecture counterpart matches; the two beyond‑lecture features (Kupiec/Christoffersen, EWMA correlation) are well‑motivated and improve the model. The main gap relative to the lecture is **Section 8.5 model‑validation tests for the GARCH marginals** (finding H1) and the **variance‑targeting safeguard** (finding H2). Closing those two gaps would put the implementation at full lecture compliance plus three credible extensions.

Suggested next actions, in priority order:

1. Add `validate_garch_marginals` (Ljung‑Box, ARCH‑LM, KS) and run it on each refit — addresses H1.
2. Add a variance‑targeting check + optional ω adjustment — addresses H2.
3. Initialise `Q_ewma` from `R_static` for internal consistency — M1.
4. Optionally tighten the empirical‑quantile call to match the lecture's `⌊(1−α)k+1⌋`‑smallest definition — M2.
5. Add a one‑paragraph "model extensions beyond the lecture" subsection to the project report (Kupiec/Christoffersen + DCC + t‑copula).

---

*All page numbers refer to the version of "Market Risk Modelling, Dr. Sebastian Irle, Summer 2026" supplied as `1 Lecture Notes.pdf` (228 pp.).*
