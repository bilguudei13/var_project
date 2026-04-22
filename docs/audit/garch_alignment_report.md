# GARCH Alignment Audit Report

**Date:** 2026-04-22  
**Auditor:** Claude Sonnet 4.6 (automated)  
**Files scanned:** `src/var_methods/GARCH R.r`, `src/var_methods/GARCH.py`, `src/var_methods/garch_evt.py`  
**Files modified:** `src/var_methods/garch_evt.py` (ALIGN-001 only)  
**Protected (read-only):** `GARCH R.r`, `GARCH.py` — mtime and size verified unchanged before and after audit.

---

## 1. Classification of `GARCH R.r`

**Role: Separate deliverable** — a comprehensive per-factor GARCH diagnostic study.

Evidence:
- Operates on `data/processed/risk_factors.csv` (6 individual factors: SPY, DGS10, GLD, EURUSD, SPY_level, VIX), **not** the portfolio P&L used by `garch_evt.py`.
- Pipeline: ADF stationarity → auto-ARMA → ARCH test → marginal distribution selection (9 candidate families) → GARCH fitting → 6 validation criteria (Ljung-Box, GoF, sign bias, Nyblom). No EVT, no VaR, no backtesting.
- Uses `rugarch` (R), while both Python files use the `arch` package.
- No shared output that `garch_evt.py` is meant to numerically replicate.

**Scope overlap with `garch_evt.py`:** Low. The R file establishes that GARCH(1,1) with Student-t innovations is appropriate for these financial series (Step 7), and `garch_evt.py` inherits that choice. No point-by-point numerical match is expected or possible (different portfolios, different packages, different time periods).

**Classification consequence:** For alignment purposes, `GARCH.py` is the **sole authoritative** Python reference. `GARCH R.r` informs background methodology choices (Student-t choice, sGARCH(1,1) default) but its R-specific choices (ARMA mean pre-filtering, `rugarch` solver, per-factor scope) do not propagate to `garch_evt.py`.

---

## 2. Methodology Comparison

| Aspect | `GARCH R.r` | `GARCH.py` | `garch_evt.py` | Aligned? |
|---|---|---|---|---|
| GARCH order (p, q) | (1,1) default; configurable via `manual_garch_order` | (1,1) | (1,1) | **Yes** |
| Mean specification | ARMA(p_arma, q_arma) + `include.mean=TRUE` (auto-ARIMA order, lines 77–80) | `Constant` (arch default, implicit) | `Constant` (explicit) | N/A (R scope differs) |
| Innovation distribution | `std` (Student-t, rugarch code); configurable via `manual_garch_dist` | `t` (Student-t, arch) | `t` (Student-t, arch) | **Yes** |
| Variance targeting vs free ω | Free ω (ugarchspec default) | Free ω (arch default) | Free ω (arch default) | **Yes** |
| Estimation method | MLE (ugarchfit default, solver="hybrid") | MLE (arch default) | MLE (arch default) | **Yes** |
| Input series | `risk_factors.csv` — per-factor log returns | `risk_factors.csv` — per-factor log returns | `total_portfolio_pnl.csv` → `pnl/V0` | N/A (different scope by design) |
| Scaling convention (×100) | No explicit scaling (rugarch handles internally) | **Conditional:** `scale_factor=100 if std<0.1 else 1` (lines 89–91) | Previously **unconditional:** always ×100; **→ fixed by ALIGN-001** | **Yes (post-alignment)** |
| Initial variance | rugarch long-run default | arch default | arch default | **Yes** |
| Parameter constraints (α+β<1) | Implicit via rugarch stationarity enforcement | Implicit via arch | Checked and printed post-fit (Irle p. 176 citation) | **Yes** |
| Refit frequency | Full-sample, once (diagnostic tool) | Full-sample, once (diagnostic tool) | Expanding window, every 50 days (B1, look-ahead-bias prevention) | N/A (expanding window is garch_evt.py's own architectural requirement for backtesting; GARCH.py is not a backtesting tool) |
| Residual extraction | `residuals(fit_g, standardize=TRUE)` from rugarch | Not extracted | `z[t-1] = (r[t-1] − mu) / sigma[t-1]` with B10 refit-boundary fix | N/A |
| Forecast horizon | One-step via `sigma(fit_g)` | Not computed | One-step via `forecast(horizon=1)` | **Yes** |
| Forecast target | σ_t (current period conditional vol) | Not computed | σ_{t+1} one-step-ahead | **Yes** |
| Backtesting targets | Not done | Not done | Total portfolio P&L (pnl_total) | N/A |
| Window length for EVT rolling | N/A | N/A | 500 days | N/A |
| Confidence level α | N/A | N/A | 99% | N/A |
| Tail modelling on residuals | None | None | EVT/POT (Irle Section 9) | N/A |

**"No" row count (pre-alignment):** 1 (scaling convention).  
**"No" row count (post-alignment):** 0.

---

## 3. Assumption Comparison

| Assumption | `GARCH R.r` | `GARCH.py` | `garch_evt.py` | Status |
|---|---|---|---|---|
| A1: Returns have fat tails (Student-t appropriate) | Yes — Step 6 distribution selection chooses `std`/`SST`/etc. by AIC | Yes — explicit `dist='t'`; checks excess kurtosis | Yes — Student-t GARCH + EVT on residuals | Aligned |
| A2: GARCH(1,1) sufficient for variance dynamics | Yes — Step 7 default `c(1,1)` | Yes — hardcoded `p=1, q=1` | Yes — hardcoded `p=1, q=1` | Aligned |
| A3: Stationarity (α+β < 1) | Implicit via rugarch | Implicit via arch | Explicit check + print | Aligned |
| A4: IID standardised residuals after GARCH filter | Yes — LB test on Ẑ_t (C1), ARCH test on Ẑ²_t (C2) | Not tested (diagnostic-only tool) | Assumed; z-mean/std printed | Aligned |
| A5: Constant mean (Constant ARMA(0,0)) | **No** — ARMA(p,q) mean (auto-selected) | Yes — Constant (arch default) | Yes — `mean="Constant"` | teammate-only (R is separate deliverable; Python files aligned) |
| A6: Full-sample GARCH suffices for VaR | Yes (diagnostic) | Yes (diagnostic) | **No** — expanding window avoids look-ahead | garch_evt.py-only (justified: B1, backtesting requirement) |
| A7: %-scaling needed for arch numerical stability | N/A (rugarch handles internally) | Yes — `scale_factor=100 if std<0.1` | Yes (post ALIGN-001) | Aligned (post-alignment) |
| B1: GPD tail shape ξ ∈ [−0.5, 1.0] | N/A (no EVT) | N/A (no EVT) | Yes — B6 clamping | garch_evt.py-only (supervisor extension, justified) |
| B2: EVT POT threshold at 90th percentile | N/A | N/A | Yes — `THRESHOLD_Q=0.90` | garch_evt.py-only (EVT-specific, not GARCH methodology) |
| B3: VaR floor at u_z (regulatory conservative) | N/A | N/A | Yes — B9 regulatory floor | garch_evt.py-only (supervisor extension, justified) |

---

## 4. Numerical Cross-Check

**R execution:** Unavailable — static analysis only for `GARCH R.r`. No R runtime detected on this machine.

**Python numerical comparison** (`GARCH.py` conditional logic vs `garch_evt.py` pre-alignment, both on portfolio returns at `t=500`):

| Metric | `GARCH.py` logic | `garch_evt.py` (pre-align) | Abs diff | Within tolerance |
|---|---|---|---|---|
| `mu` (%-units) | 0.067896 | 0.067896 | 0.00e+00 | Yes |
| `omega` (%-units) | 0.011042 | 0.011042 | 0.00e+00 | Yes |
| `alpha[1]` | 0.077206 | 0.077206 | 0.00e+00 | Yes |
| `beta[1]` | 0.898634 | 0.898634 | 0.00e+00 | Yes |
| `nu` (dof) | 6.8511 | 6.8511 | 0.00e+00 | Yes |
| Log-likelihood | −441.9956 | −441.9956 | 0.00e+00 | Yes |
| Cond. vol max abs diff (decimal) | — | — | 0.00e+00 | Yes |

**Reason for exact equality:** Portfolio returns `r_p = pnl/V0` have `std = 0.01123 < 0.1`, so `GARCH.py`'s conditional `scale_factor=100` is triggered — producing identical inputs to the optimizer as `garch_evt.py`'s unconditional ×100.

**Post-alignment numerical match:** Yes — identical to machine precision (verified above). Max deviation of conditional volatility series: 0.00e+00. No VaR deviation.

Full numerical comparison table: [`docs/audit/garch_numerical_comparison.csv`](garch_numerical_comparison.csv)

**Static comparison of `GARCH R.r`:**
- R uses `rugarch`'s Student-t (`std`), which is parameterised as ν degrees-of-freedom — same distributional family as `arch`'s `t`. Not directly numerically comparable without R runtime.
- R fits on individual factors; Python garch_evt.py fits on portfolio P&L. Different inputs → different parameter values by design.
- R's mean model includes ARMA terms (auto-selected), which changes the effective innovation series. This is a R-file-specific choice not present in `GARCH.py`; not propagated to `garch_evt.py` per the authority rule.

---

## 5. Code Structure and Documentation

### Function decomposition
- **GARCH.py:** Two functions — `plot_stylised_facts()` (EDA) and `run_garch_pipeline()` (fit + print). No residual extraction, no forecast, no VaR.
- **garch_evt.py:** Clean decomposition: `load_data()` → `fit_garch_expanding()` → `plot_garch_diagnostics()` → `compute_garch_evt_var()` → `backtest_garch_evt()` → `plot_garch_evt_results()` → `save_results()`. More modular than GARCH.py, appropriate given the additional scope.

### Parameter naming
- GARCH.py does not explicitly name individual GARCH parameters (just calls `res.summary()`).
- garch_evt.py uses `mu`, `omega`, `alpha1_d`, `beta_d` — natural names consistent with the arch library's `params['mu']`, `params['omega']`, `params['alpha[1]']`, `params['beta[1]']`. No conflict.
- R file uses rugarch's internal naming. No Python cross-file conflict.

### Docstring and comment style
- GARCH.py uses plain triple-quoted strings (no formal NumPy/Google style).
- garch_evt.py uses an extended NumPy-ish style with Parameters/Returns sections.
- garch_evt.py is more thorough — consistent with its larger scope. **No alignment change needed.**

### Citations
- GARCH.py cites no references.
- garch_evt.py cites Irle (pp. 172–186, 212–225) and McNeil & Frey (2000). Both are appropriate given garch_evt.py's GARCH+EVT scope. **No changes needed.**
- R file cites no references externally (self-contained diagnostic script).

### Error handling
- GARCH.py: no explicit convergence handling (`res = model.fit(disp='off')` — uncaught).
- garch_evt.py: uses `warnings.catch_warnings()` during fit; GPD fit failures caught in `try/except` (B5); GARCH fit errors handled implicitly by arch (warnings suppressed).
- garch_evt.py's error handling is more robust than GARCH.py's.

### Output file conventions
- GARCH.py writes to `report/figures/` (`REPORT_DIR`).
- GARCH R.r writes to `outputs/figures/` and `outputs/tables/`.
- garch_evt.py writes to `outputs/figures/` and `outputs/tables/` — **aligned with R convention**.
- The GARCH.py→`report/figures/` divergence is a pre-existing inconsistency internal to GARCH.py; it does not affect garch_evt.py.

---

## 6. Action List

### Applied actions

| ID | Category | Finding | Teammate's choice | garch_evt.py (pre-align) | Change | Risk | Status |
|---|---|---|---|---|---|---|---|
| ALIGN-001 | Methodology (numerical stability) | Scaling convention for arch optimizer: GARCH.py uses conditional `scale_factor = 100 if series.std() < 0.1 else 1`; garch_evt.py hardcoded unconditional ×100 | GARCH.py lines 89–91: `scale_factor = 1.0; if series.std() < 0.1: scale_factor = 100.0` | `train = r_p.iloc[:t] * 100.0` (and corresponding hardcoded /100, /10000 in parameter extraction) | Replace hardcoded `100.0`/`10000.0` with `scale_factor`/`scale_factor2` computed once from `r_p.std()`. See `fit_garch_expanding()` lines 211–231 (post-alignment). | **None** — `r_p.std() = 0.0112 << 0.1` so `scale_factor=100` in all realistic scenarios; verified numerically exact match. | `ready_to_apply` — applied |

### Deferred actions (needs_review)
None.

---

## 7. Supervisor Extensions Preserved

The following four elements are **supervisor-driven extensions** beyond the teammate's GARCH methodology scope. They are intentionally absent from `GARCH.py` and `GARCH R.r`. They are **not** subject to the authority rule and must be retained regardless.

| Extension | Location in garch_evt.py | Rationale |
|---|---|---|
| **Regulatory floor split (B9)** | `_pot_var_residuals()`: `q_evt_floored = max(q_evt_raw, u_z)`; `VaR_GARCH_EVT_raw` column | Conservative regulatory choice: for ξ < 0 (bounded tail), pure GPD can yield q_EVT < u_z, but policy output is floored. Separating raw vs floored preserves the statistical estimator for diagnostics. |
| **KS GoF diagnostic (B8)** | `_pot_var_residuals()`: `kstest(exceedances, "genpareto", ...)`; `ks_pvalue` returned and stored | Relative quality indicator for GPD fits. Anti-conservative (params estimated from same data) but useful for detecting grossly misspecified fits. |
| **Refit-boundary residual consistency fix (B10)** | `fit_garch_expanding()`: `mu_hat_prev` capture before refit; `z_arr[t-1]` filled using `mu_hat_prev` not updated `mu_hat` | Prevents cross-regime contamination at refit boundaries: z_arr[t-1] must use the same parameter regime that produced cond_vol_arr[t-1]. |
| **ξ clamping to [−0.5, 1.0] (B6)** | `_pot_var_residuals()`: `if xi > 1.0: xi = 1.0` / `elif xi < -0.5: xi = -0.5` | Financially plausible range: ξ > 1 implies infinite mean; ξ < −0.5 implies implausibly hard bounded tail. Clamping prevents degenerate quantile estimates. |

---

## 8. Self-Audit

| Check | Result |
|---|---|
| `GARCH R.r` mtime/size unchanged | ✅ Pass — mtime=1776840216, size=16179 (verified before and after) |
| `GARCH.py` mtime/size unchanged | ✅ Pass — mtime=1776840216, size=4183 (verified before and after) |
| garch_evt.py edits limited to `ready_to_apply` actions | ✅ Pass — only ALIGN-001 applied |
| Every "No" row in Step 1 has a corresponding action | ✅ Pass — 1 "No" row → ALIGN-001 |
| All four supervisor extensions still present | ✅ Pass — B6, B8, B9, B10 verified by token search |
| Syntax check | ✅ Pass — `ast.parse()` successful |
| Post-alignment numerical match | ✅ Pass — max deviation = 0.00e+00 (exact match) |

### Follow-up items for the user

1. **GARCH.py output directory inconsistency:** `GARCH.py` writes to `report/figures/` while all other files (R and garch_evt.py) use `outputs/figures/`. Low priority — GARCH.py is a diagnostic tool, not a pipeline output. If standardization is desired, update `GARCH.py` line 18: `REPORT_DIR = "outputs/figures"`. This is outside this audit's scope (GARCH.py is read-only here).

2. **GARCH.py mean model:** GARCH.py does not explicitly pass `mean="Constant"` to `arch_model()`. The default in the `arch` library is `'Constant'`, so behaviour is correct, but explicit is clearer. No alignment action required for garch_evt.py (it is already explicit).

3. **R runtime:** `GARCH R.r` was not executed — static analysis only. If R/rugarch is available in a future environment, run the R file and cross-check factor-level GARCH parameters against `GARCH.py` per-factor outputs for a full numerical audit.

4. **GARCH.py has no convergence guard:** `model.fit(disp='off')` is called without `try/except`. If a factor series causes optimizer failure, the script crashes. Not a garch_evt.py concern, but worth noting for the teammate.
