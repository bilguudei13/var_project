# Market Risk Modelling — Portfolio VaR Project

End-to-end Value-at-Risk study for a multi-instrument portfolio built
from SPY, GLD, EURUSD, an interest-rate swap on DGS10, and an SPY
straddle. Five VaR methods are implemented, backtested, and compared
at the 1-day 99% level: **Delta-Normal**, **Historical Simulation**,
**EVT (POT)**, **GARCH-EVT**, **Monte Carlo**, and a **GARCH-Copula**
model.

This README is the runbook reviewers should follow to reproduce the
results in `outputs/` and the notes in `report/`.

---

## 1. Repository structure

```
var_project/
├── src/
│   ├── data/                       # Data pipeline (download + transforms)
│   │   ├── config.py               # Tickers, dates, output dirs
│   │   ├── download_data.py        # Yahoo Finance + FRED
│   │   ├── compute_returns.py      # Log returns and factor returns
│   │   ├── compute_pnl.py          # Instrument and total portfolio P&L
│   │   ├── portfolio_pricing.py    # IRS / straddle re-pricing helpers
│   │   └── run_pipeline.py         # Runs download → returns → P&L in order
│   └── var_methods/                # One module per VaR method
│       ├── delta_normal.py
│       ├── historical_sim.py
│       ├── historical_sim_analysis.py
│       ├── evt.py
│       ├── GARCH.py
│       ├── garch_evt.py
│       ├── mc_gaussian.py             # canonical Monte Carlo (Gaussian, full reval)
│       ├── mc_t_copula.py
│       ├── mc_garch_t_copula.py
│       └── GARCH-Model.R                     # R: marginal GARCH, vine copula, VaR
├── backtesting/
│   ├── backtest.py                 # Kupiec / Christoffersen / binomial
│   └── plot_backtest.py            # Exception-timeline plots
├── notebooks/
│   ├── exploratory_analysis.ipynb  # Data exploration (read-only)
│   └── garch_analysis.ipynb        # GARCH diagnostics
├── data/
│   ├── raw/                        # Downloaded prices, vix, dgs10
│   └── processed/                  # Returns, factor matrix, P&L
├── outputs/
│   ├── tables/                     # All CSVs cited in the report
│   └── figures/                    # PNGs used in slides / docx
├── report/
│   ├── overall_var_output_notes.md # Comparison of methods (matches CSVs)
│   ├── historical_sim_interpretation.md
│   ├── theoretical_background.md
│   └── …                           # Word/Markdown explainers per method
├── tests/                          # pytest — HistSim core + analysis
├── renv.lock                       # R lockfile (see caveat below)
└── requirements.txt                # Python pin set
```

`output/` (singular) holds audit PDFs only and is **not** the
production output directory. All productive outputs live in
`outputs/` (plural).

---

## 2. Python setup

Python 3.11+ is required (developed on 3.13).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` is a flat pin set. See
[§9 Known limitations](#9-known-limitations-and-interpretation-caveats)
for outstanding `pip-audit` advisories.

---

## 3. R setup and the renv caveat

The canonical GARCH-Copula stack is R-based and lives in
`src/var_methods/GARCH-Model.R`. Use R 4.3.x for reproducibility. A
clean setup should activate `renv`, install the required package set,
and only then snapshot the lockfile:

```r
renv::activate()
install.packages(c(
  "xts","zoo","moments","tseries","forecast","FinTS",
  "rugarch","gamlss","gamlss.dist","gamlss.add","copula",
  "WeightedPortTest","reticulate","rvinecopulib","depmixS4"
))
renv::snapshot()
```

> **renv caveat.** `renv.lock` in this repository currently pins only
> the `renv` package itself. It does **not** capture full versions of
> `rugarch`, `copula`, `gamlss`, `rvinecopulib`, `reticulate`, etc.
> Reproducing the exact R numbers therefore depends on CRAN versions
> available at install time. The R outputs committed under
> `outputs/tables/` were produced on R 4.3.0 with recent CRAN versions
> of the packages above. Treat the R-side reproducibility as approximate
> until the lockfile is regenerated with `renv::snapshot()` from a clean
> R 4.3.x session. The lockfile was not regenerated from local R 4.1.2
> because that would under-specify or downgrade the project environment.

---

## 4. Data pipeline (run order)

From the repo root:

```bash
python src/data/run_pipeline.py
```

That script runs, in order:

1. `src/data/download_data.py` — fetches SPY/GLD/EURUSD prices,
   VIX, and DGS10 (constant-maturity 10y) into `data/raw/`.
2. `src/data/compute_returns.py` — writes
   `data/processed/log_returns.csv`,
   `data/processed/portfolio_returns.csv`, and the 6-factor matrix
   `data/processed/all_factor_returns.csv` used by Monte Carlo.
3. `src/data/compute_pnl.py` — prices the IRS and SPY straddle each
   day and writes `data/processed/instrument_pnl.csv` and
   `data/processed/total_portfolio_pnl.csv`.

Re-pricing details (clean-price IRS, Black-Scholes straddle on
SPY×strike×tenor×DGS10×VIX) live in `src/data/portfolio_pricing.py`.

---

## 5. VaR method run order

Each VaR method is a standalone script that consumes `data/processed/`
and writes to `outputs/tables/` and `outputs/figures/`. They can be
run independently:

```bash
python src/var_methods/delta_normal.py
python src/var_methods/historical_sim.py
python src/var_methods/historical_sim_analysis.py    # diagnostics on top
python src/var_methods/evt.py
python src/var_methods/GARCH.py
python src/var_methods/garch_evt.py
python src/var_methods/mc_gaussian.py    # canonical Monte Carlo (var_mc.csv, backtest_mc.csv)
```

`src/var_methods/GARCH.py` is the standalone GARCH diagnostic/model
script. It is separate from `src/var_methods/garch_evt.py`, which adds
EVT tail modelling on top and produces the headline GARCH-EVT VaR
backtest.

The GARCH-Copula model is R-based:

```r
source("src/var_methods/GARCH-Model.R")
```

Legacy Python wrapper outputs are still committed for audit continuity
(`data/processed/var_copula.csv`,
`outputs/tables/backtest_copula.csv`, and `outputs/figures/var_copula_*`),
but the dead wrapper source has been removed. Do not regenerate the
copula path from Python; use `GARCH-Model.R` for new copula work.

---

## 6. Backtesting outputs

Backtests are driven by `backtesting/backtest.py` (Kupiec LR,
Christoffersen independence + conditional coverage, lecture-style
exact binomial threshold). One CSV per method:

| File | Method |
|------|--------|
| `outputs/tables/backtest_historical_sim.csv` | HistSim |
| `outputs/tables/backtest_summary_historical_sim.csv` | HistSim binomial summary |
| `outputs/tables/backtest_evt.csv` | EVT (POT) |
| `outputs/tables/backtest_garch_evt.csv` | GARCH-EVT |
| `outputs/tables/backtest_mc.csv` | Monte Carlo |
| `outputs/tables/backtest_copula.csv` | GARCH-Copula legacy Python wrapper output |

The headline 99% VaR exception counts at the 1-day 99% level are:

| Method      | Window (days) | Exceptions | Expected | Rate    |
|-------------|---------------|------------|----------|---------|
| HistSim     | 4,269         | 42         | 42.7     | 0.98 %  |
| EVT         | 4,269         | 65         | 42.7     | 1.52 %  |
| GARCH-EVT   | 4,269         | 47         | 42.7     | 1.10 %  |
| Monte Carlo | 4,019         | 64         | 40.2     | 1.59 %  |
| Copula      | 4,269         | 188        | 42.7     | 4.40 %  |

These numbers are reproducible directly from the CSVs above; see
`report/overall_var_output_notes.md` for the matching narrative and
Kupiec / Christoffersen p-values.

Cross-method notes:

* EVT, GARCH-EVT, Monte Carlo, and the legacy Copula output use
  `-pnl_total` from `data/processed/total_portfolio_pnl.csv` on their
  backtest dates.
* HistSim uses the same economic realised-loss basis and headline range
  (max ≈ $85.5k), but reconstructs portfolio state internally. It
  differs from canonical `-pnl_total` on 156 / 4,269 dates by up to
  about $5.3k, mostly around straddle roll-over mechanics.
* Monte Carlo uses a 750-day rolling estimation window (vs HistSim's
  500), which is why it has 250 fewer backtest observations. Within its
  own window the realised loss column equals `-pnl_total` from
  `data/processed/total_portfolio_pnl.csv` to the cent — the simulated
  side is now anchored on the same fixed-share / current-price basis as
  `compute_pnl.py` (see §9 for the May 2026 fix history).
* PR #28 also expanded the repository with additional methods that are
  not headlined here (Delta-Normal, Delta-Normal-Linear, FHS,
  Vol-Adjusted HistSim, and several MC + GARCH-Copula variants); their
  CSVs live alongside the five above under
  `outputs/tables/backtest_*.csv`.

---

## 7. Reproducing key tables and figures

| Artefact | Produced by |
|----------|-------------|
| HistSim VaR series + figures 11/13/23/24 | `src/var_methods/historical_sim.py`, `historical_sim_analysis.py` |
| HistSim window-sensitivity table | `outputs/tables/histsim_window_sensitivity.csv` |
| EVT POT fit and CSV | `src/var_methods/evt.py` |
| Standalone GARCH diagnostics/model output | `src/var_methods/GARCH.py` |
| GARCH-EVT VaR + CSV | `src/var_methods/garch_evt.py` |
| Monte Carlo VaR + figure 07 | `src/var_methods/mc_gaussian.py` |
| Copula GoF + simulated VaR | `src/var_methods/GARCH-Model.R`; legacy Python-wrapper CSVs retained for audit continuity |
| All backtest CSVs | `backtesting/backtest.py` (called from each method script) |

Outputs are deterministic given a fixed Python/R version and a fixed
random seed where applicable. Monte Carlo and Copula simulations
use seeded RNGs inside their respective scripts.

> `notebooks/exploratory_analysis.ipynb` is diagnostic and may
> rewrite tracked EDA figures/tables under `outputs/`; it is not
> part of the mandatory clean reproduction path.

---

## 8. Verification commands

From the repo root, with the Python environment activated:

```bash
# Core unit tests (HistSim and HistSim analysis)
pytest tests/test_historical_sim.py tests/test_historical_sim_analysis.py -q

# Path-portability guard (no hardcoded user paths in productive files)
pytest tests/test_no_hardcoded_paths.py -q

# Style / dependency advisories (informational; see §9)
ruff check .
pip-audit -r requirements.txt --no-deps
```

Expected on a clean checkout: pytest passes; ruff and pip-audit
report the known follow-ups documented below.

---

## 9. Known limitations and interpretation caveats

These are intentionally **not** addressed in the final-submission
branch and are captured here for the reviewers.

### Methodological

- **Monte Carlo basis fix (May 2026).** The MC linear-leg basis is now
  aligned with `compute_pnl.py`: the simulated linear P&L is built from
  fixed inception share counts re-marked at the current rolling prices,
  matching the realised P&L convention. The pre-fix V0-based formula
  (`DV_linear = V0 · w' · sim_returns`) anchored the linear leg to the
  initial $1,000,000 notional regardless of SPY drift, producing a mean
  MC VaR ($16.5k) roughly half the HistSim mean VaR ($33.1k) and 320
  exceptions / 7.50 % on the legacy 4,269-day output. After the fix,
  on the 4,019-day window the canonical MC source produces 64
  exceptions / 1.59 % (mean VaR $27.8k, mean HistSim VaR $33.6k —
  ratio 0.83). Unconditional coverage is still rejected by Kupiec
  (p = 0.0005), and Christoffersen independence remains rejected, but
  this is now a genuine model finding: Gaussian MC understates fat
  tails and ignores volatility clustering, which a Gaussian-copula
  full-revaluation model is expected to do. The basis pathology
  itself is gone. Guard tests live in
  `tests/test_mc_backtest_basis.py`.
- **Legacy Copula 4.40 % exception rate (188 in 4,269 days).**
  The committed `backtest_copula.csv` and `var_copula_*` figures came
  from the deleted Python wrapper. That wrapper independently
  bootstrapped the nonlinear P&L window: a separate random index selected
  nonlinear losses, breaking the joint dependence between linear and
  nonlinear legs that the copula is supposed to capture. The artefacts
  are retained only as labelled legacy outputs; new copula work should
  use the R pipeline in `src/var_methods/GARCH-Model.R`.
- **HistSim and Copula fail Christoffersen independence.**
  HistSim has good unconditional coverage but exceptions cluster.
  Copula now over-exceeds on both unconditional and independence
  tests — see the per-method notes in
  `report/overall_var_output_notes.md`.
- **Realised-loss basis is mostly harmonised, with a documented HistSim
  residual.** EVT, GARCH-EVT, Monte Carlo, and legacy Copula are
  bit-identical to canonical `-pnl_total` on their dates. HistSim shares
  the same headline loss range and agrees on about 96% of dates, but
  reconstructs portfolio state internally and differs on 156 dates by up
  to about $5.3k. Treat this as a documented residual comparability
  caveat rather than the older PR #28 basis mismatch.

### Reproducibility / tooling

- **R `renv.lock` only pins `renv` itself.** Numerical R outputs
  therefore depend on CRAN at install time. See §3.
- **`pip-audit` reports 22 CVEs** across 9 packages (mostly the
  Jupyter / notebook stack plus `pillow`, `lxml`, `urllib3`,
  `mistune`, `pygments`, `curl-cffi`). None affect the productive
  numeric code path; they are notebook/visualisation transitive
  dependencies. Upgrade in a separate hardening pass.
- **`ruff check .` reports 244 style findings** (mostly
  `E402` import order, `F541` empty f-strings, `E702` semicolons,
  `F401` unused imports) on the post-merge codebase. They are
  non-functional; deferred to a dedicated cleanup PR so this
  submission does not churn diffs.

---

## 10. Audit history

- Audit reports prior to this submission live in `output/pdf/`.
- The last audit baseline commit referenced by those PDFs is
  `182e296`. The current branch
  (`cornelius/finalize-baseline-histsim`) may already address
  findings beyond what those PDFs describe; always treat this README
  and the CSVs in `outputs/tables/` as authoritative.
