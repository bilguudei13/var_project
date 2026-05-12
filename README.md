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
│       ├── monte_carlo.py
│       ├── var_copula.py
│       ├── steps_3_to_8_marginal_garch.R     # R: marginal GARCH (copula input)
│       └── steps_9_to_12_copula_var.R        # R: copula fit + simulated VaR
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

The copula stack runs in R (`src/var_methods/steps_3_to_8_marginal_garch.R`
and `steps_9_to_12_copula_var.R`). A typical install:

```r
install.packages(c(
  "xts","zoo","moments","tseries","forecast","FinTS",
  "rugarch","gamlss","gamlss.dist","gamlss.add","copula",
  "WeightedPortTest"
))
```

> **renv caveat.** `renv.lock` in this repository currently pins only
> the `renv` package itself. It does **not** capture full versions of
> `rugarch`, `copula`, `gamlss`, etc. Reproducing the exact R numbers
> therefore depends on CRAN versions available at install time. The R
> outputs committed under `outputs/tables/` were produced on R 4.3.0
> with recent CRAN versions of the packages above. Treat the R-side
> reproducibility as approximate until the lockfile is regenerated
> with `renv::snapshot()` from a clean session.

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
python src/var_methods/monte_carlo.py
```

The GARCH-Copula model is R-based:

```r
source("src/var_methods/steps_3_to_8_marginal_garch.R")   # Steps 3–8
source("src/var_methods/steps_9_to_12_copula_var.R")      # Steps 9–12
```

The Python wrapper `src/var_methods/var_copula.py` consumes the R
output to compute portfolio-level VaR and write
`outputs/tables/backtest_copula.csv`.

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
| `outputs/tables/backtest_copula.csv` | GARCH-Copula |

The headline 99% VaR exception counts over the **4,269-day backtest
window** (expected ≈ 42.7) are:

| Method      | Exceptions | Rate    |
|-------------|------------|---------|
| HistSim     | 42         | 0.98 %  |
| EVT         | 65         | 1.52 %  |
| GARCH-EVT   | 47         | 1.10 %  |
| Monte Carlo | 320        | 7.50 %  |
| Copula      | 188        | 4.40 %  |

These numbers are reproducible directly from the CSVs above; see
`report/overall_var_output_notes.md` for the matching narrative and
Kupiec / Christoffersen p-values. After PR #28, all five method
backtests use the **same** realised loss series (max ≈ $85.5k), so
cross-method comparison is now apples-to-apples. PR #28 also expanded
the repository with additional methods that are not headlined here
(Delta-Normal, Delta-Normal-Linear, FHS, Vol-Adjusted HistSim, and
several MC + GARCH-Copula variants); their CSVs live alongside the
five above under `outputs/tables/backtest_*.csv`.

---

## 7. Reproducing key tables and figures

| Artefact | Produced by |
|----------|-------------|
| HistSim VaR series + figures 11/13/23/24 | `src/var_methods/historical_sim.py`, `historical_sim_analysis.py` |
| HistSim window-sensitivity table | `outputs/tables/histsim_window_sensitivity.csv` |
| EVT POT fit and CSV | `src/var_methods/evt.py` |
| GARCH-EVT VaR + CSV | `src/var_methods/garch_evt.py` |
| Monte Carlo VaR + figure 07 | `src/var_methods/monte_carlo.py` |
| Copula GoF + simulated VaR | R scripts above + `var_copula.py` |
| All backtest CSVs | `backtesting/backtest.py` (called from each method script) |

Outputs are deterministic given a fixed Python/R version and a fixed
random seed where applicable. Monte Carlo and Copula simulations
use seeded RNGs inside their respective scripts.

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

- **Monte Carlo 7.5 % exception rate** is far higher than the 1 %
  target (320 exceptions in 4,269 days vs an expected ≈ 43).
  Root cause appears to be a linear-book basis mismatch:
  `src/var_methods/monte_carlo.py` scales the simulated linear P&L
  by the initial portfolio notional `V0` (constant `$1,000,000` from
  `config.py`), whereas the realised P&L in
  `data/processed/total_portfolio_pnl.csv` is built from **fixed
  share counts** and therefore scales with the drifted current
  portfolio value. Over the 2007-2024 backtest, SPY appreciates
  substantially, so realised dollar losses grow while MC VaR remains
  anchored to the initial scale — the MC mean VaR ($16.5k) ends up
  roughly half the HistSim mean VaR ($33.1k) on the same realised
  loss series. **Treat MC as a diagnostic output, not a fully
  comparable method ranking, until the linear leg is rescaled with
  the rolling current portfolio value and the dependent outputs are
  regenerated.** Fixing it should be a single follow-up PR.
- **Copula 4.40 % exception rate (188 in 4,269 days).**
  `src/var_methods/var_copula.py` independently bootstraps the
  nonlinear P&L window: a separate `rng.integers` call indexes into
  `nonlinear_window`, which breaks the joint dependence between
  linear and nonlinear legs that the copula is supposed to capture.
  The assumption is disclosed in the function docstring. It is left
  as-is in this submission because changing it would invalidate all
  currently committed copula tables and figures. Fixing it should be
  a single follow-up PR with a targeted regression test and
  before/after CSV diffs.
- **HistSim and Copula fail Christoffersen independence.**
  HistSim has good unconditional coverage but exceptions cluster.
  Copula now over-exceeds on both unconditional and independence
  tests — see the per-method notes in
  `report/overall_var_output_notes.md`.
- **Comparability of realised loss series is resolved.** Before
  PR #28, EVT / GARCH-EVT / Copula and HistSim / Monte Carlo used
  realised loss series with different maxima (~$52.9k vs ~$85.5k).
  After PR #28 all five method backtest files share the same
  realised series (max ≈ $85.5k). The remaining cross-method
  differences are method-internal, not basis-driven.

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
