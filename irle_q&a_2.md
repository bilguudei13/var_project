# Questions for Professor Meeting

## Goal for the meeting

We want to make our assumptions explicit and keep the project rigorous but efficient. We want to confirm the right level of complexity for the portfolio, the methods, the parameter windows, and the validation/backtesting so we do not either oversimplify or overbuild.

---

## Priority questions to ask first

### 1) Portfolio construction and scope
- Is our current portfolio complex enough for a strong project grade, or would you prefer us to simplify or enrich it?
- Do you prefer a portfolio with nonlinear instruments included, or is a mostly linear portfolio acceptable as long as we compare methods carefully?

### 2) Risk mapping vs position-value simplification
- Would you prefer us to model VaR using direct position returns as risk factors, or to identify underlying risk factors and use pricing functions for each instrument?
- For the option and swap, do you expect full repricing, or is a delta/delta-gamma style approximation acceptable for some methods?

### 3) Which methods are core vs optional
- For this course, which methods do you consider essential for a strong submission: Delta-Normal, Historical Simulation, Monte Carlo, GARCH, EVT, GARCH+EVT?
- Is it better to implement fewer methods well, or more methods at a simpler level?
- Would a strong final project be acceptable if Monte Carlo is basic but the other methods are well validated?

### 4) Acceptable level of complexity within each method
- For Delta-Normal, do you expect delta-only, or would you like us to extend to delta-gamma / delta-vega / theta for nonlinear instruments?
- For GARCH, is a univariate portfolio-return GARCH acceptable, or do you want factor-level modelling?
- For Monte Carlo, is multivariate normal with full repricing enough, or do you want heavier tails / copulas as well?

### 5) Parameter estimation window
- Do you have a preference for rolling vs expanding estimation windows?
- Is a 500-day estimation window a sensible default for this project, or would you prefer a different history length depending on method?
- How much do you want us to explore sensitivity to window length?

### 6) Choice of time period
- Is our chosen sample period appropriate for making method differences visible, or would you recommend a more targeted five-year period around a specific stress episode?
- Should we prioritize including COVID, the GFC, the 2022 rates shock, or a more stable comparison period?

### 7) Backtesting expectations
- What level of backtesting do you expect for each method: exception counts only, or formal Kupiec / Christoffersen-type tests as well?
- Do you want equal emphasis on unconditional coverage and exception clustering?
- Would it help if we compare backtesting performance across calm and stressed subperiods?

### 8) Validation of assumptions
- For each method, what forms of validation would you consider sufficient: statistical tests, graphical diagnostics, parameter stability checks, sensitivity analysis?
- How much formal testing do you expect versus practical discussion of model weaknesses?

---

## Good follow-up questions if there is time

### 9) Nonlinearity treatment
- Because our portfolio contains nonlinear instruments, would you prefer us to present Delta-Normal mainly as a benchmark and then show where it breaks down?
- Should we explicitly compare linear approximation versus full repricing for the option exposure?

### 10) Historical Simulation variants
- Would you value extensions like weighted historical simulation, volatility-adjusted historical simulation, or stressed historical simulation, or should we keep HS basic and spend time elsewhere?

### 11) EVT details
- For EVT, how much depth do you expect on threshold selection and diagnostics?
- Is a practical POT implementation with threshold discussion sufficient, or do you want a deeper statistical treatment?

### 12) GARCH scope
- Would you prefer a pure GARCH VaR model as its own method, or is GARCH+EVT enough as the main dynamic-volatility method?
- Is ARMA+GARCH materially better for this course, or is GARCH(1,1) adequate if explained well?

### 13) Monte Carlo design
- For Monte Carlo, do you prefer factor simulation with a simple multivariate normal structure, or would using Student-t / copula dependence add meaningful value in your view?
- Is PCA-based factor reduction something you would see as useful here, or unnecessary?

### 14) Model comparison
- In the final presentation, what would you most want to see in the comparison: VaR time series, exception plots, crisis-period behavior, assumption summaries, or ranking by strengths/weaknesses?
- Do you want one “best model” recommendation at the end, or a more balanced discussion of trade-offs?

### 15) Sensitivity / robustness
- How much sensitivity analysis would you expect on confidence level, window length, estimation frequency, and stressed-period choice?
- Would one focused robustness section be enough, or do you want sensitivity checks method by method?

### 16) Attribution / component risk
- Would you consider a small component VaR or risk-contribution section useful, or is that beyond what you want for this project?

---

## Questions that help avoid common traps
- What would you consider unnecessary complexity for this course?
- Are there any modelling extensions that look sophisticated but would not materially improve the grade?
- If we have limited time, where would you prefer us to invest it: better validation, more methods, richer portfolio modelling, or stronger presentation?
- What are the most common weaknesses you see in past VaR projects?

---

## Best short list if time is tight
1. Is our portfolio structure appropriate, especially with nonlinear instruments included?
2. Do you prefer risk-factor modelling with pricing functions, or position-return simplification?
3. Which VaR methods do you see as essential versus optional for a strong submission?
4. What level of complexity do you expect inside each method, especially Delta-Normal, Monte Carlo, and GARCH?
5. What parameter window / update frequency would you consider reasonable?
6. What level of backtesting and assumption validation do you expect for each model?
