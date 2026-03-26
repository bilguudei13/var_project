# Theoretical Background — VaR Calculation & Evolution
**Course**: Market Risk Modelling  
**Lecturer**: Dr. Sebastian Irle  
**Reference**: Lecture Notes, Market Risk Modelling (all equation numbers refer to the lecture notes)

---

## Table of Contents

1. [Risk Factors and Risk Mapping](#1-risk-factors-and-risk-mapping)
2. [Returns: Log vs Discrete](#2-returns-log-vs-discrete)
3. [Value at Risk — Definition](#3-value-at-risk--definition)
4. [Expected Shortfall — Definition](#4-expected-shortfall--definition)
5. [Delta-Normal VaR](#5-delta-normal-var)
6. [Historical Simulation VaR](#6-historical-simulation-var)
7. [Monte Carlo Simulation VaR](#7-monte-carlo-simulation-var)
8. [GARCH-Based VaR](#8-garch-based-var)
9. [Extreme Value Theory (EVT) VaR](#9-extreme-value-theory-evt-var)
10. [GARCH + EVT VaR](#10-garch--evt-var)
11. [Backtesting](#11-backtesting)
12. [Assumption Validation Tests](#12-assumption-validation-tests)

---

## 1. Risk Factors and Risk Mapping

**Source**: Irle, Section 2

The value $V_t$ of a position at time $t$ is written as a function of $d$ random risk factors $Y_t = (Y^1_t, \ldots, Y^d_t)^\top$:

$$V_t = f(t, Y_t) \tag{Eq. 1}$$

The change in value over horizon $[0, \Delta t]$ is:

$$\Delta V = f(\Delta t,\ Y_0 + \Delta Y) - f(0, Y_0) \tag{Eq. 2}$$

where $\Delta Y = Y_{\Delta t} - Y_0$ are the factor changes.

Applying **Taylor's formula** (first-order linearisation):

$$\Delta V \approx \frac{\partial f}{\partial t}\Delta t + \sum_{j=1}^{d} \frac{\partial f}{\partial y_j} \Delta Y^j \tag{Eq. 3}$$

Using **relative changes** (risk factor returns) $R^{(j)} = \Delta Y^j / Y^j_0$:

$$\Delta V \approx \frac{\partial f}{\partial t}\Delta t + \sum_{j=1}^{d} \underbrace{Y^j_0 \frac{\partial f}{\partial y_j}}_{\text{delta equivalent}} \cdot R^{(j)} \tag{Eq. 5}$$

For a **linear portfolio** $V = \sum_i a_i S^i$:
- Risk factors: asset prices $S^i$
- Delta equivalents: $a_i S^i_0$
- Value change: $\Delta V = V_0 \sum_i w_i R^{(i)} = V_0 \cdot R^{d,P}$

The process of identifying risk factors and function $f$ is called **risk mapping**.

---

## 2. Returns: Log vs Discrete

**Source**: Irle, Section 4

| | Discrete Return | Log Return |
|---|---|---|
| **Definition** | $R^d_t = S_t/S_{t-1} - 1$ | $R^c_t = \ln(S_t/S_{t-1})$ |
| **Additive across** | Portfolio | Time |

**Portfolio aggregation** (Eq. 10):
$$R^{d,P} = \sum_i w_i R^{d,i}$$

**Time aggregation** of log-returns:
$$R^c_{0,2} = R^c_{0,1} + R^c_{1,2}$$

**Approximation for small $\Delta t$** (valid for 1-day horizon, Irle p. 35):
$$R^c_t \approx R^d_t$$

**Implementation note**: We use log-returns throughout as approximations of discrete returns. Parameters are estimated from return data (not absolute changes) because return distributions are more stable over time (Irle p. 57).

---

## 3. Value at Risk — Definition

**Source**: Irle, Section 5.1

$\text{VaR}_{\alpha, \Delta t}$ is defined as the $(1-\alpha)$-quantile of the loss $-\Delta V$:

$$1 - \alpha = P(-\Delta V \geq \text{VaR}_{\alpha,\Delta t}) \tag{Eq. 11}$$

- **This project**: $\alpha = 99\%$, $\Delta t = 1$ day
- VaR is a **positive number** representing a potential loss
- An **exception (breach)** occurs when $-\Delta V_t > \text{VaR}_t$

**Square-root-of-time scaling** (Irle p. 57–58):
$$\text{VaR}_{10\text{d}} = \text{VaR}_{1\text{d}} \cdot \sqrt{10}$$

Valid under the assumption that variance scales linearly with $\Delta t$ (i.e., iid log-returns).

---

## 4. Expected Shortfall — Definition

**Source**: Irle, Section 5.2

Expected Shortfall (ES) is the **expected loss given that the loss exceeds VaR**:

$$\text{ES}_\alpha = E[-\Delta V \mid -\Delta V \geq \text{VaR}_\alpha]$$

For **normally distributed** $\Delta V \sim \mathcal{N}(\mu, \sigma^2)$ (Irle p. 60):

$$\text{ES}_\alpha = -\mu + \sigma \frac{n(N_\alpha)}{1 - \alpha}$$

where $n(\cdot)$ is the standard normal density and $N_\alpha$ is the $\alpha$-quantile of $\mathcal{N}(0,1)$.

ES is a **coherent risk measure** (satisfies subadditivity); VaR is not always coherent.

---

## 5. Delta-Normal VaR

**Source**: Irle, Section 5.4

**Core assumption**: $\Delta V \sim \mathcal{N}(\mu_{\Delta V},\ \sigma^2_{\Delta V})$

**Derivation** (Irle p. 53–55):

$$1 - \alpha = P\!\left(\frac{-\Delta V + \mu}{\sigma} \geq \frac{\text{VaR} + \mu}{\sigma}\right) = 1 - N\!\left(\frac{\text{VaR} + \mu}{\sigma}\right)$$

Solving for VaR:

$$\boxed{\text{VaR}_{\alpha} = \sigma_{\Delta V} \cdot N_\alpha - \mu_{\Delta V}} \tag{Eq. 14}$$

**Ignoring drift** (standard for 1-day horizon, Irle p. 56):

$$\boxed{\text{VaR}_{\alpha} = N_\alpha \cdot \sigma_{\Delta V}} \tag{Eq. 15}$$

**Normal quantiles** (Irle table, p. 56):

| $\alpha$ | $N_\alpha$ |
|---|---|
| 99.00% | **2.3263** |
| 95.00% | 1.6449 |
| 90.00% | 1.2816 |

**Portfolio variance** via covariance matrix $\Sigma$:

$$\sigma^2_{\Delta V} = V_0^2 \cdot \mathbf{w}^\top \Sigma \mathbf{w}$$

**Rolling estimation**: For each day $t$, estimate $\mu$ and $\Sigma$ from the $k=500$ most recent observations.

**Key limitation**: Assumes normality — violated for all assets in this portfolio (KS test, $p \approx 0$; excess kurtosis up to 105).

---

## 6. Historical Simulation VaR

**Source**: Irle, Section 6.1

**Core idea**: Use historically observed factor changes as simulation scenarios. No distributional assumption required.

**Algorithm**:
1. Collect $k$ historical risk factor returns: $\Delta y_1, \ldots, \Delta y_k$
2. For each scenario $i$, compute portfolio P&L: $\Delta V(\Delta y_i) = V_0 \cdot \mathbf{w}^\top \Delta y_i$
3. Estimate VaR as the empirical quantile (Irle p. 80):

$$\text{VaR}_{\alpha} \approx -\hat{Q}_{1-\alpha}$$

where $\hat{Q}_{1-\alpha}$ is the $\lfloor (1-\alpha) \cdot k + 1 \rfloor$-smallest value of $\{\Delta V(\Delta y_i)\}$.

**Advantages**:
- No distributional assumption
- Automatically captures fat tails and skewness
- Dependencies captured via joint historical scenarios

**Limitations** (Irle p. 84–85):
- Appropriate window size $k$ is unclear: long history → stable estimate but may include stale data
- Very long series may contain too many extreme events (overestimates risk)
- Very short series may contain too few extreme events (underestimates risk)
- Cannot extrapolate beyond historically observed events

**Rolling window**: $k = 500$ days, re-estimated daily.

---

## 7. Monte Carlo Simulation VaR

**Source**: Irle, Section 6.2

**Core idea**: Simulate risk factor changes from an assumed joint distribution, compute P&L for each scenario, estimate VaR as empirical quantile.

**Standard implementation** — multivariate normal (Irle p. 87):

$$\Delta Y = (\Delta Y^1, \ldots, \Delta Y^d)^\top \sim \mathcal{N}(\mu_{\Delta Y},\ \text{Cov}(\Delta Y))$$

**Algorithm**:
1. Estimate $\mu$ and $\Sigma$ from rolling window of $k$ historical observations
2. Simulate $M$ scenarios: $\Delta y_i \sim \mathcal{N}(\mu, \Sigma)$, $i = 1, \ldots, M$
3. Compute P&L for each: $\Delta V_i = V_0 \cdot \mathbf{w}^\top \Delta y_i$
4. VaR = $-\hat{Q}_{1-\alpha}$ of $\{\Delta V_i\}$

**Number of simulations**: $M = 10{,}000$ (sufficient for stable 99% quantile estimate).

**Advantages over Delta-Normal**:
- Can handle non-linear risk mappings
- More flexible distributional assumptions (e.g., Student-$t$ marginals with Gaussian copula — industry standard per Irle p. 88)

**Advantages over Historical Simulation**:
- More scenarios possible (not limited by historical data length)
- Can simulate scenarios not observed historically

**Key challenge**: Choice of joint distribution for $(\Delta Y^1, \ldots, \Delta Y^d)$.

---

## 8. GARCH-Based VaR

**Source**: Irle, Section 8.3 and Case Study (p. 172–186)

### 8.1 GARCH(1,1) Model

For a return series $X_t = \sigma_t Z_t$ where $Z_t \overset{iid}{\sim} \mathcal{N}(0,1)$:

$$\sigma^2_t = \alpha_0 + \alpha_1 X^2_{t-1} + \beta \sigma^2_{t-1} \tag{GARCH}$$

with constraints: $\alpha_0 > 0$, $\alpha_1 \geq 0$, $\beta \geq 0$, $\alpha_1 + \beta < 1$.

**Unconditional variance**:
$$\text{Var}(X_t) = \frac{\alpha_0}{1 - \alpha_1 - \beta}$$

### 8.2 ARMA + GARCH

For returns with serial autocorrelation (Irle p. 174–176):

$$Y_t = \sum_{i=1}^{p} \phi_i Y_{t-i} + \sum_{j=1}^{q} \theta_j X_{t-j} + X_t \quad \text{(ARMA)}$$
$$X_t = \sigma_t Z_t, \quad \sigma^2_t = \alpha_0 + \alpha_1 X^2_{t-1} + \beta \sigma^2_{t-1} \quad \text{(GARCH)}$$

Model selection via **BIC/AIC** (Irle p. 165).

### 8.3 VaR from GARCH

One-step-ahead variance forecast (Irle p. 151, Eq. 18):

$$\hat{\sigma}^2_{t+1} = \alpha_0 + \alpha_1 X^2_t + \beta \hat{\sigma}^2_t$$

VaR using conditional volatility (Irle p. 152):

$$\boxed{\text{VaR}_{\alpha, \Delta t} = -V_t \cdot \hat{\sigma}_{t+1} \cdot N_{1-\alpha}}$$

Or with Student-$t$ innovations (Irle p. 179–180):

$$\text{VaR}_{\alpha, \Delta t} = -V_t \cdot \hat{\sigma}_{t+1} \cdot t^{-1}_\nu(1-\alpha)$$

where $t^{-1}_\nu$ is the inverse CDF of the Student-$t$ with $\nu$ degrees of freedom.

### 8.4 Rolling Estimation

- Re-estimate GARCH parameters every **50 observations** using the 500 most recent (Irle p. 182, Case Study Step 7)
- Daily one-step-ahead forecast $\hat{\sigma}_{t+1|t}$

**Implementation**: R package `rugarch` (Irle p. 169 — "all tests implemented in rugarch")

---

## 9. Extreme Value Theory (EVT) VaR

**Source**: Irle, Section 9

### 9.1 Overview

EVT models the **tail of the return distribution** directly, without assuming normality across the full distribution.

Two approaches:
1. **Block Maxima** → Generalised Extreme Value (GEV) distribution
2. **Threshold Exceedances (POT)** → Generalised Pareto Distribution (GPD) ← used for VaR

### 9.2 Generalised Pareto Distribution (GPD)

**Definition** (Irle p. 212, Definition 7):

$$G_{\xi,\beta}(x) = \begin{cases} 1 - (1 + \xi x/\beta)^{-1/\xi} & \xi \neq 0 \\ 1 - e^{-x/\beta} & \xi = 0 \end{cases}$$

with $\beta > 0$, $x \geq 0$ for $\xi \geq 0$, $0 \leq x \leq -\beta/\xi$ for $\xi < 0$.

**Shape parameter** $\xi$:
- $\xi > 0$: heavy-tailed Pareto (expected for equity returns)
- $\xi = 0$: exponential (light-tailed)
- $\xi < 0$: short-tailed Pareto type II

### 9.3 Pickands-Balkema-de Haan Theorem

For a large class of distributions $F$, the excess distribution above threshold $u$:

$$F_u(x) = P(X - u \leq x \mid X > u) \approx G_{\xi,\beta(u)}(x) \quad \text{for large } u$$

### 9.4 VaR from GPD (Irle p. 221–225)

**Threshold exceedances**: $N_u$ = number of observations exceeding $u$ out of $N$ total.

VaR formula (derived in Irle p. 223–225):

$$\boxed{\text{VaR}_\alpha = u + \frac{\beta}{\xi} \left[\left(\frac{1-\alpha}{N_u/N}\right)^{-\xi} - 1\right]} \tag{EVT VaR}$$

**Expected Shortfall** (Irle p. 221, valid for $\xi < 1$):

$$\text{ES}_\alpha = \frac{\text{VaR}_\alpha}{1 - \xi} + \frac{\beta - \xi u}{1 - \xi}$$

### 9.5 Threshold Selection

Use the **mean excess plot** (MEF plot):
- Plot $e(u) = E[X - u \mid X > u]$ against $u$
- The GPD approximation is valid where the plot is approximately **linear**
- Choose $u$ as the point where linearity begins

**Implementation**: R packages `POT`, `evd`, `ismev`

---

## 10. GARCH + EVT VaR

**Source**: Irle, Section 8.3 + Section 9 (combined approach)

**Motivation**: GARCH captures **volatility clustering** (time-varying variance) but still assumes a parametric innovation distribution. EVT models the **tail of the residuals** non-parametrically.

**Algorithm**:

**Step 1 — Fit GARCH(1,1)**:
$$X_t = \sigma_t Z_t, \quad \sigma^2_t = \alpha_0 + \alpha_1 X^2_{t-1} + \beta \sigma^2_{t-1}$$

**Step 2 — Extract standardised residuals**:
$$\hat{Z}_t = X_t / \hat{\sigma}_t$$

**Step 3 — Fit GPD to tail of $\hat{Z}_t$**:
$$\hat{Z}_t \sim \text{GPD}(\xi, \beta) \text{ for } \hat{Z}_t > u$$

**Step 4 — VaR forecast**:
$$\text{VaR}_{\alpha, t+1} = \hat{\sigma}_{t+1} \cdot \text{VaR}^{\text{GPD}}_\alpha(\hat{Z})$$

where $\text{VaR}^{\text{GPD}}_\alpha$ uses the EVT formula from Section 9.4 applied to the standardised residuals.

**Advantage**: Combines time-varying volatility (GARCH) with accurate tail modelling (EVT) — the most rigorous approach in this project.

---

## 11. Backtesting

**Source**: Irle, Section 8 (Case Study, p. 183–185)

### 11.1 Exception Counting

An **exception** (breach) on day $t$:
$$-\Delta V_t > \text{VaR}_t$$

Under a correctly specified model at confidence $\alpha$:
$$N \sim B(T,\ 1 - \alpha) = B(T,\ 0.01)$$

Expected number of exceptions:
$$E[N] = T \cdot (1 - \alpha)$$

Irle example (p. 183): $T = 500$, $\alpha = 95\%$ → $E[N] = 25$, 95% CI = $[16, 35]$.

### 11.2 Kupiec Test (Unconditional Coverage)

Tests whether the observed exception rate matches the theoretical rate:

$$H_0: p = 1 - \alpha$$

Likelihood ratio test statistic:

$$LR_{uc} = -2 \ln\!\left[\frac{(1-\alpha)^{T-N} \alpha^N}{\hat{p}^N (1-\hat{p})^{T-N}}\right] \overset{d}{\to} \chi^2(1)$$

where $\hat{p} = N/T$.

### 11.3 Christoffersen Test (Conditional Coverage)

Tests both correct exception rate **and** independence of exceptions (no clustering):

$$LR_{cc} = LR_{uc} + LR_{ind} \overset{d}{\to} \chi^2(2)$$

Clustering of exceptions (as seen during GFC 2008 and COVID 2020) indicates model misspecification.

---

## 12. Assumption Validation Tests

**Source**: Irle, Section 8.5 (Criteria for Model Selection, p. 163–171)

### 12.1 Normality — Kolmogorov-Smirnov Test

Tests whether the empirical distribution matches a specified distribution:

$$H_0: F_{\text{empirical}} = F_{\text{normal}}$$

Irle references: KS test and Adjusted Pearson goodness-of-fit test (p. 167).  
We use KS (implemented in `scipy.stats.kstest`).

### 12.2 Autocorrelation — Ljung-Box Test

Tests whether the first $m$ autocorrelations of residuals are jointly zero (Irle p. 166):

$$H_0: \rho_1 = \rho_2 = \cdots = \rho_m = 0$$

Applied to: GARCH standardised residuals $\hat{Z}_t$ (check: no serial correlation).

### 12.3 ARCH Effects — Engle's ARCH Test

Tests for remaining conditional heteroskedasticity in residuals (Irle p. 166–167):

$$\hat{Z}^2_t = \alpha_0 + \alpha_1 \hat{Z}^2_{t-1} + \cdots + \alpha_m \hat{Z}^2_{t-m} + u_t$$

$$H_0: \alpha_1 = \cdots = \alpha_m = 0$$

Applied to: GARCH residuals (check: no remaining ARCH effects).

### 12.4 Asymmetry — Engle-Ng Sign Bias Test

Tests for asymmetric news impact (Irle p. 167–168):

$$\hat{Z}^2_t = c_0 + c_1 \mathbf{1}_{\hat{e}_{t-1}<0} + c_2 \mathbf{1}_{\hat{e}_{t-1}<0}\hat{e}_{t-1} + c_3 \mathbf{1}_{\hat{e}_{t-1}\geq 0}\hat{e}_{t-1} + u_t$$

$H_0: c_1 = c_2 = c_3 = 0$ (no asymmetry).  
If rejected → consider EGARCH or GJR-GARCH (asymmetric models).

### 12.5 Parameter Stability — Nyblom Test

Tests whether GARCH parameters are stable over time (Irle p. 168):

$$H_0: \text{all parameters constant over time}$$

### 12.6 EVT Threshold Selection — Mean Excess Plot

For EVT, choose threshold $u$ where the mean excess function:
$$e(u) = E[X - u \mid X > u]$$
is approximately **linear** in $u$ (Irle Section 9.2).

---

## Summary Table

| Method | Distribution Assumption | Captures Fat Tails? | Captures Volatility Clustering? | Reference |
|---|---|---|---|---|
| **Delta-Normal** | Normal (full) | NO | Partially (rolling $\sigma$) | Irle Eq. 14–15 |
| **Historical Sim.** | None | YES | Partially | Irle p. 83–85 |
| **Monte Carlo** | Normal (simulated) | NO (default) | No (unless GARCH inputs) | Irle p. 86–88 |
| **GARCH** | Normal/t innovations | Partially | YES | Irle p. 140–152 |
| **EVT (GPD/POT)** | GPD for tail | YES | NO | Irle p. 209–226 |
| **GARCH + EVT** | GPD on GARCH residuals | YES | YES | Irle p. 140–226 |

---

*Last updated: 2024*  
*All equation numbers refer to: Irle, S. — Market Risk Modelling Lecture Notes*