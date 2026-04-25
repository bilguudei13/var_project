# ATM Straddle on SPY — Volatility Derivative (Non-Linear)

## Instrument Description

A long at-the-money (ATM) straddle on SPY consisting of 20 contracts (2,000 shares). The straddle is repriced daily using Black-Scholes with the VIX as implied volatility. The strike is reset to the current SPY price every 30 trading days (rolling window).

**Theory**: Irle p. 82 — Delta-Gamma-Vega-Theta approximation.

## Position Parameters

| Parameter | Value |
|-----------|-------|
| Underlying | SPY (S&P 500 ETF) |
| Position | Long straddle (long call + long put) |
| Contracts | 20 |
| Shares per contract | 100 |
| Total shares | 2,000 (`STRADDLE_SHARES`) |
| Strike K | ATM at inception (= SPY price at start of 30-day window) |
| Expiry | Rolling 30 trading days; resets every 30 days |
| Implied volatility | VIX / 100 (daily close) |
| Risk-free rate | 5% p.a. (constant, `RF_RATE = 0.05`) |

## Pricing Formula

Black-Scholes straddle price = call + put at same strike:

$$d_1 = \frac{\ln(S/K) + (r + \tfrac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}, \quad d_2 = d_1 - \sigma\sqrt{T}$$

$$C = S \cdot N(d_1) - K e^{-rT} N(d_2)$$

$$P = K e^{-rT} N(-d_2) - S \cdot N(-d_1)$$

$$V^{\text{straddle}} = (C + P) \times 2{,}000$$

where $T = \max\!\left(\frac{30 - \text{days\_held}}{252},\, \frac{1}{252}\right)$.

## Greeks (per single straddle unit, before multiplying by 2,000)

| Greek | Formula | Interpretation |
|-------|---------|----------------|
| Delta | $N(d_1) - N(-d_1)$ | Net directional sensitivity; near zero for ATM |
| Gamma | $\dfrac{2\,N'(d_1)}{S\,\sigma\,\sqrt{T}}$ | Curvature (long gamma) |
| Vega | $2\,S\,N'(d_1)\,\sqrt{T}$ | Sensitivity to implied vol (always positive) |
| Theta | $\dfrac{\theta_C + \theta_P}{252}$ | Daily time decay (always negative for long straddle) |

where

$$\theta_C = -\frac{S\,N'(d_1)\,\sigma}{2\sqrt{T}} - r K e^{-rT} N(d_2)$$

$$\theta_P = -\frac{S\,N'(d_1)\,\sigma}{2\sqrt{T}} + r K e^{-rT} N(-d_2)$$

## Daily P&L

$$\Delta V_t^{\text{straddle}} = \bigl[V^{\text{straddle}}(S_t,\, K,\, T_t,\, \sigma_t) - V^{\text{straddle}}(S_{t-1},\, K,\, T_t + \tfrac{1}{252},\, \sigma_{t-1})\bigr] \times 2{,}000$$

Note: the previous day's price is evaluated at $T_t + 1/252$ (one day longer) to correctly capture theta decay.

## Rolling Mechanics

```
Day 0:      K ← SPY price; days_held = 0
Day 1-29:   days_held += 1; T = (30 - days_held) / 252
Day 30:     K ← current SPY; days_held = 0  (new window)
```

## Code Location

```python
# src/data/portfolio_pricing.py
def price_straddle(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call = S * norm.cdf(d1)  - K * np.exp(-r * T) * norm.cdf(d2)
    put  = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    straddle_price = call + put
    ...
    return straddle_price, greeks
```

```python
# src/data/compute_pnl.py — compute_instrument_pnl()
if days_held >= STRADDLE_DAYS:
    K, days_held = spy.iloc[t], 0
T_today = max((STRADDLE_DAYS - days_held) / 252, 1/252)
p_today, _ = price_straddle(spy.iloc[t],   K, T_today,        RF_RATE, vix_s.iloc[t]   / 100)
p_prev,  _ = price_straddle(spy.iloc[t-1], K, T_today + 1/252, RF_RATE, vix_s.iloc[t-1] / 100)
pnl_straddle_t = (p_today - p_prev) * STRADDLE_SHARES
```

## Risk Factors

| Risk Factor | Symbol | Description |
|-------------|--------|-------------|
| SPY spot price | $S_t$ | Delta and gamma exposure |
| Implied volatility | $\sigma_t = \text{VIX}_t / 100$ | Vega exposure; straddle profits when vol rises |
| Time to expiry | $T_t$ | Theta decay; straddle loses value each day |
| Risk-free rate | $r = 0.05$ | Constant; minor sensitivity via rho |

## Example

With $S = K = 500$, $T = 30/252 \approx 0.119$ yr, $r = 5\%$, $\sigma = 20\%$ (VIX = 20):

$$C + P \approx \$18.91 \quad \Rightarrow \quad V^{\text{straddle}} = \$18.91 \times 2{,}000 = \$37{,}820$$

If VIX jumps from 20 to 25 (+5 pts) in one day, vega gain ≈ $1,880 (long volatility exposure).
