# Portfolio Pricing — Overview

**Portfolio**: $1,000,000 notional | 2006-01-01 → 2024-12-31 | 4,769 trading days

## Instrument Summary

| # | Instrument | Type | Weight / Size | Risk Factors | Pricing File |
|---|------------|------|---------------|--------------|--------------|
| 1 | SPY | US Equity ETF (linear) | 25% of $1M | SPY log-return | `compute_pnl.py` |
| 2 | IEF | Treasury Bond ETF (linear) | 25% of $1M | IEF log-return | `compute_pnl.py` |
| 3 | GLD | Gold Commodity ETF (linear) | 25% of $1M | GLD log-return | `compute_pnl.py` |
| 4 | EURUSD | FX Spot (linear) | 25% of $1M | EURUSD log-return | `compute_pnl.py` |
| 5 | IRS | Interest Rate Swap (non-linear) | $1M notional, 10Y, pay-fixed 3% | DGS10 (10Y yield) | `portfolio_pricing.py` |
| 6 | Straddle | ATM Straddle on SPY (non-linear) | 20 contracts × 100 shares | SPY, VIX, time | `portfolio_pricing.py` |

## Total Daily P&L

$$\Delta V_t = \Delta V_t^{\text{linear}} + \Delta V_t^{\text{IRS}} + \Delta V_t^{\text{straddle}}$$

where

$$\Delta V_t^{\text{linear}} = V_0 \sum_{i=1}^{4} w_i \, r_{i,t}, \quad w_i = \tfrac{1}{4}, \quad r_{i,t} = \ln\frac{S_{i,t}}{S_{i,t-1}}$$

Source: `src/data/compute_pnl.py` → `compute_total_pnl()`

## Data Inputs

| File | Contents | Source |
|------|----------|--------|
| `data/raw/prices.csv` | Daily close prices for SPY, IEF, GLD, EURUSD=X | Yahoo Finance |
| `data/raw/vix.csv` | CBOE VIX daily close | Yahoo Finance (`^VIX`) |
| `data/raw/dgs10.csv` | 10Y Treasury constant maturity yield (%) | Federal Reserve (FRED) |
| `data/processed/log_returns.csv` | Log-returns for 4 linear assets | `compute_returns.py` |
| `data/processed/instrument_pnl.csv` | Daily IRS + straddle P&L | `compute_pnl.py` |
| `data/processed/total_portfolio_pnl.csv` | Total portfolio P&L | `compute_pnl.py` |

## Architecture

```
download_data.py          → data/raw/
compute_returns.py        → data/processed/log_returns.csv
compute_pnl.py            → data/processed/instrument_pnl.csv
                          → data/processed/total_portfolio_pnl.csv
```

Non-linear pricing is centralised in `src/data/portfolio_pricing.py` which exposes two functions:
- `price_irs(notional, fixed_rate, swap_rate, maturity=10)`
- `price_straddle(S, K, T, r, sigma)`

## Theoretical References

All notation follows Dr. Sebastian Irle's lecture notes:
- Risk mapping `V_t = f(t, Y_t)` — Section 1, Eq. 1
- Delta-Gamma-Vega-Theta approximation — Irle p. 82
- IRS pricing via discount factors — Section 2, Irle p. 21, Eq. 7
- Log-return linear P&L — Irle Eq. 10

---
*Generated from `src/data/portfolio_pricing.py`, `src/data/config.py`, `src/data/compute_pnl.py`*
