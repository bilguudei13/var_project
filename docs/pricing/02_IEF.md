# IEF — iShares 7-10Y Treasury ETF (Linear Position)

## Instrument Description

Long position in the iShares 7-10 Year Treasury Bond ETF (ticker: `IEF`), representing US investment-grade fixed income exposure. Priced via daily mark-to-market using log-returns.

## Position Parameters

| Parameter | Value |
|-----------|-------|
| Ticker | `IEF` |
| Portfolio weight | 25% (`w = 1/4`) |
| Notional | $250,000 (= $1,000,000 × ¼) |
| Data source | Yahoo Finance daily close |
| Date range | 2006-01-01 → 2024-12-31 |

## Pricing Formula

Daily P&L under the log-return approximation (Irle Eq. 10):

$$\Delta V_t^{\text{IEF}} = V_0 \cdot w_{\text{IEF}} \cdot r_t^{\text{IEF}}$$

where

$$r_t^{\text{IEF}} = \ln\frac{S_t^{\text{IEF}}}{S_{t-1}^{\text{IEF}}}$$

## Risk Factors

| Risk Factor | Symbol | Description |
|-------------|--------|-------------|
| IEF log-return | $r_t$ | Captures duration risk; price moves inversely to interest rates |

## Assumptions

- Log-return approximation; no yield curve modelling at the ETF level
- Duration risk enters implicitly through IEF price moves
- Coupon reinvestment captured via ETF total return pricing

## Code Location

```python
# src/data/compute_pnl.py — compute_total_pnl()
w = pd.Series(weights_aligned)[log_returns.columns].values
linear_pnl = pd.Series(V0 * log_returns.dot(w), ...)
```

```python
# src/data/config.py
TICKERS = {"IEF": "iShares 7-10Y Treasury ETF (Bonds)", ...}
WEIGHTS_DICT = {"IEF": 1/4, ...}
```

## Example

If IEF falls 0.5% on a day when rates rise:
$$\Delta V = \$1{,}000{,}000 \times 0.25 \times (-0.005) = -\$1{,}250$$
