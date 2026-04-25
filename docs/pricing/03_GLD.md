# GLD — SPDR Gold ETF (Linear Position)

## Instrument Description

Long position in the SPDR Gold Shares ETF (ticker: `GLD`), providing commodity exposure to physical gold. Priced via daily mark-to-market using log-returns.

## Position Parameters

| Parameter | Value |
|-----------|-------|
| Ticker | `GLD` |
| Portfolio weight | 25% (`w = 1/4`) |
| Notional | $250,000 (= $1,000,000 × ¼) |
| Data source | Yahoo Finance daily close |
| Date range | 2006-01-01 → 2024-12-31 |

## Pricing Formula

Daily P&L under the log-return approximation (Irle Eq. 10):

$$\Delta V_t^{\text{GLD}} = V_0 \cdot w_{\text{GLD}} \cdot r_t^{\text{GLD}}$$

where

$$r_t^{\text{GLD}} = \ln\frac{S_t^{\text{GLD}}}{S_{t-1}^{\text{GLD}}}$$

## Risk Factors

| Risk Factor | Symbol | Description |
|-------------|--------|-------------|
| GLD log-return | $r_t$ | Reflects spot gold price change in USD; driven by USD strength, inflation expectations, safe-haven demand |

## Assumptions

- GLD tracks the spot gold price with a small expense ratio (0.40% p.a.); not modelled separately
- USD-denominated; no FX conversion required
- No storage cost modelling at portfolio level

## Code Location

```python
# src/data/compute_pnl.py — compute_total_pnl()
w = pd.Series(weights_aligned)[log_returns.columns].values
linear_pnl = pd.Series(V0 * log_returns.dot(w), ...)
```

```python
# src/data/config.py
TICKERS = {"GLD": "SPDR Gold ETF (Commodity)", ...}
WEIGHTS_DICT = {"GLD": 1/4, ...}
```

## Example

If gold rises 2% in a day:
$$\Delta V = \$1{,}000{,}000 \times 0.25 \times 0.02 = \$5{,}000$$
