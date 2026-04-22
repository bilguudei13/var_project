# SPY — S&P 500 ETF (Linear Position)

## Instrument Description

Long position in the SPDR S&P 500 ETF (ticker: `SPY`), representing broad US equity exposure. Priced via daily mark-to-market using log-returns.

## Position Parameters

| Parameter | Value |
|-----------|-------|
| Ticker | `SPY` |
| Portfolio weight | 25% (`w = 1/4`) |
| Notional | $250,000 (= $1,000,000 × ¼) |
| Data source | Yahoo Finance daily close |
| Date range | 2006-01-01 → 2024-12-31 |

## Pricing Formula

Daily P&L under the log-return approximation (Irle Eq. 10):

$$\Delta V_t^{\text{SPY}} = V_0 \cdot w_{\text{SPY}} \cdot r_t^{\text{SPY}}$$

where

$$r_t^{\text{SPY}} = \ln\frac{S_t^{\text{SPY}}}{S_{t-1}^{\text{SPY}}}$$

For small returns, $e^r - 1 \approx r$, so this equals the arithmetic return scaled by notional.

## Risk Factors

| Risk Factor | Symbol | Description |
|-------------|--------|-------------|
| SPY log-return | $r_t$ | Primary driver; all other risk factors are zero for this instrument |

## Assumptions

- Continuous compounding / log-return approximation
- No dividends modelled explicitly (price reflects total return via ETF reinvestment)
- No bid/ask spread or transaction costs

## Code Location

```python
# src/data/compute_pnl.py — compute_total_pnl()
w = pd.Series(weights_aligned)[log_returns.columns].values
linear_pnl = pd.Series(V0 * log_returns.dot(w), ...)
```

```python
# src/data/config.py
TICKERS = {"SPY": "S&P 500 ETF (US Equity)", ...}
WEIGHTS_DICT = {"SPY": 1/4, ...}
V0 = 1_000_000
```

## Example (Irle Eq. 10)

If SPY rises 1% on a given day:
$$\Delta V = \$1{,}000{,}000 \times 0.25 \times 0.01 = \$2{,}500$$
