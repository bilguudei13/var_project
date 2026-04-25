# EURUSD — EUR/USD FX Spot Rate (Linear Position)

## Instrument Description

Long position in the EUR/USD exchange rate (ticker in Yahoo Finance: `EURUSD=X`; stored internally as `EURUSD`). Represents a long-EUR / short-USD exposure. Priced via daily mark-to-market using log-returns.

## Position Parameters

| Parameter | Value |
|-----------|-------|
| Yahoo ticker | `EURUSD=X` |
| Internal column name | `EURUSD` |
| Portfolio weight | 25% (`w = 1/4`) |
| Notional | $250,000 (= $1,000,000 × ¼) |
| Data source | Yahoo Finance daily close |
| Date range | 2006-01-01 → 2024-12-31 |

## Pricing Formula

Daily P&L under the log-return approximation (Irle Eq. 10):

$$\Delta V_t^{\text{EURUSD}} = V_0 \cdot w_{\text{EURUSD}} \cdot r_t^{\text{EURUSD}}$$

where

$$r_t^{\text{EURUSD}} = \ln\frac{S_t^{\text{EURUSD}}}{S_{t-1}^{\text{EURUSD}}}$$

## Risk Factors

| Risk Factor | Symbol | Description |
|-------------|--------|-------------|
| EURUSD log-return | $r_t$ | Captures currency appreciation/depreciation; driven by ECB/Fed policy differentials |

## Column Rename

The Yahoo Finance ticker `EURUSD=X` is renamed to `EURUSD` during download to avoid special-character issues in CSV column headers:

```python
# src/data/download_data.py
prices.rename(columns={"EURUSD=X": "EURUSD"}, inplace=True)
```

The `WEIGHTS_DICT` lookup compensates via:

```python
# src/data/compute_pnl.py — compute_total_pnl()
weights_aligned = {
    ("EURUSD" if k == "EURUSD=X" else k): v
    for k, v in weights_dict.items()
}
```

## Assumptions

- Long EUR position; positive return when EUR appreciates vs USD
- No carry (interest rate differential) modelled at portfolio level
- No bid/ask spread

## Example

If EUR/USD falls 0.3% in a day:
$$\Delta V = \$1{,}000{,}000 \times 0.25 \times (-0.003) = -\$750$$
