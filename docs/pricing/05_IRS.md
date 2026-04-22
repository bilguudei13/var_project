# IRS — Interest Rate Swap (Fixed Payer, Non-Linear)

## Instrument Description

A plain-vanilla interest rate swap in which the portfolio pays a fixed rate of 3% and receives the floating 10-year Treasury yield (`DGS10`). The swap has a $1,000,000 notional and 10-year maturity. The fixed-payer position gains value when rates rise.

**Theory**: Irle p. 21, Eq. 7 — IRS pricing via discount factors.

## Position Parameters

| Parameter | Value |
|-----------|-------|
| Notional | $1,000,000 |
| Fixed rate paid | 3.00% p.a. (`IRS_FIXED_RATE = 0.03`) |
| Maturity | 10 years |
| Floating rate proxy | DGS10 / 100 (10Y Treasury yield, decimal) |
| Data source | Federal Reserve (FRED DGS10 series) |
| Position | Pay-fixed (receive-floating) |

## Pricing Formula

The mark-to-market value of the fixed-payer IRS is approximated as:

$$V_t^{\text{IRS}} = N \cdot \frac{(r_t - r_{\text{fixed}}) \cdot M}{1 + r_t}$$

where:
- $N = \$1{,}000{,}000$ — notional
- $r_t$ = current swap rate (DGS10 / 100) — decimal
- $r_{\text{fixed}} = 0.03$ — fixed rate paid
- $M = 10$ — maturity in years

**DV01** (dollar value of a 1 bp move in rates):

$$\text{DV01}_t = N \cdot \frac{M}{1 + r_t} \times 0.0001$$

## Daily P&L

$$\Delta V_t^{\text{IRS}} = V_t^{\text{IRS}} - V_{t-1}^{\text{IRS}}$$

For small rate moves $\Delta r$:

$$\Delta V_t^{\text{IRS}} \approx \text{DV01}_t \times \frac{\Delta r}{0.0001}$$

## Code Location

```python
# src/data/portfolio_pricing.py
def price_irs(notional, fixed_rate, swap_rate, maturity=10):
    dv01  = notional * maturity / (1 + swap_rate) * 0.0001
    value = notional * (swap_rate - fixed_rate) * maturity / (1 + swap_rate)
    return value, dv01
```

```python
# src/data/compute_pnl.py — compute_instrument_pnl()
v_today, _ = price_irs(IRS_NOTIONAL, IRS_FIXED_RATE, dgs_s.iloc[t]   / 100)
v_prev,  _ = price_irs(IRS_NOTIONAL, IRS_FIXED_RATE, dgs_s.iloc[t-1] / 100)
pnl_irs_t  = v_today - v_prev
```

## Risk Factors

| Risk Factor | Symbol | Description |
|-------------|--------|-------------|
| 10Y Treasury yield | $r_t$ | Primary driver; 1 bp increase ≈ $9,100 gain (at $r=0.04$) |

## Approximation Notes

The formula uses the standard par-swap DV01 approximation (single-discount-factor). For an exact multi-cash-flow IRS, each fixed coupon would be discounted individually. The approximation is accurate for small shifts in the yield curve and is consistent with Irle p. 21, Eq. 7.

## Example

If 10Y yields rise from 4.00% to 4.05% (+5 bp) with $r_t = 0.04$:
$$\text{DV01} = \$1{,}000{,}000 \times \frac{10}{1.04} \times 0.0001 = \$961.54$$
$$\Delta V^{\text{IRS}} \approx \$961.54 \times 5 = \$4{,}807.69$$
