# =============================================================================
# portfolio_pricing.py
# Pricing functions for non-linear instruments in the portfolio
#
# Theory: report/theoretical_background.md
#   Section 1  — Risk mapping: V_t = f(t, Y_t)         (Eq. 1)
#   Section 1  — Delta-Gamma-Vega-Theta approximation   (Irle p. 82)
#   Section 2  — IRS pricing via discount factors       (Irle p. 21, Eq. 7)
#
# Instruments:
#   1. Interest Rate Swap (IRS) — fixed 3%, notional $1M, 10Y maturity
#      Risk factor: 10Y Treasury yield (DGS10)
#
#   2. ATM Straddle on SPY — 30-day rolling, repriced daily
#      Risk factors: SPY price (S), VIX (sigma), risk-free rate (r)
# =============================================================================

import numpy as np
from scipy.stats import norm

def price_straddle(S, K, T, r, sigma):
    """
    Price an ATM straddle using Black-Scholes.
    Straddle = Call + Put at same strike K and expiry T.

    Theory: report/theoretical_background.md — Section 1 (Irle p. 82)
    Risk factors: S (spot), sigma (implied vol), r (risk-free rate), T (expiry)

    Parameters
    ----------
    S     : float — current spot price of SPY
    K     : float — strike price (ATM: K = S at inception)
    T     : float — time to expiry in years (e.g. 30/252)
    r     : float — risk-free rate (annual, decimal)
    sigma : float — implied volatility (annual, decimal, from VIX/100)

    Returns
    -------
    straddle_price : float — call + put value
    greeks         : dict  — delta, gamma, vega, theta of the straddle
    """
    # Black-Scholes d1 and d2
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Call and Put prices
    call = S * norm.cdf(d1)  - K * np.exp(-r * T) * norm.cdf(d2)
    put  = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    straddle_price = call + put

    # Greeks (for Delta-Gamma-Vega-Theta approximation, Irle p. 82)
    # Delta: dV/dS
    delta = norm.cdf(d1) - norm.cdf(-d1)   # call delta - put delta

    # Gamma: d²V/dS²  (same for call and put, so x2)
    gamma = 2 * norm.pdf(d1) / (S * sigma * np.sqrt(T))

    # Vega: dV/d_sigma  (same for call and put, so x2)
    vega  = 2 * S * norm.pdf(d1) * np.sqrt(T)

    # Theta: dV/dt  (per day)
    theta_call = (- S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                  - r * K * np.exp(-r * T) * norm.cdf(d2))
    theta_put  = (- S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                  + r * K * np.exp(-r * T) * norm.cdf(-d2))
    theta = (theta_call + theta_put) / 252   # convert to per day

    greeks = {
        "delta": delta,
        "gamma": gamma,
        "vega" : vega,
        "theta": theta
    }

    return straddle_price, greeks

def price_irs(notional, fixed_rate, swap_rate, maturity=10):
    """
    Price an Interest Rate Swap (fixed payer).
    Value = notional * (swap_rate - fixed_rate) * DV01

    Theory: report/theoretical_background.md — Section 2 (Irle p. 21, Eq. 7)
    Risk factor: swap_rate (proxied by DGS10 / 100)

    Parameters
    ----------
    notional   : float — notional amount (e.g. 1_000_000)
    fixed_rate : float — fixed rate paid (annual, decimal e.g. 0.03)
    swap_rate  : float — current swap rate (annual, decimal e.g. 0.04)
    maturity   : float — swap maturity in years (default 10)

    Returns
    -------
    value : float — mark-to-market value of the IRS (positive = gain)
    dv01  : float — dollar value of 1bp move
    """
    # DV01: sensitivity of IRS value to 1bp move in rates
    # Approximation for a par swap: DV01 = maturity / (1 + swap_rate) * 0.0001
    dv01 = notional * maturity / (1 + swap_rate) * 0.0001

    # Mark-to-market value (fixed payer gains when rates rise)
    value = notional * (swap_rate - fixed_rate) * maturity / (1 + swap_rate)

    return value, dv01

def compute_portfolio_pnl(prices_today, prices_prev,
                          vix_today, dgs10_today,
                          vix_prev, dgs10_prev,
                          weights, notional,
                          fixed_rate, straddle_K,
                          straddle_T, rf_rate):
    """
    Compute 1-day P&L of the full portfolio.

    Portfolio components:
      1. Linear positions (SPY, EWG, EWJ, IEF, GLD, EURUSD)
         DV_linear = V0 * w' * R   (Irle Eq. 10)

      2. IRS — P&L from change in swap rate
         DV_irs = value(dgs10_today) - value(dgs10_prev)

      3. Straddle — P&L from change in S, sigma, T
         DV_straddle = price(today) - price(prev)

    Returns total P&L = DV_linear + DV_irs + DV_straddle
    """
    # --- 1. Linear portfolio P&L ---
    log_returns = np.log(prices_today / prices_prev)
    pnl_linear  = notional * np.dot(weights, log_returns)

    # --- 2. IRS P&L ---
    value_irs_today, _ = price_irs(notional, fixed_rate,
                                    dgs10_today / 100)
    value_irs_prev,  _ = price_irs(notional, fixed_rate,
                                    dgs10_prev  / 100)
    pnl_irs = value_irs_today - value_irs_prev

    # --- 3. Straddle P&L ---
    # T decreases by 1 day each day (30-day rolling)
    price_today, _ = price_straddle(prices_today["SPY"],
                                     straddle_K, straddle_T,
                                     rf_rate, vix_today / 100)
    price_prev,  _ = price_straddle(prices_prev["SPY"],
                                     straddle_K, straddle_T + 1/252,
                                     rf_rate, vix_prev  / 100)
    pnl_straddle = price_today - price_prev

    # --- Total P&L ---
    total_pnl = pnl_linear + pnl_irs + pnl_straddle

    return {
        "total"    : total_pnl,
        "linear"   : pnl_linear,
        "irs"      : pnl_irs,
        "straddle" : pnl_straddle
    }


# =============================================================================
# SANITY CHECKS
# =============================================================================


if __name__ == "__main__":
    # Test straddle pricing
    price, greeks = price_straddle(S=500, K=500, T=30/252,
                                    r=0.05, sigma=0.20)
    print(f"Straddle price : ${price:.2f}")
    print(f"Greeks         : {greeks}")

    # Test IRS pricing
    value, dv01 = price_irs(notional=1_000_000,
                             fixed_rate=0.03,
                             swap_rate=0.04)
    print(f"\nIRS value : ${value:,.2f}")
    print(f"DV01      : ${dv01:,.2f}")