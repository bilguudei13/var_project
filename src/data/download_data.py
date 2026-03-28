# =============================================================================
# download_data.py
# Downloads and saves price data for the VaR portfolio
#
# Portfolio:
#   SPY      - S&P 500 ETF          (US Equity)
#   EWG      - iShares Germany ETF  (European Equity / DAX proxy)
#   EWJ      - iShares Japan ETF    (Asian Equity / Nikkei proxy)
#   IEF      - iShares 7-10Y Treasury Bond ETF  (US Bonds)
#   GLD      - SPDR Gold Shares     (Commodity)
#   EURUSD=X - EUR/USD exchange rate (FX)
#   IRS      - Interest Rate Swap   (proxied via DGS10 from FRED)
#   Straddle - ATM 30-day rolling straddle on SPY (Black-Scholes)
#
# Theory: report/theoretical_background.md
#   Section 1 — Risk mapping: V_t = f(t, Y_t)        (Irle Eq. 1-2)
#   Section 2 — Log vs discrete returns               (Irle p. 35)
#   Section 1 — Delta-Gamma-Vega-Theta approximation  (Irle p. 82)
#
# Period: 2006-01-01 to 2024-12-31
# =============================================================================

import os
import sys
import numpy as np
import pandas as pd
import yfinance as yf
import pandas_datareader.data as web

sys.path.append("src/data")
from portfolio_pricing import price_irs, price_straddle

# =============================================================================
# SETTINGS
# =============================================================================

TICKERS = {
    "SPY":      "S&P 500 ETF (US Equity)",
    "EWG":      "iShares Germany ETF (EU Equity)",
    "EWJ":      "iShares Japan ETF (Asian Equity)",
    "IEF":      "iShares 7-10Y Treasury ETF (Bonds)",
    "GLD":      "SPDR Gold ETF (Commodity)",
    "EURUSD=X": "EUR/USD Exchange Rate (FX)"
}

START_DATE = "2006-01-01"
END_DATE   = "2024-12-31"

# Equal weights for 6 linear positions
WEIGHTS_DICT = {
    "SPY":      1/6,
    "EWG":      1/6,
    "EWJ":      1/6,
    "IEF":      1/6,
    "GLD":      1/6,
    "EURUSD=X": 1/6
}

# Portfolio notional
V0 = 1_000_000

# IRS settings
IRS_NOTIONAL   = 1_000_000
IRS_FIXED_RATE = 0.03

# Straddle settings
STRADDLE_DAYS       = 30     # rolling window in trading days
STRADDLE_CONTRACTS  = 20     # number of contracts
SHARES_PER_CONTRACT = 100    # standard contract size
STRADDLE_SHARES     = STRADDLE_CONTRACTS * SHARES_PER_CONTRACT  # 2000
RF_RATE             = 0.05   # risk-free rate (constant approximation)

# Output paths
RAW_DIR       = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")

# =============================================================================
# STEP 1 — DOWNLOAD RAW PRICES
# =============================================================================

def download_prices(tickers, start, end):
    """
    Download adjusted closing prices for all tickers plus VIX and DGS10.
    Returns: prices DataFrame, vix Series, dgs10 Series.
    """
    print("=" * 60)
    print("Downloading price data from Yahoo Finance...")
    print(f"Period : {start} to {end}")
    print(f"Tickers: {list(tickers.keys())}")
    print("=" * 60)

    data = yf.download(
        tickers     = list(tickers.keys()),
        start       = start,
        end         = end,
        auto_adjust = True,
        progress    = True
    )

    prices = data["Close"].rename(columns={"EURUSD=X": "EURUSD"})

    print(f"\nDownloaded {len(prices)} trading days")
    print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"Columns   : {list(prices.columns)}")

    # VIX — implied volatility, risk factor for straddle pricing
    print("\nDownloading VIX...")
    vix = yf.download("^VIX", start=start, end=end,
                      auto_adjust=True, progress=False)["Close"]
    vix.name = "VIX"
    print(f"VIX: {len(vix)} days")

    # DGS10 from FRED — 10Y Treasury yield, risk factor for IRS
    print("Downloading DGS10 from FRED...")
    dgs10 = web.DataReader("DGS10", "fred", start, end)["DGS10"]
    dgs10 = dgs10.ffill().dropna()
    dgs10.name = "DGS10"
    print(f"DGS10: {len(dgs10)} days")

    return prices, vix, dgs10

# =============================================================================
# STEP 2 — CLEAN THE DATA
# =============================================================================

def clean_prices(prices):
    """
    Clean raw price data:
    - Forward-fill small gaps (FX vs equity holiday mismatches)
    - Drop remaining NaNs
    """
    print("\n" + "=" * 60)
    print("Cleaning price data...")
    print("=" * 60)

    print("\nMissing values before cleaning:")
    print(prices.isnull().sum())

    prices_clean = prices.ffill(limit=3).dropna()

    print("\nMissing values after cleaning:")
    print(prices_clean.isnull().sum())
    print(f"\nFinal dataset : {len(prices_clean)} trading days")
    print(f"Rows dropped  : {len(prices) - len(prices_clean)}")

    return prices_clean

# =============================================================================
# STEP 3 — COMPUTE LOG RETURNS
# =============================================================================

def compute_log_returns(prices):
    """
    Compute daily log returns: r_t = log(P_t / P_{t-1})
    Theory: report/theoretical_background.md — Section 2 (Irle p. 35)
    Log-returns ≈ discrete returns for 1-day horizon.
    """
    print("\n" + "=" * 60)
    print("Computing log returns...")
    print("=" * 60)

    log_returns = np.log(prices / prices.shift(1)).dropna()

    print(f"Log returns: {len(log_returns)} observations")
    print("\nBasic statistics:")
    print(log_returns.describe().round(6))

    return log_returns

# =============================================================================
# STEP 4 — COMPUTE LINEAR PORTFOLIO RETURNS
# =============================================================================

def compute_portfolio_returns(log_returns, weights_dict):
    """
    Compute weighted portfolio log returns for the 6 linear positions.
    R^{d,P} = sum_i w_i * R_i   (Irle Eq. 10)
    """
    print("\n" + "=" * 60)
    print("Computing linear portfolio returns...")
    print("=" * 60)

    weights_aligned = {
        ("EURUSD" if k == "EURUSD=X" else k): v
        for k, v in weights_dict.items()
    }
    weight_series     = pd.Series(weights_aligned)[log_returns.columns]
    portfolio_returns = log_returns.dot(weight_series)
    portfolio_returns.name = "portfolio"

    print("Portfolio return stats:")
    print(portfolio_returns.describe().round(6))

    return portfolio_returns

# =============================================================================
# STEP 5 — COMPUTE INSTRUMENT P&L (IRS + STRADDLE)
# =============================================================================

def compute_instrument_pnl(prices, vix, dgs10):
    """
    Compute daily mark-to-market P&L for IRS and straddle.

    Theory: report/theoretical_background.md
      Section 1 — Risk mapping DV = f(t, Y_{t+dt}) - f(t, Y_t)  (Irle Eq. 2)
      Section 1 — Delta-Gamma-Vega-Theta for straddle             (Irle p. 82)
      Section 2 — IRS pricing via DV01                            (Irle p. 21)

    IRS:
      DV_irs_t = V_irs(r_t) - V_irs(r_{t-1})

    Straddle (30-day rolling ATM):
      DV_straddle_t = BS(S_t, K, T_t) - BS(S_{t-1}, K, T_{t-1} + 1/252)
      Strike K reset every STRADDLE_DAYS to current SPY price
    """
    # Align to common trading dates
    spy    = prices["SPY"]
    common = spy.index.intersection(vix.index).intersection(dgs10.index)
    spy   = spy.loc[common]
    vix_s = vix.loc[common].squeeze()
    dgs_s = dgs10.loc[common].squeeze()

    pnl_irs      = []
    pnl_straddle = []
    dates        = []

    K         = spy.iloc[0]   # initial ATM strike
    days_held = 0

    print("\nComputing instrument P&L (IRS + Straddle)...")

    for t in range(1, len(common)):

        # IRS P&L: DV = V(r_t) - V(r_{t-1})
        v_today, _ = price_irs(IRS_NOTIONAL, IRS_FIXED_RATE,
                                dgs_s.iloc[t]   / 100)
        v_prev,  _ = price_irs(IRS_NOTIONAL, IRS_FIXED_RATE,
                                dgs_s.iloc[t-1] / 100)
        pnl_irs_t  = v_today - v_prev

        # Straddle P&L — reset strike every STRADDLE_DAYS
        if days_held >= STRADDLE_DAYS:
            K         = spy.iloc[t]
            days_held = 0

        T_today = max((STRADDLE_DAYS - days_held) / 252, 1/252)
        T_prev  = T_today + 1/252

        p_today, _ = price_straddle(spy.iloc[t],   K, T_today,
                                     RF_RATE, vix_s.iloc[t]   / 100)
        p_prev,  _ = price_straddle(spy.iloc[t-1], K, T_prev,
                                     RF_RATE, vix_s.iloc[t-1] / 100)
        pnl_straddle_t = (p_today - p_prev) * STRADDLE_SHARES

        pnl_irs.append(pnl_irs_t)
        pnl_straddle.append(pnl_straddle_t)
        dates.append(common[t])
        days_held += 1

    pnl_df = pd.DataFrame({
        "pnl_irs"      : pnl_irs,
        "pnl_straddle" : pnl_straddle,
    }, index=dates)

    print(f"Instrument P&L computed: {len(pnl_df)} days")
    print(f"\nIRS P&L stats:")
    print(pnl_df["pnl_irs"].describe().round(2))
    print(f"\nStraddle P&L stats:")
    print(pnl_df["pnl_straddle"].describe().round(2))

    return pnl_df

# =============================================================================
# STEP 6 — COMBINE INTO TOTAL PORTFOLIO P&L
# =============================================================================

def compute_total_portfolio_pnl(log_returns, instrument_pnl,
                                 weights_dict, V0):
    """
    Combine linear P&L with IRS and straddle P&L.
    Total DV = DV_linear + DV_irs + DV_straddle

    Theory: report/theoretical_background.md — Section 1 (Irle Eq. 2)
    DV_linear = V0 * w' * R   (Irle Eq. 10)
    """
    # Align weights to column order
    weights_aligned = {
        ("EURUSD" if k == "EURUSD=X" else k): v
        for k, v in weights_dict.items()
    }
    weight_series = pd.Series(weights_aligned)[log_returns.columns]
    weights_array = weight_series.values

    # Linear P&L
    linear_pnl       = V0 * log_returns.dot(weights_array)
    linear_pnl.name  = "pnl_linear"

    # Align dates
    common             = linear_pnl.index.intersection(instrument_pnl.index)
    linear_aligned     = linear_pnl.loc[common]
    instrument_aligned = instrument_pnl.loc[common]

    total_pnl = pd.DataFrame({
        "pnl_linear"   : linear_aligned,
        "pnl_irs"      : instrument_aligned["pnl_irs"],
        "pnl_straddle" : instrument_aligned["pnl_straddle"],
    })
    total_pnl["pnl_total"] = (total_pnl["pnl_linear"] +
                               total_pnl["pnl_irs"]    +
                               total_pnl["pnl_straddle"])

    print(f"\n{'='*60}")
    print("TOTAL PORTFOLIO P&L STATS")
    print(f"{'='*60}")
    print(total_pnl.describe().round(2))

    return total_pnl

# =============================================================================
# STEP 7 — SAVE TO DISK
# =============================================================================

def save_data(prices, log_returns, portfolio_returns,
              vix, dgs10, instrument_pnl, total_pnl):
    """Save all datasets to CSV."""
    print("\n" + "=" * 60)
    print("Saving all data to disk...")
    print("=" * 60)

    files = {
        os.path.join(RAW_DIR,       "prices.csv")             : prices,
        os.path.join(RAW_DIR,       "vix.csv")                : vix,
        os.path.join(RAW_DIR,       "dgs10.csv")              : dgs10,
        os.path.join(PROCESSED_DIR, "log_returns.csv")        : log_returns,
        os.path.join(PROCESSED_DIR, "portfolio_returns.csv")  : portfolio_returns,
        os.path.join(PROCESSED_DIR, "instrument_pnl.csv")     : instrument_pnl,
        os.path.join(PROCESSED_DIR, "total_portfolio_pnl.csv"): total_pnl,
    }

    for path, df in files.items():
        df.to_csv(path)
        print(f"Saved -> {path}")

    print("\nAll files saved successfully.")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    os.makedirs(RAW_DIR,       exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # 1. Download prices, VIX, DGS10
    prices, vix, dgs10 = download_prices(TICKERS, START_DATE, END_DATE)

    # 2. Clean prices
    prices_clean = clean_prices(prices)

    # 3. Log returns
    log_returns = compute_log_returns(prices_clean)

    # 4. Linear portfolio returns
    portfolio_returns = compute_portfolio_returns(log_returns, WEIGHTS_DICT)

    # 5. Instrument P&L (IRS + straddle)
    instrument_pnl = compute_instrument_pnl(prices_clean, vix, dgs10)

    # 6. Total portfolio P&L
    total_pnl = compute_total_portfolio_pnl(
        log_returns, instrument_pnl, WEIGHTS_DICT, V0
    )

    # 7. Save everything
    save_data(prices_clean, log_returns, portfolio_returns,
              vix, dgs10, instrument_pnl, total_pnl)

    print("\n" + "=" * 60)
    print("Data download complete!")
    print("=" * 60)