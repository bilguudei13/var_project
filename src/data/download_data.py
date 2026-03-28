# =============================================================================
# download_data.py
# Downloads and saves price data for the VaR portfolio
#
# Portfolio:
#   SPY   - S&P 500 ETF          (US Equity)
#   EWG   - iShares Germany ETF  (European Equity / DAX proxy)
#   EWJ   - iShares Japan ETF    (Asian Equity / Nikkei proxy)
#   IEF   - iShares 7-10Y Treasury Bond ETF  (US Bonds)
#   GLD   - SPDR Gold Shares     (Commodity)
#   EURUSD=X - EUR/USD exchange rate (FX)
#
# Period: 2006-01-01 to 2024-12-31
# =============================================================================

from tracemalloc import start

import yfinance as yf
import pandas as pd
import os
import sys
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

# Portfolio weights (equal weighting)
WEIGHTS = {
    "SPY":      1/6,
    "EWG":      1/6,
    "EWJ":      1/6,
    "IEF":      1/6,
    "GLD":      1/6,
    "EURUSD=X": 1/6
}

# Output paths
RAW_DIR       = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")

# =============================================================================
# STEP 1 — DOWNLOAD RAW PRICES
# =============================================================================

def download_prices(tickers, start, end):
    """
    Download adjusted closing prices for all tickers.
    Returns a DataFrame with dates as index and tickers as columns.
    """
    print("=" * 60)
    print("Downloading price data from Yahoo Finance...")
    print(f"Period : {start} to {end}")
    print(f"Tickers: {list(tickers.keys())}")
    print("=" * 60)

    data = yf.download(
        tickers  = list(tickers.keys()),
        start    = start,
        end      = end,
        auto_adjust = True,   # adjusts for splits and dividends
        progress = True
    )

    # Download VIX (implied volatility — risk factor for straddle)
    print("\nDownloading VIX...")
    vix = yf.download("^VIX", start=start, end=end,
                   auto_adjust=True, progress=False)["Close"]
    vix.name = "VIX"
    # Download 10Y Treasury yield from FRED (risk factor for IRS)
    print("Downloading DGS10 from FRED...")
    import pandas_datareader.data as web
    dgs10 = web.DataReader("DGS10", "fred", start, end)
    dgs10 = dgs10["DGS10"]
    dgs10.name = "DGS10"
    dgs10 = dgs10.ffill()  # forward-fill weekend/holiday gaps
    dgs10 = dgs10.dropna()

    # Keep only closing prices
    prices = data["Close"]

    # Rename EURUSD=X to EURUSD for cleaner column names
    prices = prices.rename(columns={"EURUSD=X": "EURUSD"})

    print(f"\nDownloaded {len(prices)} trading days")
    print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"Columns: {list(prices.columns)}")

    return prices, vix, dgs10


# =============================================================================
# STEP 2 — CLEAN THE DATA
# =============================================================================

def clean_prices(prices):
    """
    Clean the raw price data:
    - Report missing values
    - Forward-fill small gaps (e.g. FX closed on days equity is open)
    - Drop any remaining NaNs
    """
    print("\n" + "=" * 60)
    print("Cleaning price data...")
    print("=" * 60)

    # Report missing values before cleaning
    missing = prices.isnull().sum()
    print("\nMissing values per ticker (before cleaning):")
    print(missing)

    # Forward-fill missing values (max 3 consecutive days)
    # This handles FX holidays vs equity holidays
    prices_clean = prices.ffill(limit=3)

    # Drop rows where any ticker is still missing
    prices_clean = prices_clean.dropna()

    missing_after = prices_clean.isnull().sum()
    print("\nMissing values per ticker (after cleaning):")
    print(missing_after)

    print(f"\nFinal dataset: {len(prices_clean)} trading days")
    print(f"Rows dropped : {len(prices) - len(prices_clean)}")

    return prices_clean

# =============================================================================
# STEP 3 — COMPUTE LOG RETURNS
# =============================================================================

def compute_log_returns(prices):
    """
    Compute daily log returns: r_t = log(P_t / P_{t-1})
    Log returns are used because:
    - They are approximately normally distributed
    - They are time-additive (multi-period returns = sum of daily returns)
    - They are the standard in financial risk modelling
    """
    print("\n" + "=" * 60)
    print("Computing log returns...")
    print("=" * 60)

    log_returns = prices.apply(lambda x: (x / x.shift(1)).transform("log"))
    log_returns = log_returns.dropna()

    print(f"Log returns computed: {len(log_returns)} observations")
    print("\nBasic statistics:")
    print(log_returns.describe().round(6))

    return log_returns

# =============================================================================
# STEP 4 — COMPUTE PORTFOLIO RETURNS
# =============================================================================

def compute_portfolio_returns(log_returns, weights):
    """
    Compute weighted portfolio log returns.
    weights: dict of {ticker: weight}
    """
    print("\n" + "=" * 60)
    print("Computing portfolio returns...")
    print("=" * 60)

    # Align weights to column order
    # Note: EURUSD=X was renamed to EURUSD
    weights_aligned = {
        ("EURUSD" if k == "EURUSD=X" else k): v
        for k, v in weights.items()
    }

    weight_series = pd.Series(weights_aligned)
    weight_series = weight_series[log_returns.columns]  # align order

    # Weighted sum of returns
    portfolio_returns = log_returns.dot(weight_series)
    portfolio_returns.name = "portfolio"

    print(f"Portfolio return stats:")
    print(portfolio_returns.describe().round(6))

    return portfolio_returns

# =============================================================================
# STEP 5 — SAVE TO DISK
# =============================================================================

def save_data(prices, log_returns, portfolio_returns, vix, dgs10):
    """
    Save all datasets to CSV files.
    """
    print("\n" + "=" * 60)
    print("Saving data to disk...")
    print("=" * 60)

    # Raw prices
    prices_path = os.path.join(RAW_DIR, "prices.csv")
    prices.to_csv(prices_path)
    print(f"Saved raw prices     -> {prices_path}")

    # Log returns per asset
    returns_path = os.path.join(PROCESSED_DIR, "log_returns.csv")
    log_returns.to_csv(returns_path)
    print(f"Saved log returns    -> {returns_path}")

    # Portfolio returns
    portfolio_path = os.path.join(PROCESSED_DIR, "portfolio_returns.csv")
    portfolio_returns.to_csv(portfolio_path)
    print(f"Saved portfolio      -> {portfolio_path}")

    # Save VIX
    vix_path = os.path.join(RAW_DIR, "vix.csv")
    vix.to_csv(vix_path)
    print(f"Saved VIX              -> {vix_path}")

    # Save DGS10
    dgs10_path = os.path.join(RAW_DIR, "dgs10.csv")
    dgs10.to_csv(dgs10_path)
    print(f"Saved DGS10            -> {dgs10_path}")

    print("\nAll files saved successfully.")

# =============================================================================
# MAIN
# =============================================================================

def compute_instrument_pnl(prices, vix, dgs10):
    """
    Compute daily P&L for IRS and straddle.
    Theory: report/theoretical_background.md — Section 1 (Irle p. 82)

    IRS:      DV_irs_t      = V_irs(r_t) - V_irs(r_{t-1})
    Straddle: DV_straddle_t = BS(S_t, K, T_t) - BS(S_{t-1}, K, T_{t-1}+1/252)
    Strike K reset every 30 days to ATM (K = SPY price at reset date)
    """
    # Settings
    NOTIONAL   = 1_000_000
    FIXED_RATE = 0.03
    RF_RATE    = 0.05       # risk-free rate (constant approximation)
    STRADDLE_DAYS = 30      # rolling window in trading days
    STRADDLE_CONTRACTS = 20    # number of option contracts
    SHARES_PER_CONTRACT = 100  # standard options contract size
    STRADDLE_SHARES = STRADDLE_CONTRACTS * SHARES_PER_CONTRACT  # = 2000 shares

    # Align all series to common dates
    spy    = prices["SPY"]
    common = spy.index.intersection(vix.index).intersection(dgs10.index)
    spy    = spy.loc[common]
    vix_s  = vix.loc[common].squeeze()
    dgs_s  = dgs10.loc[common].squeeze()

    pnl_irs      = []
    pnl_straddle = []
    dates        = []

    # Initial straddle strike = first SPY price
    K       = spy.iloc[0]
    days_held = 0

    print("\nComputing instrument P&L...")

    for t in range(1, len(common)):

        # --- IRS P&L ---
        v_today, _ = price_irs(NOTIONAL, FIXED_RATE, dgs_s.iloc[t]   / 100)
        v_prev,  _ = price_irs(NOTIONAL, FIXED_RATE, dgs_s.iloc[t-1] / 100)
        pnl_irs_t  = v_today - v_prev

        # --- Straddle P&L ---
        # Reset strike every 30 days (rolling ATM)
        if days_held >= STRADDLE_DAYS:
            K         = spy.iloc[t]   # new ATM strike
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


if __name__ == "__main__":

    # Create output directories if they don't exist
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Run pipeline
    prices, vix, dgs10 = download_prices(TICKERS, START_DATE, END_DATE)
    prices_clean       = clean_prices(prices)
    log_returns        = compute_log_returns(prices_clean)
    portfolio_returns  = compute_portfolio_returns(log_returns, WEIGHTS)
    # Compute and save instrument P&L
    vix_raw   = pd.read_csv(os.path.join(RAW_DIR, "vix.csv"),
                         index_col=0, parse_dates=True)
    dgs10_raw = pd.read_csv(os.path.join(RAW_DIR, "dgs10.csv"),
                         index_col=0, parse_dates=True)

    instrument_pnl = compute_instrument_pnl(prices_clean, vix_raw, dgs10_raw)

    pnl_path = os.path.join(PROCESSED_DIR, "instrument_pnl.csv")
    instrument_pnl.to_csv(pnl_path)
    print(f"\nInstrument P&L saved -> {pnl_path}")
    # Save everything
    save_data(prices_clean, log_returns, portfolio_returns, vix, dgs10)

    print("\n" + "=" * 60)
    print("Data download complete!")
    print("=" * 60)