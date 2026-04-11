# =============================================================================
# config.py
# Portfolio configuration — edit this file to change the portfolio
# Imported by download_data.py and all VaR scripts
# =============================================================================

# --- Linear positions ---
TICKERS = {
    "SPY":      "S&P 500 ETF (US Equity)",
    "IEF":      "iShares 7-10Y Treasury ETF (Bonds)",
    "GLD":      "SPDR Gold ETF (Commodity)",
    "EURUSD=X": "EUR/USD Exchange Rate (FX)"
}

WEIGHTS_DICT = {
    "SPY":      1/4,
    "IEF":      1/4,
    "GLD":      1/4,
    "EURUSD=X": 1/4
}

# --- Date range ---
START_DATE = "2006-01-01"
END_DATE   = "2024-12-31"

# --- Portfolio notional ---
V0 = 1_000_000

# --- IRS settings ---
IRS_NOTIONAL   = 1_000_000
IRS_FIXED_RATE = 0.03        # fixed rate paid (3%)

# --- Straddle settings ---
STRADDLE_DAYS       = 30     # rolling window in trading days
STRADDLE_CONTRACTS  = 20     # number of contracts
SHARES_PER_CONTRACT = 100    # standard contract size
STRADDLE_SHARES     = STRADDLE_CONTRACTS * SHARES_PER_CONTRACT  # 2000
RF_RATE             = 0.05   # risk-free rate (constant approximation)

# --- Output paths ---
import os
RAW_DIR       = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")