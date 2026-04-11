# =============================================================================
# download_data.py
# Downloads and saves RAW price data only
# Run this once, or when you need to refresh market data
#
# Outputs:
#   data/raw/prices.csv
#   data/raw/vix.csv
#   data/raw/dgs10.csv
# =============================================================================

import os
import sys
import numpy as np
import pandas as pd
import yfinance as yf
import pandas_datareader.data as web

sys.path.append("src/data")
from config import TICKERS, START_DATE, END_DATE, RAW_DIR, PROCESSED_DIR

def download_prices(tickers, start, end):
    print("=" * 60)
    print("Downloading price data from Yahoo Finance...")
    print(f"Period : {start} to {end}")
    print(f"Tickers: {list(tickers.keys())}")
    print("=" * 60)

    data   = yf.download(list(tickers.keys()), start=start, end=end,
                          auto_adjust=True, progress=True)
    prices = data["Close"].rename(columns={"EURUSD=X": "EURUSD"})
    print(f"Downloaded {len(prices)} days | Columns: {list(prices.columns)}")

    print("\nDownloading VIX...")
    vix = yf.download("^VIX", start=start, end=end,
                       auto_adjust=True, progress=False)["Close"]
    vix.name = "VIX"

    print("Downloading DGS10 from FRED...")
    dgs10 = web.DataReader("DGS10", "fred", start, end)["DGS10"]
    dgs10 = dgs10.ffill().dropna()
    dgs10.name = "DGS10"

    return prices, vix, dgs10

def clean_prices(prices):
    print("\nCleaning prices...")
    prices_clean = prices.ffill(limit=3).dropna()
    print(f"Final: {len(prices_clean)} days | Dropped: {len(prices) - len(prices_clean)}")
    return prices_clean

def save_raw(prices, vix, dgs10):
    os.makedirs(RAW_DIR, exist_ok=True)
    prices.to_csv(os.path.join(RAW_DIR, "prices.csv"))
    vix.to_csv(os.path.join(RAW_DIR,    "vix.csv"))
    dgs10.to_csv(os.path.join(RAW_DIR,  "dgs10.csv"))
    print(f"Saved -> data/raw/prices.csv, vix.csv, dgs10.csv")

if __name__ == "__main__":
    prices, vix, dgs10 = download_prices(TICKERS, START_DATE, END_DATE)
    prices_clean       = clean_prices(prices)
    save_raw(prices_clean, vix, dgs10)
    print("\nRaw data download complete!")