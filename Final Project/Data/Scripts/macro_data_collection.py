from fredapi import Fred
import pandas as pd
import yfinance as yf
import datetime
import os
from pathlib import Path

# Setup paths
BASE_DIR = Path(__file__).parent.parent.parent  # Go up to project root
RAW_DATA_DIR = BASE_DIR / "Data" / "Raw Data"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)  # Create if doesn't exist

# API Key
API_KEY = os.getenv("FRED_KEY")

# Date range
start = datetime.datetime(2016, 1, 1)
end = datetime.datetime.now()

# Initialize FRED
fred = Fred(API_KEY)

# Define the macro series
fred_series = {
    "FEDFUNDS": "Fed Funds Rate",
    "CPIAUCSL": "Consumer Price Index",
    "M2SL": "M2 Money Supply",
    "DGS10": "10-Year Treasury Yield",
    "DCOILWTICO": "Crude Oil Price",
    "GOLDAMGBD228NLBM": "Gold Price"
}

# Fetch FRED data
print("Fetching FRED data...")
macro_df = pd.DataFrame()
for code, name in fred_series.items():
    try:
        data = fred.get_series(code, start, end)
        macro_df[name] = data
        print(f"  ✓ {name}")
    except Exception as e:
        print(f"  ✗ Error fetching {code}: {e}")

# Save FRED data
fred_output = RAW_DATA_DIR / "macro_fred_data.csv"
macro_df.to_csv(fred_output)
print(f"✅ FRED data saved to {fred_output}")

# Fetch Yahoo Finance data
print("\nFetching Yahoo Finance data...")
tickers = ["^VIX", "^GSPC", "DX-Y.NYB", "GC=F"]  # VIX, S&P 500, USD Index, Gold

# Explicitly tell yfinance not to auto-adjust, so Adj Close is returned
data = yf.download(tickers, start="2016-01-01", end=None, auto_adjust=False)

# Use Adj Close if it exists, otherwise fallback to Close
if "Adj Close" in data.columns.get_level_values(0):
    yahoo_df = data["Adj Close"]
else:
    yahoo_df = data["Close"]

# Save Yahoo data
yahoo_output = RAW_DATA_DIR / "macro_yahoo_data.csv"
yahoo_df.to_csv(yahoo_output)
print(f"✅ Yahoo Finance data saved to {yahoo_output}")

print(f"\n📂 All raw data saved to: {RAW_DATA_DIR}")
