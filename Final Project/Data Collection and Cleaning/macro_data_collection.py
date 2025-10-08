
from fredapi import Fred
import pandas as pd
import yfinance as yf
import datetime
import os

API_KEY = os.getenv("FRED_KEY")

start = datetime.datetime(2016, 1, 1)
end = datetime.datetime.now()

fred = Fred(API_KEY) # You need to use the actual API key here, not the .env variable.

start = datetime.datetime(2016, 1, 1)
end = datetime.datetime.now()

# Define the macro series
fred_series = {
    "FEDFUNDS": "Fed Funds Rate",
    "CPIAUCSL": "Consumer Price Index",
    "M2SL": "M2 Money Supply",
    "DGS10": "10-Year Treasury Yield",
    "DCOILWTICO": "Crude Oil Price",
    "GOLDAMGBD228NLBM": "Gold Price"
}

# Fetch data
macro_df = pd.DataFrame()
for code, name in fred_series.items():
    try:
        data = fred.get_series(code, start, end)
        macro_df[name] = data
    except Exception as e:
        print(f"Error fetching {code}: {e}")

macro_df.to_csv("macro_fred_data.csv")
print("✅ Macro data successfully saved to macro_fred_data.csv")


import yfinance as yf

tickers = ["^VIX", "^GSPC", "DX-Y.NYB", "GC=F"]  # VIX, S&P 500, USD Index, Gold

# Explicitly tell yfinance not to auto-adjust, so Adj Close is returned
data = yf.download(tickers, start="2016-01-01", end=None, auto_adjust=False)

# Use Adj Close if it exists, otherwise fallback to Close
if "Adj Close" in data.columns.get_level_values(0):
    yahoo_df = data["Adj Close"]
else:
    yahoo_df = data["Close"]

yahoo_df.to_csv("macro_yahoo.csv")
print("✅ Yahoo Finance data successfully saved to macro_yahoo.csv")
