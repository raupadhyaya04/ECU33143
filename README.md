# ECU33143 – Big Data in Economics Project

This repository contains my project for my **Big Data in Economics** module.  
The research investigates whether and how **macroeconomic indicators influence cryptocurrency price movements**, and which factors appear to drive these relationships over time.

---

## 🔍 Research Overview
The core question explored is:

> *Do macroeconomic variables such as interest rates, inflation, and market risk sentiment have a measurable impact on cryptocurrency prices?*

The analysis combines **macroeconomic**, **financial market**, and **cryptocurrency** data sources to identify correlations, potential causal links, and shifts in sensitivity across time.

---

## 📊 Data Collection
Data is collected programmatically from public APIs:

| Source | Type of Data | Access Method |
|--------|---------------|----------------|
| **Binance API** | Cryptocurrency prices (historical + live) | `python-binance` |
| **FRED API** | Macroeconomic indicators (interest rates, inflation, money supply, etc.) | `fredapi` |
| **Yahoo Finance API** | Financial indices and commodities (S&P 500, VIX, Gold, USD Index) | `yfinance` |

To reproduce results:
1. Obtain a [FRED API key](https://fred.stlouisfed.org/docs/api/api_key.html).
2. Run the data collection scripts to generate `.csv` datasets locally.
