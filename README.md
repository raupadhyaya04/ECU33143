# ECU33143 – Big Data in Economics Research Project

This repository contains my completed research project for **ECU33143: Introduction to Big Data in Economics**, investigating the pricing dynamics of cryptocurrencies through both macroeconomic fundamentals and social media sentiment analysis.

---

## 🔍 Research Overview

This project addresses two interrelated questions:

> **1. Do cryptocurrencies respond systematically to macroeconomic indicators?**  
> **2. Does social media sentiment predict returns or improve risk forecasting?**

### Key Finding
**Social media sentiment does not predict cryptocurrency returns** (weak Granger causality: only 1/10 assets significant, p > 0.70 for most), **but dramatically improves tail-risk forecasting** (Conditional VaR achieved 100% Kupiec test pass rate vs 0% for Historical VaR).

This demonstrates that **sentiment operates as a risk factor rather than a leading indicator**—capturing latent market stress that manifests in tail events rather than mean returns.

---

## 📊 Methodology

### Part 1: Macro-Crypto Analysis
- **Vector Autoregression (VAR)** with Impulse Response Functions (IRFs) to identify dynamic macro relationships
- **Granger Causality Tests** to assess temporal precedence between macro variables and crypto returns
- Analysis across 10 cryptocurrencies: BTC, ETH, BNB, ADA, XRP, SOL, DOGE, DOT, AVAX, LINK

### Part 2: Sentiment-Crypto Analysis  
- **Reddit Sentiment Analysis** using VADER on 25+ cryptocurrency-focused subreddits
- **Granger Causality Tests** for bidirectional sentiment-return relationships
- **Value-at-Risk (VaR) Models**: Historical vs Sentiment-Conditional (via quantile regression)
- **Kupiec Backtesting** at 95% and 99% confidence levels to validate risk model accuracy

### Results Summary
- **Macro Integration**: Heterogeneous and limited; DOT and ADA showed strongest sensitivities to VIX and M2
- **Sentiment Causality**: Weak predictive power (9/10 assets p > 0.05)
- **VaR Performance**: Conditional VaR exception rates matched theoretical expectations (5.19% vs 5.00% at 95%; 1.16% vs 1.00% at 99%)
- **Practical Implication**: Use sentiment for risk management frameworks, not directional trading strategies

---

## 📊 Data Collection

Data is collected programmatically from multiple public APIs:

| Source | Type of Data | Access Method | Purpose |
|--------|---------------|----------------|---------|
| **Binance API** | Cryptocurrency OHLCV data (8 years, 10 assets) | `python-binance` | Price analysis |
| **FRED API** | Macroeconomic indicators (rates, CPI, M2, etc.) | `fredapi` | Macro fundamentals |
| **Yahoo Finance** | Financial indices (S&P 500, VIX, Gold, USD Index, Oil) | `yfinance` | Risk/market factors |
| **Reddit API (PRAW)** | Social media sentiment from 25+ crypto subreddits | `praw` | Sentiment analysis |

### To Reproduce Results:
1. Obtain API keys:
   - [FRED API key](https://fred.stlouisfed.org/docs/api/api_key.html)
   - [Reddit API credentials](https://www.reddit.com/prefs/apps) (Client ID, Client Secret)
2. Set environment variables:
   - export FRED_API_KEY="your_key_here"
   - export REDDIT_CLIENT_ID="your_id_here"
   - export REDDIT_CLIENT_SECRET="your_secret_here"

3. Run data collection scripts in order:
   - cd "Final Project"
   - ./Data/Data Collection and Cleaning/crypto_data_collection.py
   - ./Data/Data Collection and Cleaning/macro_data_collection.py
   - ./Data/Data Collection and Cleaning/data_cleaning.py
   - ./Data/Data Collection and Cleaning/reddit_sentiment_data_collection.py

---

## 🔬 Key Technical Components

### Macro Analysis
- **Stationarity transformations**: Log-differencing for price-like variables, first-differencing for rates
- **Mixed-frequency alignment**: Forward-fill interpolation for daily-to-monthly macro data
- **Newey-West HAC standard errors**: Robust to heteroskedasticity and autocorrelation
- **AIC-based lag selection**: Optimal lag determination for VAR models

### Sentiment Analysis
- **VADER Sentiment**: Polarity scoring of Reddit posts/comments
- **Data sparsity handling**: Forward-fill + 7-day moving average for irregular posting
- **Quantile Regression**: Conditional VaR incorporating sentiment features
- **Kupiec POF Test**: Proportion-of-failures backtesting for VaR validation

---

## 📈 Visualizations

The project includes 24 publication-quality figures:
- **Figures 1-10**: Impulse Response Functions (macro shocks → crypto returns)
- **Figure 11**: Macro Granger Causality Heatmap
- **Figure 12**: Comprehensive Sentiment Analysis Dashboard (Granger + VaR performance)
- **Figure 13**: Detailed Sentiment Granger Results
- **Figure 14**: VaR Model Performance Comparison
- **Figures 15-24**: Individual VaR Backtests (95% & 99% confidence) for all assets

---

## 🚫 Data Policy

**Raw data files (`.csv`) are excluded from version control** for the following reasons:
- **Reproducibility**: All data can be regenerated using provided scripts
- **Repository size**: Multi-year datasets are large (100+ MB combined)
- **API terms**: Respects data provider redistribution policies
- **Freshness**: Ensures users get latest data when reproducing analysis

To obtain data: Run collection scripts with your own API credentials.

---

## 📚 Dependencies

pip install -r requirements.txt

---

## 🎓 Academic Context

**Course**: ECU33143 – Introduction to Big Data in Economics  
**Institution**: Trinity College Dublin
**Completion**: December 2025

---

## 🔮 Potential Future Extensions:

Planned extensions for dissertation research:
- Multi-platform sentiment integration
- Machine learning approaches (LSTM, Transformer models)
- Out-of-sample validation and trading strategy backtesting
- Regime-switching models for time-varying relationships
- High-frequency intraday analysis

---

## ⚖️ License

This project is for academic purposes. Data sources have their own terms of use.  
Code is available under MIT License for educational and research purposes.
