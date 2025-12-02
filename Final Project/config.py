import os
from pathlib import Path

# ============================================
# AUTO-DETECT PROJECT ROOT
# ============================================
# This finds the project root no matter where the script is called from
def find_project_root():
    """Find project root by looking for marker files/folders."""
    current = Path.cwd()
    
    # Traverse up until we find the project root indicators
    for parent in [current] + list(current.parents):
        # Check for characteristic files/folders of your project
        if (parent / "Data").exists() or (parent / "Crypto-Macro").exists():
            return parent
    
    # Fallback: assume current directory
    return current

BASE_DIR = find_project_root()

# ============================================
# PATHS (all relative to auto-detected root)
# ============================================
DATA_DIR = BASE_DIR / "Data"
CRYPTO_DATA_DIR = DATA_DIR / "Crypto Data"
MACRO_DATA_DIR = DATA_DIR / "Macro Data"
SENTIMENT_DATA_DIR = DATA_DIR / "Sentiment Data"

# Results - relative to segment folders
CRYPTO_MACRO_DIR = BASE_DIR / "Crypto-Macro"
CRYPTO_SENTIMENT_DIR = BASE_DIR / "Crypto-Sentiment"
VALIDATION_DIR = BASE_DIR / "Validation"

# ============================================
# CRYPTOCURRENCY SYMBOLS
# ============================================
CRYPTO_SYMBOLS = ['ADA', 'AVAX', 'BNB', 'BTC', 'DOGE', 'DOT', 'ETH', 'LINK', 'SOL', 'XRP']

CRYPTO_TICKERS = {
    'ADA': 'ADAUSDT', 'AVAX': 'AVAXUSDT', 'BNB': 'BNBUSDT', 'BTC': 'BTCUSDT',
    'DOGE': 'DOGEUSDT', 'DOT': 'DOTUSDT', 'ETH': 'ETHUSDT',
    'LINK': 'LINKUSDT', 'SOL': 'SOLUSDT', 'XRP': 'XRPUSDT'
}

# ============================================
# MACRO VARIABLES
# ============================================
MACRO_LOG_DIFF_COLS = [
    'Consumer Price Index', 'M2 Money Supply', 'GC=F', '^GSPC', 'DX-Y.NYB'
]

MACRO_FIRST_DIFF_COLS = [
    'Fed Funds Rate', '10-Year Treasury Yield', 'Crude Oil Price', '^VIX'
]

MACRO_CANDIDATES = [
    'Fed Funds Rate', '10-Year Treasury Yield', 'Consumer Price Index', 
    'M2 Money Supply', 'Crude Oil Price', 'GC=F', 'DX-Y.NYB', '^GSPC', '^VIX'
]

MACRO_VAR_SELECTION = ['^GSPC', '^VIX', 'M2 Money Supply']

# ============================================
# MODEL PARAMETERS
# ============================================
OLS_LAGS = 3
GRANGER_MAX_LAG = 3
VAR_MAX_LAG = 3
IRF_HORIZON = 10

VAR_ROLLING_WINDOW = 250
VAR_ALPHAS = [0.95, 0.99]

SENTIMENT_SMOOTH_DAYS = 7
SENTIMENT_ROLLING_CORR_WINDOW = 30

# ============================================
# VISUALIZATION SETTINGS
# ============================================
FIGSIZE_STANDARD = (12, 6)
FIGSIZE_WIDE = (14, 6)
FIGURE_DPI = 300

HEATMAP_CMAP = 'RdYlGn_r'
HEATMAP_PVALUE_MAX = 0.10

# ============================================
# ML VALIDATION SETTINGS
# ============================================
ML_TEST_SIZE = 0.2
ML_N_SPLITS = 5
ML_RANDOM_STATE = 42
ML_N_ESTIMATORS = 100