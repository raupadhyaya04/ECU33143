import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Config
CRYPTO_SYMBOLS = ['ADA', 'AVAX', 'BNB', 'BTC', 'DOGE', 'DOT', 'ETH', 'LINK', 'SOL', 'XRP']
MACRO_VARIABLES = [
    'Fed Funds Rate', '10-Year Treasury Yield', 'Consumer Price Index',
    'M2 Money Supply', 'Crude Oil Price', 'GC=F', 'DX-Y.NYB', '^GSPC', '^VIX'
]

N_SPLITS = 5
N_ESTIMATORS = 100
MAX_DEPTH = 10
RANDOM_STATE = 42

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "Data"
CRYPTO_DATA_DIR = DATA_DIR / "Crypto Data"
MACRO_DATA_DIR = DATA_DIR / "Macro Data"
MACRO_ANALYSIS_DIR = BASE_DIR / "Crypto-Macro"
RESULTS_DIR = Path(__file__).parent / "Results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def reconstruct_macro_crypto_data(symbol):
    """Load macro + crypto data, properly handling monthly FRED data."""
    print(f"  📂 Loading data...")
    
    try:
        # Load FRED (monthly) and Yahoo (daily)
        fred = pd.read_csv(MACRO_DATA_DIR / "macro_fred_data_filled.csv",
                          index_col=0, parse_dates=True)
        yahoo = pd.read_csv(MACRO_DATA_DIR / "macro_yahoo_data_filled.csv",
                           index_col=0, parse_dates=True)
        
        # Forward-fill FRED to END of each month, then resample to daily
        # This ensures monthly data propagates through the whole month
        fred_daily = fred.resample('D').ffill()  # No limit - fill entire month
        
        # Merge with Yahoo (already daily)
        macro_df = pd.concat([fred_daily, yahoo], axis=1)
        macro_df = macro_df.sort_index()
        
        # Drop any remaining NaNs
        macro_df = macro_df.dropna()
        
        print(f"    ✓ Macro data: {len(macro_df)} rows (daily)")
        
        # Load crypto
        print(f"    Loading {symbol} price data...")
        crypto_df = pd.read_csv(CRYPTO_DATA_DIR / f"{symbol}USDT.csv")
        
        if 'timestamp' in crypto_df.columns:
            try:
                crypto_df['timestamp'] = pd.to_datetime(crypto_df['timestamp'], unit='ms')
            except:
                crypto_df['timestamp'] = pd.to_datetime(crypto_df['timestamp'])
        
        crypto_df.set_index('timestamp', inplace=True)
        
        # Daily aggregation
        crypto_daily = crypto_df['close'].resample('D').last()
        crypto_daily_ret = np.log(crypto_daily).diff()
        crypto_daily_ret = crypto_daily_ret.to_frame(name=f'{symbol}_ret')
        
        # Merge
        merged = pd.merge(macro_df, crypto_daily_ret, left_index=True, right_index=True, how='inner')
        merged = merged.dropna()
        
        print(f"    ✓ Final data: {len(merged)} rows")
        
        return merged if len(merged) >= 100 else None
        
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def validate_single_crypto(symbol):
    """Run Random Forest validation."""
    print(f"\n{'='*60}")
    print(f"🔍 Validating {symbol}")
    print(f"{'='*60}")
    
    df = reconstruct_macro_crypto_data(symbol)
    
    if df is None:
        print(f"  ❌ Skipping - insufficient data")
        return None
    
    # Prepare data
    available_vars = [v for v in MACRO_VARIABLES if v in df.columns]
    X = df[available_vars].copy()
    y = df[f'{symbol}_ret'].copy()
    
    print(f"  📊 Dataset: {len(X)} observations, {len(X.columns)} features")
    print(f"    Date range: {X.index.min().date()} to {X.index.max().date()}")
    
    # Time series CV
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    rf = RandomForestRegressor(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
                               random_state=RANDOM_STATE, n_jobs=-1)
    
    cv_scores = []
    all_importances = []
    
    print(f"  🔄 Running {N_SPLITS}-fold time series CV...")
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        rf.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = rf.predict(X.iloc[test_idx])
        r2 = r2_score(y.iloc[test_idx], pred)
        cv_scores.append(r2)
        all_importances.append(rf.feature_importances_)
        print(f"    Fold {fold}: R²={r2:.3f}")
    
    # Results
    avg_r2 = np.mean(cv_scores)
    avg_importance = np.mean(all_importances, axis=0)
    
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': avg_importance
    }).sort_values('importance', ascending=False)
    
    print(f"\n  📊 Average R²: {avg_r2:.3f}")
    print(f"  🎯 Feature Importance:")
    for _, row in importance_df.iterrows():
        print(f"    {row['feature']:30s} {row['importance']:.4f}")
    
    importance_df.to_csv(RESULTS_DIR / f"{symbol}_feature_importance.csv", index=False)
    
    return {
        'symbol': symbol,
        'r2_mean': avg_r2,
        'n_samples': len(X),
        'top_feature': importance_df.iloc[0]['feature'],
        'top_importance': importance_df.iloc[0]['importance']
    }


def compare_with_granger():
    """Compare with Granger causality results."""
    print(f"\n{'='*60}")
    print(f"📊 COMPARING WITH GRANGER CAUSALITY")
    print(f"{'='*60}")
    
    granger_file = MACRO_ANALYSIS_DIR / "Results" / "Analysis" / "granger_causality_results.csv"
    
    if not granger_file.exists():
        print("  ⚠️ Granger results not found, skipping")
        return
    
    try:
        granger_df = pd.read_csv(granger_file)
        sig = granger_df[granger_df['Significant'] == '✅']
        print(f"\nGranger significant: {len(sig)}/{len(granger_df)}")
        
        if len(sig) > 0:
            counts = sig.groupby('Macro Variable').size().sort_values(ascending=False)
            print("\nMost influential (Granger):")
            for var, count in counts.head(5).items():
                print(f"  {var:30s} {count} cryptos")
        
        # Compare
        comparisons = []
        for symbol in CRYPTO_SYMBOLS:
            imp_file = RESULTS_DIR / f"{symbol}_feature_importance.csv"
            if not imp_file.exists():
                continue
            
            imp_df = pd.read_csv(imp_file)
            top3_ml = set(imp_df.head(3)['feature'].values)
            
            granger_symbol = granger_df[granger_df['Crypto'] == symbol]
            sig_granger = granger_symbol[granger_symbol['Significant'] == '✅']
            sig_vars = set(sig_granger['Macro Variable'].values)
            
            overlap = top3_ml & sig_vars
            
            comparisons.append({
                'Crypto': symbol,
                'Top 3 ML': ', '.join(top3_ml),
                'Significant Granger': ', '.join(sig_vars) if sig_vars else 'None',
                'Overlap': ', '.join(overlap) if overlap else 'None',
                'Overlap Count': len(overlap)
            })
        
        if comparisons:
            comp_df = pd.DataFrame(comparisons)
            comp_df.to_csv(RESULTS_DIR / "ml_granger_comparison.csv", index=False)
            print(f"\n✅ Comparison saved")
            print(f"Agreement: {comp_df['Overlap Count'].sum()} overlaps")
    
    except Exception as e:
        print(f"  ⚠️ Error: {e}")


def run_all_validations():
    """Run ML validation for all cryptos."""
    print(f"\n{'='*60}")
    print(f"MACRO-CRYPTO ML VALIDATION")
    print(f"{'='*60}")
    print(f"Symbols: {len(CRYPTO_SYMBOLS)}")
    print(f"Output: {RESULTS_DIR}")
    print(f"{'='*60}")
    
    results = []
    for symbol in CRYPTO_SYMBOLS:
        result = validate_single_crypto(symbol)
        if result:
            results.append(result)
    
    if results:
        summary = pd.DataFrame(results)
        summary.to_csv(RESULTS_DIR / "ml_validation_summary.csv", index=False)
        
        print(f"\n{'='*60}")
        print(f"SUMMARY")
        print(f"{'='*60}")
        print(summary.to_string(index=False))
        print(f"\nAverage R²: {summary['r2_mean'].mean():.3f}")
    
    compare_with_granger()
    
    print(f"\n{'='*60}")
    print(f"✅ COMPLETE - Results in {RESULTS_DIR}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run_all_validations()
