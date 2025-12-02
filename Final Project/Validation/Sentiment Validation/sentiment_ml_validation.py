import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

# ============================================
# CONFIG
# ============================================

CRYPTO_SYMBOLS = ['BTC', 'ETH', 'BNB', 'SOL', 'ADA']
QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]

N_SPLITS = 5
N_ESTIMATORS = 100
RANDOM_STATE = 42

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "Data"
CRYPTO_DATA_DIR = DATA_DIR / "Crypto Data"
SENTIMENT_DATA_DIR = DATA_DIR / "Sentiment Data"
RESULTS_DIR = Path(__file__).parent / "Results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================
# QUANTILE PREDICTION
# ============================================

def quantile_loss(y_true, y_pred, quantile):
    """Pinball loss for quantile regression."""
    error = y_true - y_pred
    return np.mean(np.maximum(quantile * error, (quantile - 1) * error))


# ============================================
# DATA LOADING
# ============================================

def load_data_with_sentiment(symbol):
    """Load crypto returns + sentiment features."""
    
    # Load crypto
    crypto_file = CRYPTO_DATA_DIR / f"{symbol}USDT.csv"
    if not crypto_file.exists():
        print(f"    ❌ Crypto file not found: {crypto_file.name}")
        return None, None
    
    crypto_df = pd.read_csv(crypto_file)
    
    try:
        crypto_df['timestamp'] = pd.to_datetime(crypto_df['timestamp'], unit='ms')
    except:
        crypto_df['timestamp'] = pd.to_datetime(crypto_df['timestamp'])
    
    crypto_df = crypto_df.set_index('timestamp').sort_index()
    crypto_daily = crypto_df['close'].resample('D').last()
    returns = np.log(crypto_daily).diff().to_frame(name='returns')
    
    print(f"    Crypto: {len(returns)} daily observations")
    
    # Load sentiment - YOUR FILE PATTERN
    sent_file = SENTIMENT_DATA_DIR / f"{symbol}USDT_reddit_sentiment.csv"
    
    if not sent_file.exists():
        print(f"    ❌ Sentiment file not found: {sent_file.name}")
        return None, None
    
    print(f"    Sentiment: {sent_file.name} ✅")
    
    sent_df = pd.read_csv(sent_file)
    
    # Find date column
    date_col = None
    for col in ['date', 'Date', 'timestamp', 'time', 'Time', 'created_utc']:
        if col in sent_df.columns:
            date_col = col
            break
    
    if date_col is None:
        print(f"    ❌ No date column found. Columns: {list(sent_df.columns)}")
        return None, None
    
    print(f"    Using date column: '{date_col}'")
    
    # Parse dates
    sent_df[date_col] = pd.to_datetime(sent_df[date_col], errors='coerce')
    sent_df = sent_df.dropna(subset=[date_col])
    sent_df = sent_df.set_index(date_col).sort_index()
    
    # Resample to daily and get numeric features only
    sent_daily = sent_df.resample('D').mean()
    sent_daily = sent_daily.select_dtypes(include=[np.number])
    
    print(f"    Sentiment features: {len(sent_daily.columns)} numeric columns")
    print(f"      Features: {list(sent_daily.columns)}")
    
    if len(sent_daily.columns) == 0:
        print(f"    ❌ No numeric sentiment features")
        return None, None
    
    # Merge
    data = pd.merge(sent_daily, returns, left_index=True, right_index=True, how='inner')
    print(f"    After merge: {len(data)} observations")
    
    data = data.dropna()
    print(f"    After dropna: {len(data)} observations")
    
    if len(data) < 200:
        print(f"    ❌ Insufficient data (need 200+, have {len(data)})")
        return None, None
    
    X_with_sent = data.drop(columns=['returns'])
    y = data['returns']
    
    print(f"    ✅ Ready: {len(y)} obs, {len(X_with_sent.columns)} features")
    
    return (X_with_sent, y), (pd.DataFrame(index=y.index), y)


# ============================================
# VALIDATION
# ============================================

def validate_quantile_prediction(symbol):
    """Compare quantile prediction WITH vs WITHOUT sentiment."""
    print(f"\n{'='*60}")
    print(f"🔍 Quantile validation: {symbol}")
    print(f"{'='*60}")
    
    data_with, data_without = load_data_with_sentiment(symbol)
    
    if data_with is None:
        print("  ❌ Skipping")
        return None
    
    X_with, y_with = data_with
    X_without, y_without = data_without
    
    print(f"\n  🔄 Running {N_SPLITS}-fold CV...")
    
    # Time series CV
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    
    results_with = {q: [] for q in QUANTILES}
    results_without = {q: [] for q in QUANTILES}
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X_with), 1):
        # WITH sentiment
        if len(X_with.columns) > 0:
            rf_with = RandomForestRegressor(
                n_estimators=N_ESTIMATORS,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
            rf_with.fit(X_with.iloc[train_idx], y_with.iloc[train_idx])
            
            # Get all tree predictions for quantiles
            all_preds = np.array([
                tree.predict(X_with.iloc[test_idx])
                for tree in rf_with.estimators_
            ])
            
            y_test = y_with.iloc[test_idx].values
            
            for q in QUANTILES:
                y_pred_q = np.percentile(all_preds, q * 100, axis=0)
                loss = quantile_loss(y_test, y_pred_q, q)
                results_with[q].append(loss)
        
        # WITHOUT sentiment (baseline = historical quantiles)
        y_train = y_with.iloc[train_idx].values
        y_test = y_with.iloc[test_idx].values
        
        for q in QUANTILES:
            y_pred_q = np.percentile(y_train, q * 100)
            y_pred_q = np.full(len(y_test), y_pred_q)
            loss = quantile_loss(y_test, y_pred_q, q)
            results_without[q].append(loss)
        
        print(f"    Fold {fold} complete")
    
    # Summarize
    summary = []
    print(f"\n  📊 Pinball Loss (lower = better):")
    print(f"  {'Quantile':<12} {'With Sent':<15} {'Baseline':<15} {'Improvement'}")
    print(f"  {'-'*60}")
    
    for q in QUANTILES:
        loss_with = np.mean(results_with[q]) if results_with[q] else np.nan
        loss_without = np.mean(results_without[q])
        improvement = ((loss_without - loss_with) / loss_without * 100) if not np.isnan(loss_with) else 0
        
        print(f"  {q:<12.2f} {loss_with:<15.4f} {loss_without:<15.4f} {improvement:>6.1f}%")
        
        summary.append({
            'Crypto': symbol,
            'Quantile': q,
            'Loss_With_Sentiment': loss_with,
            'Loss_Baseline': loss_without,
            'Improvement_%': improvement
        })
    
    return summary


def run_all_quantile_validations():
    """Run validation for all cryptos."""
    print(f"\n{'='*60}")
    print("QUANTILE REGRESSION ML VALIDATION")
    print(f"{'='*60}")
    print(f"Question: Does sentiment improve tail quantile prediction?")
    print(f"Method: Quantile Random Forest vs Historical Baseline")
    print(f"Quantiles: {QUANTILES}")
    print(f"Cryptos: {CRYPTO_SYMBOLS}")
    print(f"{'='*60}")
    
    all_results = []
    
    for symbol in CRYPTO_SYMBOLS:
        results = validate_quantile_prediction(symbol)
        if results:
            all_results.extend(results)
    
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(RESULTS_DIR / "quantile_ml_validation.csv", index=False)
        
        print(f"\n{'='*60}")
        print("SUMMARY: Sentiment Impact on Tail Prediction")
        print(f"{'='*60}")
        
        avg_by_quantile = results_df.groupby('Quantile')['Improvement_%'].mean()
        print("\nAverage improvement by quantile:")
        for q, imp in avg_by_quantile.items():
            marker = "🎯" if q in [0.05, 0.95] else "  "
            print(f"  {marker} {q:.2f}: {imp:>6.1f}%")
        
        tail_improvement = results_df[results_df['Quantile'].isin([0.05, 0.95])]['Improvement_%'].mean()
        center_improvement = results_df[~results_df['Quantile'].isin([0.05, 0.95])]['Improvement_%'].mean()
        
        print(f"\n  {'='*50}")
        print(f"  🎯 TAIL quantiles (5%, 95%):   {tail_improvement:>6.1f}% improvement")
        print(f"  📊 CENTER quantiles:            {center_improvement:>6.1f}% improvement")
        print(f"  {'='*50}")
        
        if tail_improvement > center_improvement:
            print(f"\n  ✅ CONCLUSION: Sentiment provides STRONGER improvement")
            print(f"     in tail prediction → validates quantile regression approach!")
        else:
            print(f"\n  ⚠️  Sentiment improvement similar across distribution")
        
    else:
        print("\n❌ No results generated")
    
    print(f"\n{'='*60}")
    print("✅ COMPLETE")
    print(f"Results: {RESULTS_DIR}/quantile_ml_validation.csv")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run_all_quantile_validations()
