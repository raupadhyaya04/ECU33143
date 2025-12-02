import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.api import VAR
from statsmodels.stats.outliers_influence import variance_inflation_factor
from pathlib import Path


# ---------------- CONFIG ----------------
CRYPTO_SYMBOLS = ['ADA', 'AVAX', 'BNB', 'BTC', 'DOGE', 'DOT', 'ETH', 'LINK', 'SOL', 'XRP']
CRYPTO_TICKERS = {
    'ADA': 'ADAUSDT', 'AVAX': 'AVAXUSDT', 'BNB': 'BNBUSDT', 'BTC': 'BTCUSDT',
    'DOGE': 'DOGEUSDT', 'DOT': 'DOTUSDT', 'ETH': 'ETHUSDT',
    'LINK': 'LINKUSDT', 'SOL': 'SOLUSDT', 'XRP': 'XRPUSDT'
}

MACRO_LOG_DIFF_COLS = [
    'Consumer Price Index', 'M2 Money Supply', 'GC=F', '^GSPC', 'DX-Y.NYB'
]
MACRO_FIRST_DIFF_COLS = [
    'Fed Funds Rate', '10-Year Treasury Yield', 'Crude Oil Price', '^VIX'
]
MACRO_CANDIDATES = [
    'Fed Funds Rate', '10-Year Treasury Yield',
    'Consumer Price Index', 'M2 Money Supply',
    'Crude Oil Price', 'GC=F', 'DX-Y.NYB',
    '^GSPC', '^VIX'
]

MACRO_VAR_SELECTION = ['^GSPC', '^VIX', 'M2 Money Supply']

DATA_DIR = Path("./Data")
RESULTS_DIR = Path("./Crypto-Macro/Results/Analysis")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OLS_LAGS = 3
GRANGER_MAX_LAG = 3
VAR_MAX_LAG = 3
IRF_HORIZON = 10
# ----------------------------------------


# ---------- DATA LOADING ----------

def load_macro_data() -> pd.DataFrame:
    """Load and combine FRED + Yahoo macro data."""
    df_fred = pd.read_csv(DATA_DIR / "Macro Data/macro_fred_data_filled.csv", 
                          index_col=0, parse_dates=True)
    df_yahoo = pd.read_csv(DATA_DIR / "Macro Data/macro_yahoo_data_filled.csv", 
                           index_col=0, parse_dates=True)
    df_macro = pd.concat([df_fred, df_yahoo], axis=1)
    df_macro = df_macro.ffill().bfill()
    return df_macro


def load_crypto_data() -> dict:
    """Load all cryptocurrency data. Returns dict of {symbol: dataframe}."""
    crypto_data = {}
    for sym, ticker in CRYPTO_TICKERS.items():
        df = pd.read_csv(DATA_DIR / f"Crypto Data/{ticker}.csv", 
                        index_col=0, parse_dates=True)
        crypto_data[sym] = df
    return crypto_data


# ---------- HELPER FUNCTIONS ----------

def adf_test(series: pd.Series, name: str) -> tuple:
    """Quick ADF test; returns (stat, p)."""
    s = series.dropna()
    stat, p, *_ = adfuller(s, autolag='AIC')
    status = 'STATIONARY ✅' if p <= 0.05 else 'NON-STATIONARY ❌'
    print(f"[ADF] {name:>20s}: stat={stat:8.3f}, p={p:.4f}  ->  {status}")
    return stat, p


def nw_lags(n: int) -> int:
    """Newey–West lag rule-of-thumb (Andrews 1991)."""
    return max(1, int(np.floor(4 * (n/100)**(2/9))))


def pct_log_diff(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Log-returns for positive-valued columns."""
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = np.log(out[c]).diff()
    return out


def first_diff(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """First difference for columns."""
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].diff()
    return out


def add_lags(df: pd.DataFrame, cols: list, lags: int) -> pd.DataFrame:
    """Add lagged versions of specified columns."""
    out = df.copy()
    for c in cols:
        for L in range(1, lags+1):
            out[f"{c}_lag{L}"] = out[c].shift(L)
    return out


def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    """Compute Variance Inflation Factors."""
    Xn = X.drop(columns=[c for c in X.columns if c.lower() in ('const','constant')], 
                errors='ignore').copy()
    Xn = Xn.replace([np.inf, -np.inf], np.nan).dropna()
    if Xn.shape[1] == 0:
        return pd.DataFrame(columns=['feature','VIF'])
    vifs = []
    for i, col in enumerate(Xn.columns):
        vifs.append((col, variance_inflation_factor(Xn.values, i)))
    return pd.DataFrame(vifs, columns=['feature','VIF']).sort_values('VIF', ascending=False)


# ---------- OLS REGRESSION ----------

def run_crypto_macro_regression(
    df_macro: pd.DataFrame,
    df_coin: pd.DataFrame,
    coin_name: str,
    price_col: str = 'close',
    macro_cols: list = None,
    log_return_targets: bool = True,
    macro_log_diff: list = None,
    macro_first_diff: list = None,
    lags: int = 3,
    dropna_how: str = 'any'
) -> tuple:
    """
    Run OLS regression with HAC standard errors.
    Returns: (model, X_used, y_used, vif_table, adf_results)
    """
    # 1) Merge on index
    df = pd.merge(
        df_macro.copy(),
        df_coin[[price_col]].rename(columns={price_col: f"{coin_name}_close"}),
        left_index=True, right_index=True, how='inner'
    ).sort_index()

    if macro_cols is None:
        macro_cols = [c for c in df.columns if c != f"{coin_name}_close"]

    # 2) Stationarity transforms
    if log_return_targets:
        df[f"{coin_name}_ret"] = np.log(df[f"{coin_name}_close"]).diff()
        y_col = f"{coin_name}_ret"
    else:
        y_col = f"{coin_name}_close"

    macro_log_diff = macro_log_diff or []
    macro_first_diff = macro_first_diff or []

    df_trans = df.copy()
    df_trans = pct_log_diff(df_trans, macro_log_diff)
    df_trans = first_diff(df_trans, macro_first_diff)

    # 3) Build design matrix
    X = df_trans[macro_cols].copy()

    # 4) Add lags
    X = add_lags(X, cols=macro_cols, lags=lags)

    # 5) Align y
    y = df_trans[y_col]

    # 6) Clean
    data = pd.concat([y, X], axis=1)
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(how=dropna_how)

    y = data[y_col]
    X = data.drop(columns=[y_col])

    # 7) ADF tests
    print(f"\n--- ADF tests for {coin_name} ---")
    adf_results = {}
    stat, p = adf_test(y, f"{coin_name} target ({'Δlog' if log_return_targets else 'level'})")
    adf_results['target'] = {'stat': stat, 'p_value': p}
    
    for c in macro_cols[:10]:  # Sample first 10
        if c in data.columns:
            stat, p = adf_test(data[c], f"{c} (t)")
            adf_results[c] = {'stat': stat, 'p_value': p}

    # 8) Fit OLS with HAC (Newey-West)
    X = sm.add_constant(X)
    T = len(y)
    model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': nw_lags(T)})

    # 9) VIF diagnostic
    vif_tbl = compute_vif(X)

    return model, X, y, vif_tbl, adf_results


# ---------- GRANGER CAUSALITY ----------

def run_granger_causality_tests(
    df_macro: pd.DataFrame,
    crypto_data: dict,
    macro_candidates: list,
    max_lag: int = 3
) -> pd.DataFrame:
    """
    Run Granger causality tests for all crypto-macro pairs.
    Returns: DataFrame with results
    """
    print("\n\n================== RUNNING GRANGER CAUSALITY TESTS ==================\n")
    
    granger_results = []
    
    for name, dfc in crypto_data.items():
        print(f"\nTesting Granger causality for {name}...")
        
        # Merge macro + crypto price
        df = pd.merge(
            df_macro[macro_candidates],
            dfc[['close']].rename(columns={'close': f'{name}_close'}),
            left_index=True, right_index=True, how='inner'
        ).sort_index()

        # Compute log returns
        df = df.apply(lambda x: np.log(x).diff())
        df = df.dropna()

        for macro_var in macro_candidates:
            try:
                test_res = grangercausalitytests(
                    df[[f'{name}_close', macro_var]], 
                    maxlag=max_lag, 
                    verbose=False
                )
                # Extract smallest p-value across lags
                p_vals = [test_res[i+1][0]['ssr_chi2test'][1] for i in range(max_lag)]
                min_p = min(p_vals)
                granger_results.append({
                    'Crypto': name,
                    'Macro Variable': macro_var,
                    'Min p-Value': round(min_p, 4),
                    'Significant': '✅' if min_p < 0.05 else '❌'
                })
            except Exception as e:
                print(f"Skipped {name}-{macro_var} due to error: {e}")
                continue

    df_granger = pd.DataFrame(granger_results)
    df_granger.sort_values(['Macro Variable','Crypto'], inplace=True)
    
    return df_granger


# ---------- VAR AND IRF ----------

def compute_var_irf(
    df_macro: pd.DataFrame,
    df_coin: pd.DataFrame,
    coin_name: str,
    macro_vars: list,
    macro_log_diff: list,
    macro_first_diff: list,
    max_lag: int = 3,
    horizon: int = 10
) -> dict:
    """
    Fit VAR model and compute IRFs.
    Returns: dict with VAR results and IRF data
    """
    print(f"\nProcessing VAR/IRF for {coin_name}...")
    
    # Merge crypto return + selected macro factors
    df_merge = pd.merge(
        df_macro.copy(),
        df_coin[['close']].rename(columns={'close': f'{coin_name}_close'}),
        left_index=True, right_index=True, how='inner'
    ).sort_index()

    # Compute Δlog returns for crypto
    df_merge[f'{coin_name}_ret'] = np.log(df_merge[f'{coin_name}_close']).diff()

    # Transform macros for stationarity
    df_merge = pct_log_diff(df_merge, macro_log_diff)
    df_merge = first_diff(df_merge, macro_first_diff)

    # Keep only relevant vars
    keep_cols = [f'{coin_name}_ret'] + macro_vars
    df_var = df_merge[keep_cols].dropna()

    if len(df_var) < 50:
        print(f"Skipping {coin_name} — insufficient data ({len(df_var)} obs).")
        return None

    try:
        var_model = VAR(df_var)
        results = var_model.fit(maxlags=max_lag, ic='aic')
        
        # Generate IRFs
        irf = results.irf(horizon)
        
        # Extract IRF data for each macro variable
        irf_data = {}
        idx_resp = list(irf.model.endog_names).index(f'{coin_name}_ret')
        
        for macro in macro_vars:
            idx_imp = list(irf.model.endog_names).index(macro)
            irf_vals = irf.irfs[:, idx_resp, idx_imp]
            irf_data[macro] = irf_vals
        
        return {
            'model': results,
            'irf_object': irf,
            'irf_data': irf_data,
            'response_var': f'{coin_name}_ret',
            'impulse_vars': macro_vars,
            'horizon': horizon
        }
        
    except Exception as e:
        print(f"Error fitting VAR for {coin_name}: {e}")
        return None


# ---------- SUMMARY TABLES ----------

def summarize_ols_models(results_dict: dict, sig_threshold: float = 0.05) -> pd.DataFrame:
    """Create summary table of OLS regression results."""
    rows = []
    for coin, content in results_dict.items():
        model = content['model']
        params = model.params
        tvals = model.tvalues
        pvals = model.pvalues
        for var in params.index:
            if var.lower() == 'const':
                continue
            rows.append({
                'Crypto': coin,
                'Variable': var,
                'Coef': params[var],
                't-Stat': tvals[var],
                'p-Value': pvals[var],
                'Significant': '✅' if pvals[var] < sig_threshold else '❌'
            })
    df_summary = pd.DataFrame(rows)
    df_summary = df_summary.sort_values(['Variable','Crypto']).reset_index(drop=True)
    return df_summary


# ---------- MAIN ANALYSIS RUNNER ----------

def run_all_analyses():
    """
    Run all analyses and save results.
    Returns: (ols_results, granger_df, var_irf_results, macro_data, crypto_data)
    """
    print("="*90)
    print("LOADING DATA")
    print("="*90)
    
    # Load data
    df_macro = load_macro_data()
    crypto_data = load_crypto_data()
    
    print(f"\nLoaded macro data: {df_macro.shape}")
    print(f"Loaded {len(crypto_data)} cryptocurrencies")
    
    # ---------- OLS REGRESSION ----------
    print("\n" + "="*90)
    print("RUNNING OLS REGRESSIONS")
    print("="*90)
    
    ols_results = {}
    
    for name, dfc in crypto_data.items():
        model, X_used, y_used, vif_tbl, adf_results = run_crypto_macro_regression(
            df_macro=df_macro,
            df_coin=dfc,
            coin_name=name,
            price_col='close',
            macro_cols=[c for c in df_macro.columns],
            log_return_targets=True,
            macro_log_diff=MACRO_LOG_DIFF_COLS,
            macro_first_diff=MACRO_FIRST_DIFF_COLS,
            lags=OLS_LAGS
        )
        
        print("\n", "="*90)
        print(f"{name} — OLS with HAC (NW) — Δlog target, {OLS_LAGS} lags")
        print("="*90)
        print(model.summary())
        print("\n[VIF]")
        print(vif_tbl.to_string(index=False))
        
        ols_results[name] = {
            'model': model,
            'vif': vif_tbl,
            'X': X_used,
            'y': y_used,
            'adf': adf_results
        }
    
    # Create OLS summary table
    ols_summary = summarize_ols_models(ols_results)
    pd.set_option('display.max_rows', None)
    print("\n" + "="*90)
    print("OLS SUMMARY TABLE")
    print("="*90)
    print(ols_summary)
    
    # Save OLS results
    ols_summary.to_csv(RESULTS_DIR / "macro_crypto_regression_summary.csv", index=False)
    print(f"\n✅ OLS summary saved to {RESULTS_DIR / 'macro_crypto_regression_summary.csv'}")
    
    # ---------- GRANGER CAUSALITY ----------
    granger_df = run_granger_causality_tests(
        df_macro=df_macro,
        crypto_data=crypto_data,
        macro_candidates=MACRO_CANDIDATES,
        max_lag=GRANGER_MAX_LAG
    )
    
    print("\n" + "="*90)
    print("GRANGER CAUSALITY SUMMARY")
    print("="*90)
    print(granger_df)
    
    # Save Granger results
    granger_df.to_csv(RESULTS_DIR / "granger_causality_results.csv", index=False)
    print(f"\n✅ Granger results saved to {RESULTS_DIR / 'granger_causality_results.csv'}")
    
    # ---------- VAR/IRF ----------
    print("\n" + "="*90)
    print("COMPUTING VAR MODELS AND IMPULSE RESPONSE FUNCTIONS")
    print("="*90)
    
    var_irf_results = {}
    
    for name, dfc in crypto_data.items():
        result = compute_var_irf(
            df_macro=df_macro,
            df_coin=dfc,
            coin_name=name,
            macro_vars=MACRO_VAR_SELECTION,
            macro_log_diff=MACRO_LOG_DIFF_COLS,
            macro_first_diff=MACRO_FIRST_DIFF_COLS,
            max_lag=VAR_MAX_LAG,
            horizon=IRF_HORIZON
        )
        if result:
            var_irf_results[name] = result
            
            # Save IRF data
            irf_df = pd.DataFrame(result['irf_data'])
            irf_df.to_csv(RESULTS_DIR / f"{name}_irf_data.csv", index=False)
    
    print(f"\n✅ VAR/IRF analysis complete. Results saved to {RESULTS_DIR}/")
    
    print("\n" + "="*90)
    print("ALL ANALYSES COMPLETE")
    print("="*90)
    print(f"\nSaved files:")
    print(f"  - OLS summary: {RESULTS_DIR / 'macro_crypto_regression_summary.csv'}")
    print(f"  - Granger results: {RESULTS_DIR / 'granger_causality_results.csv'}")
    print(f"  - IRF data: {RESULTS_DIR}/*_irf_data.csv")
    
    return ols_results, granger_df, var_irf_results, df_macro, crypto_data


if __name__ == "__main__":
    ols_results, granger_df, var_irf_results, df_macro, crypto_data = run_all_analyses()