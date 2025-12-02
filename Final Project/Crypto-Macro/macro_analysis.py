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

# Macro variables used inside VAR / IRF
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
    df_fred = pd.read_csv(
        DATA_DIR / "Macro Data/macro_fred_data_filled.csv",
        index_col=0, parse_dates=True
    )
    df_yahoo = pd.read_csv(
        DATA_DIR / "Macro Data/macro_yahoo_data_filled.csv",
        index_col=0, parse_dates=True
    )
    df_macro = pd.concat([df_fred, df_yahoo], axis=1)
    df_macro = df_macro.ffill().bfill()
    return df_macro


def load_crypto_data() -> dict:
    """Load all cryptocurrency data. Returns dict of {symbol: dataframe}."""
    crypto_data = {}
    for sym, ticker in CRYPTO_TICKERS.items():
        df = pd.read_csv(
            DATA_DIR / f"Crypto Data/{ticker}.csv",
            index_col=0, parse_dates=True
        )
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
    """
    Newey–West lag rule-of-thumb.
    Using S = 0.75 * T^(1/3) as in common practice (Stock & Watson style).
    """
    return max(1, int(np.floor(0.75 * n ** (1/3))))


def pct_log_diff(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Log-returns for positive-valued columns (Δlog)."""
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
        for L in range(1, lags + 1):
            out[f"{c}_lag{L}"] = out[c].shift(L)
    return out


def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    """Compute Variance Inflation Factors."""
    Xn = X.drop(
        columns=[c for c in X.columns if c.lower() in ('const', 'constant')],
        errors='ignore'
    ).copy()
    Xn = Xn.replace([np.inf, -np.inf], np.nan).dropna()
    if Xn.shape[1] == 0:
        return pd.DataFrame(columns=['feature', 'VIF'])
    vifs = []
    for i, col in enumerate(Xn.columns):
        vifs.append((col, variance_inflation_factor(Xn.values, i)))
    return pd.DataFrame(vifs, columns=['feature', 'VIF']).sort_values('VIF', ascending=False)


def build_crypto_macro_panel(
    df_macro: pd.DataFrame,
    df_coin: pd.DataFrame,
    coin_name: str,
    macro_cols: list
) -> pd.DataFrame:
    """
    Build joint macro-crypto panel with consistent transformations:
      - Crypto: Δlog price
      - Macros: Δlog for MACRO_LOG_DIFF_COLS, first diff for MACRO_FIRST_DIFF_COLS.
    """
    df_merge = pd.merge(
        df_macro.copy(),
        df_coin[['close']].rename(columns={'close': f'{coin_name}_close'}),
        left_index=True, right_index=True, how='inner'
    ).sort_index()

    # Crypto return
    df_merge[f'{coin_name}_ret'] = np.log(df_merge[f'{coin_name}_close']).diff()

    # Transform macros
    df_merge = pct_log_diff(df_merge, MACRO_LOG_DIFF_COLS)
    df_merge = first_diff(df_merge, MACRO_FIRST_DIFF_COLS)

    keep_cols = [f'{coin_name}_ret'] + [c for c in macro_cols if c in df_merge.columns]
    df_panel = df_merge[keep_cols].replace([np.inf, -np.inf], np.nan).dropna()
    return df_panel


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

    for c in macro_cols[:10]:  # Sample first 10 for brevity
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
    Returns: DataFrame with raw and Bonferroni-adjusted results.
    """
    print("\n\n================== RUNNING GRANGER CAUSALITY TESTS ==================\n")

    granger_results = []

    for name, dfc in crypto_data.items():
        print(f"\nTesting Granger causality for {name}...")

        # Build panel with consistent transforms
        df_panel = build_crypto_macro_panel(
            df_macro=df_macro,
            df_coin=dfc,
            coin_name=name,
            macro_cols=macro_candidates
        )

        if df_panel.shape[0] < max_lag + 5:
            print(f"  Skipping {name} — not enough data after transforms ({df_panel.shape[0]} obs).")
            continue

        for macro_var in macro_candidates:
            if macro_var not in df_panel.columns:
                continue
            try:
                # Granger: does macro_var help predict crypto returns?
                test_res = grangercausalitytests(
                    df_panel[[f'{name}_ret', macro_var]],
                    maxlag=max_lag,
                    verbose=False
                )
                p_vals = [test_res[i + 1][0]['ssr_chi2test'][1] for i in range(max_lag)]
                min_p = min(p_vals)
                granger_results.append({
                    'Crypto': name,
                    'Macro Variable': macro_var,
                    'Min p-Value': round(min_p, 4),
                    'Significant': '✅' if min_p < 0.05 else '❌'
                })
            except Exception as e:
                print(f"  Skipped {name}-{macro_var} due to error: {e}")
                continue

    df_granger = pd.DataFrame(granger_results)
    if df_granger.empty:
        return df_granger

    df_granger.sort_values(['Macro Variable', 'Crypto'], inplace=True)

    # Multiple-testing adjustment (Bonferroni over all pairs)
    m = len(df_granger)
    df_granger['p_adj'] = (df_granger['Min p-Value'] * m).clip(upper=1.0)
    df_granger['Significant_adj'] = df_granger['p_adj'].apply(lambda p: '✅' if p < 0.05 else '❌')

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
    Returns: dict with VAR results, IRF data, and meta information.
    """
    print(f"\nProcessing VAR/IRF for {coin_name}...")

    # Joint panel (re-use transform logic)
    df_panel = build_crypto_macro_panel(
        df_macro=df_macro,
        df_coin=df_coin,
        coin_name=coin_name,
        macro_cols=macro_vars
    )

    # For VAR we use the same transformed panel
    df_var = df_panel[[f'{coin_name}_ret'] + macro_vars].dropna()

    if len(df_var) < 50:
        print(f"  Skipping {coin_name} — insufficient data for VAR ({len(df_var)} obs).")
        return None

    try:
        var_model = VAR(df_var)
        results = var_model.fit(maxlags=max_lag, ic='aic')
        selected_lag = results.k_ar
        print(f"  Selected VAR lag for {coin_name}: {selected_lag}")

        # Stability check
        stable = results.is_stable()
        print(f"  VAR stability for {coin_name}: {stable}")
        if not stable:
            print(f"  Skipping IRF for {coin_name} (unstable VAR).")
            return None

        # Generate IRFs
        irf = results.irf(horizon)

        # Extract IRF data for each macro variable: response = crypto_ret, impulse = macro
        irf_data = {}
        idx_resp = list(irf.model.endog_names).index(f'{coin_name}_ret')

        for macro in macro_vars:
            if macro not in irf.model.endog_names:
                continue
            idx_imp = list(irf.model.endog_names).index(macro)
            irf_vals = irf.irfs[:, idx_resp, idx_imp]
            irf_data[macro] = irf_vals

        return {
            'model': results,
            'irf_object': irf,
            'irf_data': irf_data,
            'response_var': f'{coin_name}_ret',
            'impulse_vars': macro_vars,
            'horizon': horizon,
            'selected_lag': selected_lag,
            'stable': stable
        }

    except Exception as e:
        print(f"  Error fitting VAR for {coin_name}: {e}")
        return None


# ---------- SUMMARY TABLES ----------

def summarize_ols_models(results_dict: dict, sig_threshold: float = 0.05) -> pd.DataFrame:
    """Create summary table of OLS regression results with R2 and direction."""
    rows = []
    for coin, content in results_dict.items():
        model = content['model']
        params = model.params
        tvals = model.tvalues
        pvals = model.pvalues
        r2 = float(model.rsquared)
        r2_adj = float(model.rsquared_adj)
        for var in params.index:
            if var.lower() == 'const':
                continue
            sig = pvals[var] < sig_threshold
            direction = 'pos' if (sig and params[var] > 0) else ('neg' if (sig and params[var] < 0) else 'ns')
            rows.append({
                'Crypto': coin,
                'Variable': var,
                'Coef': params[var],
                't-Stat': tvals[var],
                'p-Value': pvals[var],
                'Significant': '✅' if sig else '❌',
                'Direction': direction,
                'R2': r2,
                'Adj_R2': r2_adj
            })
    df_summary = pd.DataFrame(rows)
    df_summary = df_summary.sort_values(['Variable', 'Crypto']).reset_index(drop=True)
    return df_summary


# ---------- MAIN ANALYSIS RUNNER ----------

def run_all_analyses():
    """
    Run all analyses and save results.
    Returns: (ols_results, granger_df, var_irf_results, macro_data, crypto_data)
    """
    print("=" * 90)
    print("LOADING DATA")
    print("=" * 90)

    # Load data
    df_macro = load_macro_data()
    crypto_data = load_crypto_data()

    print(f"\nLoaded macro data: {df_macro.shape}")
    print(f"Loaded {len(crypto_data)} cryptocurrencies")

    # ---------- OLS REGRESSION ----------
    print("\n" + "=" * 90)
    print("RUNNING OLS REGRESSIONS")
    print("=" * 90)

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

        print("\n", "=" * 90)
        print(f"{name} — OLS with HAC (NW) — Δlog target, {OLS_LAGS} lags")
        print("=" * 90)
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
    print("\n" + "=" * 90)
    print("OLS SUMMARY TABLE")
    print("=" * 90)
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

    print("\n" + "=" * 90)
    print("GRANGER CAUSALITY SUMMARY")
    print("=" * 90)
    print(granger_df)

    # Save Granger results
    granger_df.to_csv(RESULTS_DIR / "granger_causality_results.csv", index=False)
    print(f"\n✅ Granger results saved to {RESULTS_DIR / 'granger_causality_results.csv'}")

    # ---------- VAR/IRF ----------
    print("\n" + "=" * 90)
    print("COMPUTING VAR MODELS AND IMPULSE RESPONSE FUNCTIONS")
    print("=" * 90)

    var_irf_results = {}
    var_meta_rows = []

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

            # Save IRF data (central IRFs only)
            irf_df = pd.DataFrame(result['irf_data'])
            irf_df.to_csv(RESULTS_DIR / f"{name}_irf_data.csv", index=False)

            # Meta for this VAR
            model = result['model']
            var_meta_rows.append({
                'Crypto': name,
                'n_obs': int(model.nobs),
                'selected_lag': result.get('selected_lag', model.k_ar),
                'stable': bool(result.get('stable', model.is_stable()))
            })

    # Save VAR meta
    if var_meta_rows:
        var_meta_df = pd.DataFrame(var_meta_rows)
        var_meta_df.to_csv(RESULTS_DIR / "var_model_meta.csv", index=False)
        print(f"\n✅ VAR meta saved to {RESULTS_DIR / 'var_model_meta.csv'}")

    print(f"\n✅ VAR/IRF analysis complete. Results saved to {RESULTS_DIR}/")

    print("\n" + "=" * 90)
    print("ALL ANALYSES COMPLETE")
    print("=" * 90)
    print(f"\nSaved files:")
    print(f"  - OLS summary: {RESULTS_DIR / 'macro_crypto_regression_summary.csv'}")
    print(f"  - Granger results: {RESULTS_DIR / 'granger_causality_results.csv'}")
    print(f"  - VAR meta: {RESULTS_DIR / 'var_model_meta.csv'}")
    print(f"  - IRF data: {RESULTS_DIR}/*_irf_data.csv")

    return ols_results, granger_df, var_irf_results, df_macro, crypto_data


if __name__ == "__main__":
    ols_results, granger_df, var_irf_results, df_macro, crypto_data = run_all_analyses()