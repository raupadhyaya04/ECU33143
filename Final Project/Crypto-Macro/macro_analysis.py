import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Data Loading and combining (for analysis)
df_fred = pd.read_csv("../Data/Macro Data/macro_fred_data_filled.csv", index_col=0, parse_dates=True)
df_yahoo = pd.read_csv("../Data/Macro Data/macro_yahoo_data_filled.csv", index_col=0, parse_dates=True)
df_macro = pd.concat([df_fred, df_yahoo], axis=1)
df_macro = df_macro.ffill().bfill()

df_ADA = pd.read_csv("../Data/Crypto Data/ADAUSDT.csv", index_col=0, parse_dates=True)
df_AVA = pd.read_csv("../Data/Crypto Data/AVAXUSDT.csv", index_col=0, parse_dates=True)
df_BNB = pd.read_csv("../Data/Crypto Data/BNBUSDT.csv", index_col=0, parse_dates=True)
df_BTC = pd.read_csv("../Data/Crypto Data/BTCUSDT.csv", index_col=0, parse_dates=True)
df_DOGE = pd.read_csv("../Data/Crypto Data/DOGEUSDT.csv", index_col=0, parse_dates=True)
df_DOT = pd.read_csv("../Data/Crypto Data/DOTUSDT.csv", index_col=0, parse_dates=True)
df_ETH = pd.read_csv("../Data/Crypto Data/ETHUSDT.csv", index_col=0, parse_dates=True)
df_LINK = pd.read_csv("../Data/Crypto Data/LINKUSDT.csv", index_col=0, parse_dates=True)
df_SOL = pd.read_csv("../Data/Crypto Data/SOLUSDT.csv", index_col=0, parse_dates=True)
df_XRP = pd.read_csv("../Data/Crypto Data/XRPUSDT.csv", index_col=0, parse_dates=True)

# ---------- Helpers ----------

def adf_test(series: pd.Series, name: str):
    """Quick ADF printout; returns (stat, p)."""
    s = series.dropna()
    stat, p, *_ = adfuller(s, autolag='AIC')
    print(f"[ADF] {name:>20s}: stat={stat:8.3f}, p={p:.4f}  ->  {'STATIONARY ✅' if p<=0.05 else 'NON-STATIONARY ❌'}")
    return stat, p

def nw_lags(n: int) -> int:
    """
    Newey–West lag rule-of-thumb (Andrews 1991 variant).
    """
    return max(1, int(np.floor(4 * (n/100)**(2/9))))

def pct_log_diff(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """log-returns for positive-valued columns; safe for macro like CPI/M2/Gold/S&P, etc."""
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = np.log(out[c]).diff()
    return out

def first_diff(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].diff()
    return out

def add_lags(df: pd.DataFrame, cols: list, lags: int) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        for L in range(1, lags+1):
            out[f"{c}_lag{L}"] = out[c].shift(L)
    return out

def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    # drop constant for VIF; VIF blows up if any col is constant or collinear
    Xn = X.drop(columns=[c for c in X.columns if c.lower() in ('const','constant')], errors='ignore').copy()
    Xn = Xn.replace([np.inf, -np.inf], np.nan).dropna()
    if Xn.shape[1] == 0:
        return pd.DataFrame(columns=['feature','VIF'])
    vifs = []
    for i, col in enumerate(Xn.columns):
        vifs.append((col, variance_inflation_factor(Xn.values, i)))
    return pd.DataFrame(vifs, columns=['feature','VIF']).sort_values('VIF', ascending=False)

# ---------- Core pipeline ----------

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
):
    """
    Returns: model, summary, VIF table, (X, y) aligned used in fit
    """
    # 1) Merge on index
    df = pd.merge(
        df_macro.copy(),
        df_coin[[price_col]].rename(columns={price_col: f"{coin_name}_close"}),
        left_index=True, right_index=True, how='inner'
    ).sort_index()

    # Make sure macro_cols defaults to all macro features
    if macro_cols is None:
        macro_cols = [c for c in df.columns if c != f"{coin_name}_close"]

    # 2) Stationarity transforms
    # Target → log-returns (Δlog price)
    if log_return_targets:
        df[f"{coin_name}_ret"] = np.log(df[f"{coin_name}_close"]).diff()
        y_col = f"{coin_name}_ret"
    else:
        # not recommended, but available
        y_col = f"{coin_name}_close"

    # Macros → specify which to log-diff vs first-diff
    macro_log_diff = macro_log_diff or []      # e.g., ['Consumer Price Index','M2 Money Supply','GC=F','^GSPC','DX-Y.NYB']
    macro_first_diff = macro_first_diff or []  # e.g., ['Fed Funds Rate','10-Year Treasury Yield','^VIX','Crude Oil Price']

    df_trans = df.copy()
    df_trans = pct_log_diff(df_trans, macro_log_diff)
    df_trans = first_diff(df_trans, macro_first_diff)

    # 3) Build design matrix
    X = df_trans[macro_cols].copy()

    # 4) Add contemporaneous + lagged predictors
    X = add_lags(X, cols=macro_cols, lags=lags)

    # 5) Align y
    y = df_trans[y_col]

    # 6) Clean
    data = pd.concat([y, X], axis=1)
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(how=dropna_how)

    y = data[y_col]
    X = data.drop(columns=[y_col])

    # 7) ADF sanity prints (optional; comment out if noisy)
    print("\n--- ADF tests (post-transform) ---")
    adf_test(y, f"{coin_name} target ({'Δlog' if log_return_targets else 'level'})")
    for c in macro_cols[:10]:  # sample first 10 to reduce spam
        if c in data.columns:
            adf_test(data[c], f"{c} (t)")

    # 8) Fit OLS with HAC (Newey-West) robust SEs
    X = sm.add_constant(X)
    T = len(y)
    model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': nw_lags(T)})

    # 9) VIF diagnostic
    vif_tbl = compute_vif(X)

    return model, model.summary(), vif_tbl, (X, y)

# ---------- Choose macro transform sets once ----------

# Example split (tweak to your column names):
macro_log_diff_cols = [
    'Consumer Price Index', 'M2 Money Supply', 'GC=F', '^GSPC', 'DX-Y.NYB'
]
macro_first_diff_cols = [
    'Fed Funds Rate', '10-Year Treasury Yield', 'Crude Oil Price', '^VIX'
]

# Macro prep: forward/backward fill and (OPTIONAL) align to daily close
df_macro = df_macro.ffill().bfill()

# ---------- Run across all coins in a loop ----------

coins = {
    'ADA': df_ADA, 'AVAX': df_AVA, 'BNB': df_BNB, 'BTC': df_BTC,
    'DOGE': df_DOGE, 'DOT': df_DOT, 'ETH': df_ETH,
    'LINK': df_LINK, 'SOL': df_SOL, 'XRP': df_XRP
}

all_results = {}

for name, dfc in coins.items():
    model, summary, vif_tbl, (X_used, y_used) = run_crypto_macro_regression(
        df_macro=df_macro,
        df_coin=dfc,
        coin_name=name,
        price_col='close',
        macro_cols=[c for c in df_macro.columns],   # use all macro columns you loaded
        log_return_targets=True,                    # use Δlog price as target
        macro_log_diff=macro_log_diff_cols,
        macro_first_diff=macro_first_diff_cols,
        lags=3
    )
    print("\n", "="*90, f"\n{name} — OLS with HAC (NW) — Δlog target, 3 lags\n", "="*90)
    print(summary)
    print("\n[VIF]")
    print(vif_tbl.to_string(index=False))
    all_results[name] = {'model': model, 'vif': vif_tbl, 'X': X_used, 'y': y_used}


# ---------- Automated summary table ----------

def summarize_models(results_dict, sig_threshold=0.05):
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

summary_table = summarize_models(all_results)
pd.set_option('display.max_rows', None)
print(summary_table)

# Optional: save to CSV
summary_table.to_csv("./Results/macro_crypto_regression_summary.csv", index=False)
print("\n✅ Summary table saved to ./Results/macro_crypto_regression_summary.csv")


# ---------- Granger Causality Tests ----------

from statsmodels.tsa.stattools import grangercausalitytests

print("\n\n================== RUNNING GRANGER CAUSALITY TESTS ==================\n")

macro_candidates = [
    'Fed Funds Rate', '10-Year Treasury Yield',
    'Consumer Price Index', 'M2 Money Supply',
    'Crude Oil Price', 'GC=F', 'DX-Y.NYB',
    '^GSPC', '^VIX'
]
max_lag = 3
granger_results = []

for name, dfc in coins.items():
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
            test_res = grangercausalitytests(df[[f'{name}_close', macro_var]], maxlag=max_lag, verbose=False)
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

print("\n================== GRANGER SUMMARY TABLE ==================\n")
print(df_granger)

# Save results
os.makedirs("./Results", exist_ok=True)
df_granger.to_csv("./Results/granger_causality_results.csv", index=False)
print("\n✅ Granger causality results saved to ./Results/granger_causality_results.csv")


# ---------- Granger Causality Visualization ----------

# Load Granger results
df_granger = pd.read_csv("./Results/granger_causality_results.csv")

# Pivot for heatmap
heatmap_data = df_granger.pivot(index="Crypto", columns="Macro Variable", values="Min p-Value")

# Optional: Rename macro variables for cleaner display
label_map = {
    "10-Year Treasury Yield": "10Y Yield",
    "Consumer Price Index": "CPI",
    "Crude Oil Price": "Oil",
    "DX-Y.NYB": "USD Index",
    "Fed Funds Rate": "Fed Rate",
    "GC=F": "Gold",
    "M2 Money Supply": "M2",
    "^GSPC": "S&P 500",
    "^VIX": "VIX"
}
heatmap_data = heatmap_data.rename(columns=label_map)

# Sort cryptos alphabetically or by sensitivity (optional)
# heatmap_data = heatmap_data.loc[(heatmap_data < 0.05).sum(axis=1).sort_values(ascending=False).index]

# Create heatmap
plt.figure(figsize=(11, 7))
sns.heatmap(
    heatmap_data,
    annot=True, fmt=".3f",
    cmap="RdYlGn_r",
    cbar_kws={'label': 'p-value'},
    linewidths=0.5, linecolor='gray',
    annot_kws={"size": 8}
)

# Titles and labels
plt.title("Granger Causality: Macro Variables → Crypto Returns", fontsize=14, weight='bold', pad=12)
plt.xlabel("Macro Variable", fontsize=12, labelpad=10)
plt.ylabel("Cryptocurrency", fontsize=12, labelpad=10)

# Rotate tick labels for clarity
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=11)

# Improve layout
plt.tight_layout()

# Save output
os.makedirs("./Results/Heatmaps", exist_ok=True)
plt.savefig("./Results/Heatmaps/granger_heatmap.png", dpi=300, bbox_inches='tight')
plt.close()

print("\n✅ Granger causality heatmap saved to ./Results/Heatmaps/granger_heatmap.png")


# ---------- Impulse Response Visualization (VAR-based) ----------

from statsmodels.tsa.api import VAR
import matplotlib.pyplot as plt
import os

# Make output folder for charts
os.makedirs("./Results/IRF", exist_ok=True)

macro_vars = ['^GSPC', '^VIX', 'M2 Money Supply']
horizon = 10  # days ahead for IRF horizon

print("\n\n================== GENERATING VAR IRF VISUALS ==================\n")

for name, dfc in coins.items():
    print(f"\nProcessing IRFs for {name}...")
    # Merge crypto return + selected macro factors
    df_merge = pd.merge(
        df_macro.copy(),
        dfc[['close']].rename(columns={'close': f'{name}_close'}),
        left_index=True, right_index=True, how='inner'
    ).sort_index()

    # Compute Δlog returns for crypto
    df_merge[f'{name}_ret'] = np.log(df_merge[f'{name}_close']).diff()

    # Transform macros for stationarity
    df_merge = pct_log_diff(df_merge, macro_log_diff_cols)
    df_merge = first_diff(df_merge, macro_first_diff_cols)

    # Keep only relevant vars
    keep_cols = [f'{name}_ret'] + macro_vars
    df_var = df_merge[keep_cols].dropna()

    if len(df_var) < 50:
        print(f"Skipping {name} — insufficient data ({len(df_var)} obs).")
        continue

    try:
        var_model = VAR(df_var)
        results = var_model.fit(maxlags=3, ic='aic')
    except Exception as e:
        print(f"Error fitting VAR for {name}: {e}")
        continue

    # Generate IRFs (10-period horizon)
    irf = results.irf(horizon)

    # Create figure with one subplot per macro
    fig, axes = plt.subplots(1, len(macro_vars), figsize=(15, 4))
    fig.suptitle(f"Impulse Response of {name} Returns to Macro Shocks", fontsize=14)

    for i, macro in enumerate(macro_vars):
        ax = axes[i]
        try:
            # statsmodels >=0.14: no 'ax' arg, returns a figure
            tmp_fig = irf.plot(impulse=macro, response=f'{name}_ret', orth=True)
            # grab the axes from the returned figure and copy to our grid
            irf_ax = tmp_fig.axes[0]
            for line in irf_ax.lines:
                ax.plot(line.get_xdata(), line.get_ydata(), color=line.get_color())
            ax.set_title(f"{macro} Shock → {name}_Return")
            ax.axhline(0, color='black', lw=0.8)
            ax.grid(True, alpha=0.3)
            plt.close(tmp_fig)
        except Exception as e:
            # fallback manual extraction
            idx_resp = list(irf.model.endog_names).index(f'{name}_ret')
            idx_imp = list(irf.model.endog_names).index(macro)
            irf_vals = irf.irfs[:, idx_resp, idx_imp]
            ax.plot(irf_vals, color='tab:blue')
            ax.axhline(0, color='black', lw=0.8)
            ax.set_title(f"{macro} Shock → {name}_Return")
            ax.grid(True, alpha=0.3)

        ax.set_xlabel("Days after Shock")
        ax.set_ylabel("Response")

    plt.tight_layout()
    plt.savefig(f"./Results/IRF/{name}_IRF.png", dpi=300)
    plt.close(fig)

print("\nAll IRF plots saved under ./Results/IRF/")
