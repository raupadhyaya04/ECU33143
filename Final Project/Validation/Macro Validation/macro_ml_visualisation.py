import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).parent.parent.parent
VALIDATION_DIR = BASE_DIR / "Validation" / "Macro Validation"
RESULTS_DIR = VALIDATION_DIR / "Results"
PLOTS_DIR = VALIDATION_DIR / "Plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Consistent crypto order
CRYPTO_SYMBOLS = ['ADA', 'AVAX', 'BNB', 'BTC', 'DOGE', 'DOT',
                  'ETH', 'LINK', 'SOL', 'XRP']

# -----------------------------
# 1. Load summary
# -----------------------------
summary_file = RESULTS_DIR / "ml_validation_summary.csv"
if not summary_file.exists():
    raise FileNotFoundError(f"Summary file not found: {summary_file}")

summary = pd.read_csv(summary_file)

# Determine which column holds the crypto symbol
if 'Crypto' in summary.columns:
    crypto_col = 'Crypto'
elif 'symbol' in summary.columns:
    crypto_col = 'symbol'
else:
    raise KeyError(f"No crypto column found in summary. Columns: {list(summary.columns)}")

# Normalise to a 'Crypto' column with the desired order
summary['Crypto'] = summary[crypto_col].astype(str)
summary['Crypto'] = pd.Categorical(
    summary['Crypto'],
    categories=CRYPTO_SYMBOLS,
    ordered=True
)
summary = summary.sort_values('Crypto')

# Determine R2 column names
if 'R² Mean' in summary.columns:
    r2_col = 'R² Mean'
    r2_std_col = 'R² Std' if 'R² Std' in summary.columns else None
elif 'r2_mean' in summary.columns:
    r2_col = 'r2_mean'
    r2_std_col = 'r2_std' if 'r2_std' in summary.columns else None
else:
    raise KeyError(f"No R² column found in summary. Columns: {list(summary.columns)}")

# Top feature columns
if 'Top Feature' in summary.columns:
    top_feat_col = 'Top Feature'
elif 'top_feature' in summary.columns:
    top_feat_col = 'top_feature'
else:
    raise KeyError(f"No top feature column found. Columns: {list(summary.columns)}")

if 'Top Importance' in summary.columns:
    top_imp_col = 'Top Importance'
elif 'top_importance' in summary.columns:
    top_imp_col = 'top_importance'
else:
    raise KeyError(f"No top importance column found. Columns: {list(summary.columns)}")

# -----------------------------
# 2. Bar chart: R² per crypto
# -----------------------------
plt.figure(figsize=(10, 5))

yerr = summary[r2_std_col] if r2_std_col and r2_std_col in summary.columns else None

bars = plt.bar(
    summary['Crypto'],
    summary[r2_col],
    yerr=yerr,
    capsize=4,
    color='steelblue',
    alpha=0.8
)
plt.axhline(0, color='black', linewidth=1)
plt.ylabel("R² (mean across CV folds)")
plt.title("Random Forest R² by Cryptocurrency")

for bar, r2 in zip(bars, summary[r2_col]):
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f"{r2:.2f}",
        ha='center',
        va='bottom',
        fontsize=8
    )

plt.tight_layout()
plt.savefig(PLOTS_DIR / "R2_per_crypto.png", dpi=300)
plt.close()

# -----------------------------
# 3. Bar chart: top feature per crypto
# -----------------------------
plt.figure(figsize=(10, 5))
bars = plt.bar(
    summary['Crypto'],
    summary[top_imp_col],
    color='darkorange',
    alpha=0.8
)
plt.ylabel("Importance of top feature")
plt.title("Importance of Most Influential Macro Variable (per Crypto)")

for bar, feat, imp in zip(bars, summary[top_feat_col], summary[top_imp_col]):
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f"{feat}\n{imp:.2f}",
        ha='center',
        va='bottom',
        fontsize=7
    )

plt.tight_layout()
plt.savefig(PLOTS_DIR / "top_feature_per_crypto.png", dpi=300)
plt.close()

# -----------------------------
# 4. Heatmap: feature importance (crypto × macro)
# -----------------------------
importance_mats = []

for symbol in CRYPTO_SYMBOLS:
    f = RESULTS_DIR / f"{symbol}_feature_importance.csv"
    if not f.exists():
        continue

    imp_df = pd.read_csv(f)

    # Accept either 'importance_mean' (from CV) or 'importance'
    if 'importance_mean' in imp_df.columns:
        val_col = 'importance_mean'
    elif 'importance' in imp_df.columns:
        val_col = 'importance'
    else:
        continue

    imp_df = imp_df[['feature', val_col]].set_index('feature')
    imp_df.columns = [symbol]
    importance_mats.append(imp_df)

if importance_mats:
    importance_full = pd.concat(importance_mats, axis=1).fillna(0.0)

    # Order features by average importance
    importance_full['avg_importance'] = importance_full.mean(axis=1)
    importance_full = importance_full.sort_values('avg_importance', ascending=False)
    importance_full = importance_full.drop(columns=['avg_importance'])

    plt.figure(figsize=(10, max(4, 0.4 * len(importance_full))))
    sns.heatmap(
        importance_full,
        annot=False,
        cmap='viridis',
        cbar_kws={'label': 'Feature importance'}
    )
    plt.title("Feature Importance Heatmap (Macro Variables × Cryptos)")
    plt.xlabel("Cryptocurrency")
    plt.ylabel("Macro Variable")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "feature_importance_heatmap.png", dpi=300)
    plt.close()
else:
    print("No per-crypto feature importance files found, skipping heatmap.")

print(f"Plots saved to: {PLOTS_DIR}")
