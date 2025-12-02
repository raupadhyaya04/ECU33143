import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Import config from analysis
from macro_analysis import (
    CRYPTO_SYMBOLS, MACRO_VAR_SELECTION, IRF_HORIZON,
    RESULTS_DIR
)

# Global style
sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.figsize": (12, 6),
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.titleweight": "bold"
})

VIZ_DIR = Path("./Crypto-Macro/Results")
VIZ_DIR.mkdir(parents=True, exist_ok=True)

# Subdirectories
(VIZ_DIR / "Heatmaps").mkdir(exist_ok=True)
(VIZ_DIR / "IRF").mkdir(exist_ok=True)


# ---------- VISUALIZATION FUNCTIONS ----------

def plot_granger_heatmap(df_granger: pd.DataFrame, save_path: Path):
    """Create heatmap of Granger causality p-values."""
    # Pivot for heatmap
    heatmap_data = df_granger.pivot(
        index="Crypto",
        columns="Macro Variable",
        values="Min p-Value"
    )

    # Rename macro variables for cleaner display
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

    # Use constrained layout so colorbar / labels don't spill
    fig, ax = plt.subplots(figsize=(11, 7), layout="constrained")

    sns.heatmap(
        heatmap_data,
        annot=True, fmt=".3f",
        cmap="RdYlGn_r",
        cbar_kws={'label': 'p-value'},
        linewidths=0.5, linecolor='gray',
        annot_kws={"size": 8},
        vmin=0, vmax=0.10,
        ax=ax
    )

    ax.set_title(
        "Granger Causality: Macro Variables → Crypto Returns",
        fontsize=14, weight='bold', pad=8
    )
    ax.set_xlabel("Macro Variable", fontsize=12, labelpad=8)
    ax.set_ylabel("Cryptocurrency", fontsize=12, labelpad=8)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=11)

    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Granger heatmap saved to {save_path}")


def plot_irf_for_crypto(crypto_name: str, irf_data: dict, save_path: Path):
    """
    Plot impulse response functions for one cryptocurrency in a clear way.

    irf_data structure (expected):
      irf_data = {
        'irf_data': {macro_name: array_like of IRF values, ...}
        # OPTIONAL if you saved CIs:
        # 'irf_lower': {macro_name: array_like of lower CI values, ...}
        # 'irf_upper': {macro_name: array_like of upper CI values, ...}
        # OPTIONAL: 'response_labels': list of variable names along 2nd axis
        'impulse_vars': list of macro variable names,
        'horizon': int horizon
      }

    The plot will show, for each macro shock:
      - Solid blue line: IRF of the selected crypto return
      - Light blue band: 95% confidence interval (if provided)
      - Black dashed line: zero baseline
    """
    impulse_vars = irf_data["impulse_vars"]
    irf_vals = irf_data["irf_data"]
    horizon = irf_data["horizon"]

    irf_lower = irf_data.get("irf_lower", None)
    irf_upper = irf_data.get("irf_upper", None)
    response_labels = irf_data.get("response_labels", None)

    n_vars = len(impulse_vars)
    max_cols = 3
    n_cols = min(max_cols, n_vars)
    n_rows = math.ceil(n_vars / n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.8 * n_cols, 3.3 * n_rows),
        layout="constrained"
    )

    # Normalise axes to 2D array
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = np.array([[ax] for ax in axes])

    fig.suptitle(
        f"Impulse Responses of {crypto_name} Returns to Macro Shocks",
        fontsize=14, fontweight="bold"
    )

    x_vals = np.arange(horizon + 1)
    axes_flat = axes.ravel()

    for i, macro in enumerate(impulse_vars):
        ax = axes_flat[i]

        # Raw IRF array for this shock
        y_raw = np.asarray(irf_vals[macro])

        # If IRF is 2D (horizon x n_responses), pick the column for this crypto
        # or fall back to the first column.
        if y_raw.ndim == 2:
            if response_labels is not None and crypto_name in response_labels:
                idx = response_labels.index(crypto_name)
                y = y_raw[:, idx]
            else:
                y = y_raw[:, 0]
        else:
            y = y_raw

        # Main IRF line (one line only)
        ax.plot(
            x_vals, y,
            color="tab:blue", linewidth=2,
            label=f"IRF: {crypto_name} return"
        )

        # Zero line
        ax.axhline(
            0, color="black", linewidth=0.8,
            linestyle="--", alpha=0.7, label="Zero baseline"
        )

        # Optional confidence band (must also be 1D)
        if (
            irf_lower is not None and irf_upper is not None
            and macro in irf_lower and macro in irf_upper
        ):
            low_raw = np.asarray(irf_lower[macro])
            high_raw = np.asarray(irf_upper[macro])

            if low_raw.ndim == 2:
                if response_labels is not None and crypto_name in response_labels:
                    idx = response_labels.index(crypto_name)
                    y_low = low_raw[:, idx]
                    y_high = high_raw[:, idx]
                else:
                    y_low = low_raw[:, 0]
                    y_high = high_raw[:, 0]
            else:
                y_low = low_raw
                y_high = high_raw

            ax.fill_between(
                x_vals, y_low, y_high,
                color="tab:blue", alpha=0.2,
                label="95% confidence band"
            )

        macro_clean = macro.replace("^", "").replace("=F", "")
        ax.set_title(f"{macro_clean} shock → {crypto_name} return", fontsize=10)
        ax.set_xlabel("Days after shock", fontsize=9)
        ax.set_ylabel("Response", fontsize=9)
        ax.grid(True, alpha=0.3)

        # One small legend per panel
        ax.legend(loc="upper right", fontsize=7, frameon=True)

    # Hide unused subplots if any
    for j in range(len(impulse_vars), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ IRF plot saved for {crypto_name}")


def plot_ols_significance_heatmap(ols_summary: pd.DataFrame, save_path: Path):
    """
    Create heatmap showing which OLS coefficients are significant.
    """
    # Filter to show only key variables (remove lags for cleaner view)
    df_filtered = ols_summary[~ols_summary['Variable'].str.contains('lag', case=False)]

    pivot = df_filtered.pivot_table(
        index='Variable',
        columns='Crypto',
        values='p-Value',
        aggfunc='first'
    )

    fig, ax = plt.subplots(figsize=(12, 8), layout="constrained")

    sns.heatmap(
        pivot,
        annot=True, fmt=".3f",
        cmap="RdYlGn_r",
        cbar_kws={'label': 'p-value'},
        linewidths=0.5, linecolor='gray',
        annot_kws={"size": 7},
        vmin=0, vmax=0.10,
        ax=ax
    )

    ax.set_title(
        "OLS Regression Significance: Macro Variables → Crypto Returns",
        fontsize=14, weight='bold', pad=8
    )
    ax.set_xlabel("Cryptocurrency", fontsize=12, labelpad=8)
    ax.set_ylabel("Macro Variable (contemporaneous)", fontsize=12, labelpad=8)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ OLS significance heatmap saved to {save_path}")


def plot_vif_comparison(ols_results: dict, save_path: Path):
    """
    Plot VIF comparison across all cryptocurrencies.
    """
    fig, axes = plt.subplots(
        2, 5,
        figsize=(20, 8),
        layout="constrained"
    )
    axes = axes.flatten()

    for idx, (crypto, results) in enumerate(ols_results.items()):
        if idx >= len(axes):
            break

        ax = axes[idx]
        vif_df = results['vif'].head(10)

        # Color code: red if VIF > 10 (multicollinearity concern)
        colors = ['red' if v > 10 else 'steelblue' for v in vif_df['VIF']]

        ax.barh(vif_df['feature'], vif_df['VIF'], color=colors)
        ax.axvline(10, color='darkred', linestyle='--', linewidth=2, label='VIF=10')
        ax.set_xlabel('VIF')
        ax.set_title(crypto, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        ax.invert_yaxis()

        if idx == 0:
            ax.legend(loc="lower right", fontsize=8, frameon=True)

    # Hide any unused axes if fewer than 10 cryptos
    for j in range(idx + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Variance Inflation Factors (VIF) — Top 10 Variables per Crypto",
        fontsize=16, fontweight='bold'
    )

    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ VIF comparison plot saved to {save_path}")


def plot_coefficient_comparison(ols_summary: pd.DataFrame, variable_name: str, save_path: Path):
    """
    Plot coefficient values across all cryptos for a specific macro variable.
    """
    # Filter for specific variable (contemporaneous, no lags)
    df_var = ols_summary[
        (ols_summary['Variable'] == variable_name) &
        (~ols_summary['Variable'].str.contains('lag', case=False))
    ].copy()

    if df_var.empty:
        print(f"No data found for variable: {variable_name}")
        return

    df_var = df_var.sort_values('Coef')

    # Color by significance
    colors = ['green' if sig == '✅' else 'gray' for sig in df_var['Significant']]

    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    ax.barh(df_var['Crypto'], df_var['Coef'], color=colors)
    ax.axvline(0, color='black', linewidth=1, linestyle='--')
    ax.set_xlabel('Coefficient Value', fontsize=12)
    ax.set_ylabel('Cryptocurrency', fontsize=12)
    ax.set_title(
        f"Impact of {variable_name} on Crypto Returns",
        fontsize=14, fontweight='bold'
    )
    ax.grid(True, alpha=0.3, axis='x')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='Significant (p<0.05)'),
        Patch(facecolor='gray', label='Not Significant')
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9, frameon=True)

    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Coefficient comparison plot saved for {variable_name}")


def create_summary_dashboard(ols_summary: pd.DataFrame, granger_df: pd.DataFrame, save_path: Path):
    """
    Create comprehensive dashboard with key metrics.
    """
    fig = plt.figure(figsize=(18, 10), layout="constrained")
    gs = fig.add_gridspec(2, 3)

    # 1. Significant OLS relationships count
    ax1 = fig.add_subplot(gs[0, 0])
    sig_count = ols_summary[ols_summary['Significant'] == '✅'].groupby('Crypto').size()
    sig_count = sig_count.sort_values(ascending=False)
    ax1.bar(sig_count.index, sig_count.values, color='steelblue')
    ax1.set_title("# of Significant OLS Coefficients", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Count")
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')

    # 2. Significant Granger relationships count
    ax2 = fig.add_subplot(gs[0, 1])
    granger_sig = granger_df[granger_df['Significant'] == '✅'].groupby('Crypto').size()
    granger_sig = granger_sig.sort_values(ascending=False)
    ax2.bar(granger_sig.index, granger_sig.values, color='coral')
    ax2.set_title("# of Significant Granger Relationships", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Count")
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. Most influential macro variables (by OLS significance)
    ax3 = fig.add_subplot(gs[0, 2])
    macro_influence = ols_summary[ols_summary['Significant'] == '✅'].groupby('Variable').size()
    macro_influence = macro_influence.sort_values(ascending=False).head(10)
    ax3.barh(macro_influence.index, macro_influence.values, color='purple')
    ax3.set_title("Most Influential Macro Variables (OLS)", fontsize=12, fontweight='bold')
    ax3.set_xlabel("# of Significant Relationships")
    ax3.invert_yaxis()
    ax3.grid(True, alpha=0.3, axis='x')

    # 4. Granger causality mini-heatmap (top variables)
    ax4 = fig.add_subplot(gs[1, :2])
    top_macro = granger_df.groupby('Macro Variable')['Min p-Value'].mean().sort_values().head(6).index
    granger_subset = granger_df[granger_df['Macro Variable'].isin(top_macro)]
    pivot = granger_subset.pivot(index='Crypto', columns='Macro Variable', values='Min p-Value')
    sns.heatmap(
        pivot, annot=True, fmt=".3f",
        cmap="RdYlGn_r", vmin=0, vmax=0.10,
        ax=ax4, cbar_kws={'label': 'p-value'}
    )
    ax4.set_title("Granger Causality (Top 6 Macro Variables)", fontsize=12, fontweight='bold')
    ax4.set_xlabel("")
    ax4.set_ylabel("")

    # 5. Summary statistics text
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis("off")

    total_ols_tests = len(ols_summary)
    sig_ols_tests = (ols_summary['Significant'] == '✅').sum()
    total_granger_tests = len(granger_df)
    sig_granger_tests = (granger_df['Significant'] == '✅').sum()

    summary_text = f"""
MACRO-CRYPTO ANALYSIS SUMMARY
==============================

OLS Regression:
  Total Tests: {total_ols_tests}
  Significant (p<0.05): {sig_ols_tests}
  Significance Rate: {sig_ols_tests/total_ols_tests*100:.1f}%

Granger Causality:
  Total Tests: {total_granger_tests}
  Significant (p<0.05): {sig_granger_tests}
  Significance Rate: {sig_granger_tests/total_granger_tests*100:.1f}%

Top Macro Influences (OLS):
  {macro_influence.index[0]}: {macro_influence.iloc[0]} cryptos
  {macro_influence.index[1]}: {macro_influence.iloc[1]} cryptos
  {macro_influence.index[2]}: {macro_influence.iloc[2]} cryptos
"""
    ax5.text(
        0.05, 0.5, summary_text,
        fontsize=10, family="monospace",
        va="center",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    )

    fig.suptitle(
        "Cryptocurrency-Macro Analysis Dashboard",
        fontsize=16, fontweight='bold'
    )

    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Summary dashboard saved to {save_path}")


# ---------- VISUALIZATION RUNNER ----------

def generate_all_visualizations(ols_results: dict, granger_df: pd.DataFrame, var_irf_results: dict):
    """
    Generate all visualizations from analysis results.
    """
    print("\n" + "="*90)
    print("GENERATING VISUALIZATIONS")
    print("="*90)

    # Load OLS summary
    ols_summary = pd.read_csv(RESULTS_DIR / "macro_crypto_regression_summary.csv")

    # 1. Granger Causality Heatmap
    print("\n[VIZ] Creating Granger causality heatmap...")
    plot_granger_heatmap(granger_df, save_path=VIZ_DIR / "Heatmaps/granger_heatmap.png")

    # 2. OLS Significance Heatmap
    print("\n[VIZ] Creating OLS significance heatmap...")
    plot_ols_significance_heatmap(ols_summary, save_path=VIZ_DIR / "Heatmaps/ols_significance_heatmap.png")

    # 3. VIF Comparison
    print("\n[VIZ] Creating VIF comparison plot...")
    plot_vif_comparison(ols_results, save_path=VIZ_DIR / "vif_comparison.png")

    # 4. IRF Plots for each cryptocurrency
    print("\n[VIZ] Creating IRF plots...")
    for crypto, irf_data in var_irf_results.items():
        plot_irf_for_crypto(
            crypto_name=crypto,
            irf_data=irf_data,
            save_path=VIZ_DIR / f"IRF/{crypto}_IRF.png"
        )

    # 5. Coefficient comparison for key variables
    print("\n[VIZ] Creating coefficient comparison plots...")
    key_variables = ['^GSPC', '^VIX', 'M2 Money Supply', 'Fed Funds Rate']
    for var in key_variables:
        if var in ols_summary['Variable'].values:
            plot_coefficient_comparison(
                ols_summary, var,
                save_path=VIZ_DIR / f"coefficient_comparison_{var.replace('^', '').replace('=F', '')}.png"
            )

    # 6. Summary Dashboard
    print("\n[VIZ] Creating summary dashboard...")
    create_summary_dashboard(ols_summary, granger_df, save_path=VIZ_DIR / "summary_dashboard.png")

    print("\n" + "="*90)
    print("VISUALIZATION COMPLETE")
    print("="*90)
    print(f"\nSaved {len(list(VIZ_DIR.rglob('*.png')))} figures to {VIZ_DIR}/")
    print(f"\nKey visualizations:")
    print(f"  - Granger heatmap: {VIZ_DIR}/Heatmaps/granger_heatmap.png")
    print(f"  - OLS significance: {VIZ_DIR}/Heatmaps/ols_significance_heatmap.png")
    print(f"  - IRF plots: {VIZ_DIR}/IRF/*_IRF.png")
    print(f"  - Summary dashboard: {VIZ_DIR}/summary_dashboard.png")


def generate_visualizations_from_saved_data():
    """
    Load saved analysis results and generate visualizations.
    Looks for:
      - macro_crypto_regression_summary.csv
      - granger_causality_results.csv
      - *_irf_data.csv (anywhere under RESULTS_DIR)
    Saves IRF PNGs into: Crypto-Macro/Results/IRF/
    """
    # Load summary data
    ols_summary_path = RESULTS_DIR / "macro_crypto_regression_summary.csv"
    granger_path = RESULTS_DIR / "granger_causality_results.csv"

    if not ols_summary_path.exists():
        print(f"[ERR] Missing OLS summary: {ols_summary_path}")
        return
    if not granger_path.exists():
        print(f"[ERR] Missing Granger results: {granger_path}")
        return

    ols_summary = pd.read_csv(ols_summary_path)
    granger_df = pd.read_csv(granger_path)

    # Load IRF data: search recursively for *_irf_data.csv under RESULTS_DIR
    var_irf_results = {}
    print(f"\n[INFO] Scanning for IRF CSVs under {RESULTS_DIR} ...")
    for irf_file in RESULTS_DIR.rglob("*_irf_data.csv"):
        crypto = irf_file.stem.replace("_irf_data", "")
        try:
            irf_df = pd.read_csv(irf_file)
            var_irf_results[crypto] = {
                "irf_data": irf_df.to_dict("list"),
                "impulse_vars": MACRO_VAR_SELECTION,
                "horizon": IRF_HORIZON
            }
            print(f"  - Found IRF data for {crypto} at {irf_file}")
        except Exception as e:
            print(f"  ! Failed to load {irf_file}: {e}")

    if not var_irf_results:
        print("[WARN] No *_irf_data.csv files found; IRF plots will not be created.")

    # Granger heatmap
    print("\n[VIZ] Creating Granger causality heatmap...")
    plot_granger_heatmap(granger_df, save_path=VIZ_DIR / "Heatmaps/granger_heatmap.png")

    # OLS significance heatmap
    print("\n[VIZ] Creating OLS significance heatmap...")
    plot_ols_significance_heatmap(
        ols_summary, save_path=VIZ_DIR / "Heatmaps/ols_significance_heatmap.png"
    )

    # IRF plots
    print("\n[VIZ] Creating IRF plots...")
    for crypto, irf_data in var_irf_results.items():
        out_path = IRF_PLOT_DIR / f"{crypto}_IRF.png"
        plot_irf_for_crypto(
            crypto_name=crypto,
            irf_data=irf_data,
            save_path=out_path
        )

    # Summary dashboard
    print("\n[VIZ] Creating summary dashboard...")
    create_summary_dashboard(
        ols_summary, granger_df,
        save_path=VIZ_DIR / "summary_dashboard.png"
    )

    print(f"\n✅ Visualization complete! Saved PNGs under {VIZ_DIR}/")


if __name__ == "__main__":
    # Option 1: Load from saved data (faster, partial viz)
    generate_visualizations_from_saved_data()

    # Option 2: Use fresh analysis results (full viz)
    # from macro_analysis import run_all_analyses
    # ols_results, granger_df, var_irf_results, df_macro, crypto_data = run_all_analyses()
    # generate_all_visualizations(ols_results, granger_df, var_irf_results)