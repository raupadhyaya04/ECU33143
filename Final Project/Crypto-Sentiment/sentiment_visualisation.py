import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Import analysis functions for VaR calculation
from sentiment_analysis import (
    rolling_historical_var, conditional_var_quantreg, 
    SYMBOLS, ALPHAS, ROLL_WIN, OUT_DIR
)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

VIZ_DIR = Path("./Crypto-Sentiment/Results/Visualisations")
VIZ_DIR.mkdir(parents=True, exist_ok=True)


# ---------- VISUALIZATION FUNCTIONS ----------

def plot_var_backtest(df: pd.DataFrame, symbol: str, alpha: float, save_path: Path):
    """Plot returns vs VaR threshold with breach highlighting."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    dates = pd.to_datetime(df["date"])
    returns = df["ret1d"].values
    var_hist = rolling_historical_var(df["ret1d"], alpha=alpha, window=ROLL_WIN).values
    var_qr = conditional_var_quantreg(df, alpha=alpha).values
    
    # Top panel: Returns and VaR thresholds
    ax1.plot(dates, returns * 100, label="Daily Returns (%)", color="gray", alpha=0.6, linewidth=0.8)
    ax1.plot(dates, -var_hist * 100, label=f"Historical VaR {int(alpha*100)}%", color="blue", linewidth=1.5)
    ax1.plot(dates, -var_qr * 100, label=f"Conditional VaR {int(alpha*100)}%", color="orange", linewidth=1.5)
    
    # Highlight breaches
    breaches_hist = returns < -var_hist
    breaches_qr = returns < -var_qr
    ax1.scatter(dates[breaches_hist], returns[breaches_hist] * 100, 
                color="red", s=30, marker="o", label="Historical VaR Breach", zorder=5)
    ax1.scatter(dates[breaches_qr], returns[breaches_qr] * 100, 
                color="darkred", s=30, marker="x", label="Conditional VaR Breach", zorder=5)
    
    ax1.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax1.set_ylabel("Returns / VaR (%)")
    ax1.set_title(f"{symbol} - VaR Backtest ({int(alpha*100)}% Confidence)", fontsize=14, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Bottom panel: Sentiment
    ax2.plot(dates, df["reddit_sentiment"], label="Reddit Sentiment", color="green", alpha=0.7, linewidth=1)
    ax2.plot(dates, df["sent_7d_ma"], label="7-Day MA Sentiment", color="darkgreen", linewidth=1.5)
    ax2.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax2.set_ylabel("Sentiment Score")
    ax2.set_xlabel("Date")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_sentiment_return_scatter(df: pd.DataFrame, symbol: str, save_path: Path):
    """Scatter plot: Sentiment vs next-day returns with regression line."""
    df_copy = df.copy()
    df_copy["ret1d_next"] = df_copy["ret1d"].shift(-1)
    df_clean = df_copy.dropna(subset=["reddit_sentiment", "ret1d_next"])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.scatter(df_clean["reddit_sentiment"], df_clean["ret1d_next"] * 100, 
               alpha=0.4, s=20, color="steelblue")
    
    # Add regression line
    z = np.polyfit(df_clean["reddit_sentiment"], df_clean["ret1d_next"] * 100, 1)
    p = np.poly1d(z)
    x_line = np.linspace(df_clean["reddit_sentiment"].min(), df_clean["reddit_sentiment"].max(), 100)
    ax.plot(x_line, p(x_line), "r--", linewidth=2, label=f"Fit: y={z[0]:.3f}x+{z[1]:.3f}")
    
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Reddit Sentiment Score", fontsize=12)
    ax.set_ylabel("Next-Day Return (%)", fontsize=12)
    ax.set_title(f"{symbol} - Sentiment vs Returns", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_rolling_correlation(df: pd.DataFrame, symbol: str, save_path: Path, window=30):
    """Rolling correlation between sentiment and returns."""
    df_copy = df.copy()
    df_copy["date_dt"] = pd.to_datetime(df_copy["date"])
    df_copy = df_copy.set_index("date_dt").sort_index()
    
    rolling_corr = df_copy["reddit_sentiment"].rolling(window).corr(df_copy["ret1d"])
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(rolling_corr.index, rolling_corr, color="purple", linewidth=1.5)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.fill_between(rolling_corr.index, 0, rolling_corr, where=(rolling_corr > 0), 
                     color="green", alpha=0.3, label="Positive Correlation")
    ax.fill_between(rolling_corr.index, 0, rolling_corr, where=(rolling_corr < 0), 
                     color="red", alpha=0.3, label="Negative Correlation")
    
    ax.set_ylabel(f"Rolling Correlation ({window}-day)", fontsize=12)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_title(f"{symbol} - Rolling Correlation: Sentiment vs Returns", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_granger_heatmap(gc_df: pd.DataFrame, save_path: Path):
    """Heatmap of Granger causality p-values across all symbols."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Sentiment -> Returns
    senti_to_ret = gc_df.pivot_table(index="symbol", values="p_senti_to_ret_min", aggfunc="first")
    sns.heatmap(senti_to_ret, annot=True, fmt=".4f", cmap="RdYlGn_r", 
                vmin=0, vmax=0.10, cbar_kws={"label": "p-value"}, ax=ax1)
    ax1.set_title("Granger: Sentiment → Returns", fontsize=14, fontweight="bold")
    ax1.set_ylabel("")
    
    # Returns -> Sentiment
    ret_to_senti = gc_df.pivot_table(index="symbol", values="p_ret_to_senti_min", aggfunc="first")
    sns.heatmap(ret_to_senti, annot=True, fmt=".4f", cmap="RdYlGn_r", 
                vmin=0, vmax=0.10, cbar_kws={"label": "p-value"}, ax=ax2)
    ax2.set_title("Granger: Returns → Sentiment", fontsize=14, fontweight="bold")
    ax2.set_ylabel("")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_var_performance_comparison(var_df: pd.DataFrame, save_path: Path):
    """Bar chart comparing VaR model performance across symbols."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    for idx, alpha in enumerate(ALPHAS):
        subset = var_df[var_df["alpha"] == alpha]
        
        # Exception rates
        ax = axes[idx, 0]
        pivot = subset.pivot(index="symbol", columns="method", values="rate")
        pivot.plot(kind="bar", ax=ax, color=["steelblue", "coral"])
        ax.axhline(1 - alpha, color="red", linestyle="--", linewidth=2, label=f"Expected Rate ({1-alpha:.2%})")
        ax.set_ylabel("Exception Rate", fontsize=11)
        ax.set_title(f"VaR Exception Rates ({int(alpha*100)}% Confidence)", fontsize=12, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        
        # Kupiec p-values
        ax = axes[idx, 1]
        pivot_p = subset.pivot(index="symbol", columns="method", values="p_value")
        pivot_p.plot(kind="bar", ax=ax, color=["steelblue", "coral"])
        ax.axhline(0.05, color="red", linestyle="--", linewidth=2, label="5% Significance")
        ax.set_ylabel("Kupiec Test p-value", fontsize=11)
        ax.set_title(f"VaR Kupiec Test p-values ({int(alpha*100)}% Confidence)", fontsize=12, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_summary_dashboard(var_df: pd.DataFrame, gc_df: pd.DataFrame, save_path: Path):
    """Comprehensive dashboard with key metrics."""
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. VaR Exception Rate Comparison (95%)
    ax1 = fig.add_subplot(gs[0, :2])
    var_95 = var_df[var_df["alpha"] == 0.95]
    pivot = var_95.pivot(index="symbol", columns="method", values="rate")
    pivot.plot(kind="bar", ax=ax1, color=["steelblue", "coral"], width=0.7)
    ax1.axhline(0.05, color="red", linestyle="--", linewidth=2, label="Expected (5%)")
    ax1.set_title("VaR Exception Rates (95% Confidence)", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Exception Rate")
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha="right")
    
    # 2. Granger p-values (Sentiment → Returns)
    ax2 = fig.add_subplot(gs[0, 2])
    senti_to_ret = gc_df.sort_values("p_senti_to_ret_min")
    colors = ["green" if p < 0.05 else "gray" for p in senti_to_ret["p_senti_to_ret_min"]]
    ax2.barh(senti_to_ret["symbol"], senti_to_ret["p_senti_to_ret_min"], color=colors)
    ax2.axvline(0.05, color="red", linestyle="--", linewidth=2)
    ax2.set_xlabel("p-value")
    ax2.set_title("Sentiment → Returns", fontsize=12, fontweight="bold")
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3, axis="x")
    
    # 3. VaR Exception Rate Comparison (99%)
    ax3 = fig.add_subplot(gs[1, :2])
    var_99 = var_df[var_df["alpha"] == 0.99]
    pivot_99 = var_99.pivot(index="symbol", columns="method", values="rate")
    pivot_99.plot(kind="bar", ax=ax3, color=["steelblue", "coral"], width=0.7)
    ax3.axhline(0.01, color="red", linestyle="--", linewidth=2, label="Expected (1%)")
    ax3.set_title("VaR Exception Rates (99% Confidence)", fontsize=13, fontweight="bold")
    ax3.set_ylabel("Exception Rate")
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis="y")
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha="right")
    
    # 4. Granger p-values (Returns → Sentiment)
    ax4 = fig.add_subplot(gs[1, 2])
    ret_to_senti = gc_df.sort_values("p_ret_to_senti_min")
    colors2 = ["green" if p < 0.05 else "gray" for p in ret_to_senti["p_ret_to_senti_min"]]
    ax4.barh(ret_to_senti["symbol"], ret_to_senti["p_ret_to_senti_min"], color=colors2)
    ax4.axvline(0.05, color="red", linestyle="--", linewidth=2)
    ax4.set_xlabel("p-value")
    ax4.set_title("Returns → Sentiment", fontsize=12, fontweight="bold")
    ax4.invert_yaxis()
    ax4.grid(True, alpha=0.3, axis="x")
    
    # 5. Selected VAR Lags
    ax5 = fig.add_subplot(gs[2, 0])
    gc_df_sorted = gc_df.sort_values("selected_lag", ascending=False)
    ax5.barh(gc_df_sorted["symbol"], gc_df_sorted["selected_lag"], color="purple")
    ax5.set_xlabel("Optimal Lag (days)")
    ax5.set_title("VAR Model Lag Selection", fontsize=12, fontweight="bold")
    ax5.invert_yaxis()
    ax5.grid(True, alpha=0.3, axis="x")
    
    # 6. Kupiec Test Pass Rate
    ax6 = fig.add_subplot(gs[2, 1])
    pass_rate = var_df.groupby("method").apply(
        lambda x: (x["p_value"] > 0.05).sum() / len(x) * 100
    )
    ax6.bar(pass_rate.index, pass_rate.values, color=["steelblue", "coral"])
    ax6.set_ylabel("Pass Rate (%)")
    ax6.set_title("Kupiec Test Pass Rate", fontsize=12, fontweight="bold")
    ax6.set_ylim([0, 100])
    ax6.grid(True, alpha=0.3, axis="y")
    
    # 7. Summary Statistics Table
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis("off")
    summary_text = f"""
    SUMMARY STATISTICS
    ==================
    Total Symbols: {len(SYMBOLS)}
    
    Granger Causality (p<0.05):
      Sentiment → Returns: {(gc_df['p_senti_to_ret_min'] < 0.05).sum()}/{len(gc_df)}
      Returns → Sentiment: {(gc_df['p_ret_to_senti_min'] < 0.05).sum()}/{len(gc_df)}
    
    VaR Performance:
      Avg Exception Rate (95%): {var_df[var_df['alpha']==0.95]['rate'].mean():.2%}
      Avg Exception Rate (99%): {var_df[var_df['alpha']==0.99]['rate'].mean():.2%}
    
    Kupiec Test Pass (p>0.05):
      Historical: {((var_df[var_df['method']=='historical']['p_value'] > 0.05).sum() / len(var_df[var_df['method']=='historical']) * 100):.1f}%
      Conditional: {((var_df[var_df['method']=='quantile_reg']['p_value'] > 0.05).sum() / len(var_df[var_df['method']=='quantile_reg']) * 100):.1f}%
    """
    ax7.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment="center", 
             family="monospace", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    
    fig.suptitle("Cryptocurrency Sentiment & Risk Analysis Dashboard", 
                 fontsize=16, fontweight="bold", y=0.98)
    
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# ---------- VISUALIZATION RUNNER ----------
def generate_all_visualizations(symbol_data: dict, var_df: pd.DataFrame, gc_df: pd.DataFrame):
    """
    Generate all visualizations from analysis results.
    
    Args:
        symbol_data: dict of {symbol: dataframe} from analysis
        var_df: VaR backtest summary dataframe
        gc_df: Granger causality summary dataframe
    """
    print("\n[VIZ] Generating visualizations...")
    
    # Per-symbol plots
    for sym, df in symbol_data.items():
        try:
            print(f"[VIZ] Creating plots for {sym}...")
            
            # VaR backtest plots for both alphas
            for a in ALPHAS:
                plot_var_backtest(df, sym, alpha=a, 
                                save_path=VIZ_DIR / f"{sym}_var_backtest_{int(a*100)}.png")
            
            # Sentiment-return scatter
            plot_sentiment_return_scatter(df, sym, 
                                         save_path=VIZ_DIR / f"{sym}_sentiment_return_scatter.png")
            
            # Rolling correlation
            plot_rolling_correlation(df, sym, 
                                    save_path=VIZ_DIR / f"{sym}_rolling_correlation.png")
            
            print(f"[OK] {sym} visualizations complete.")
        except Exception as e:
            print(f"[ERR] {sym} visualization failed: {e}")
    
    # Aggregate visualizations
    print("\n[VIZ] Creating aggregate visualizations...")
    plot_granger_heatmap(gc_df, save_path=VIZ_DIR / "granger_heatmap.png")
    plot_var_performance_comparison(var_df, save_path=VIZ_DIR / "var_performance_comparison.png")
    create_summary_dashboard(var_df, gc_df, save_path=VIZ_DIR / "summary_dashboard.png")
    
    print(f"\n✅ Visualization Complete!")
    print(f"Saved {len(list(VIZ_DIR.glob('*.png')))} figures to {VIZ_DIR}/")
    print(f"\nKey visualizations:")
    print(f"  - Per-symbol VaR backtests: {VIZ_DIR}/*_var_backtest_*.png")
    print(f"  - Sentiment-return analysis: {VIZ_DIR}/*_sentiment_return_scatter.png")
    print(f"  - Rolling correlations: {VIZ_DIR}/*_rolling_correlation.png")
    print(f"  - Granger causality heatmap: {VIZ_DIR}/granger_heatmap.png")
    print(f"  - Summary dashboard: {VIZ_DIR}/summary_dashboard.png")


def generate_visualizations_from_saved_data():
    """
    Load saved analysis results and generate visualizations.
    Useful when you don't want to re-run analysis.
    """
    # Load summary CSVs
    var_df = pd.read_csv(OUT_DIR / "var_backtests_summary.csv")
    gc_df = pd.read_csv(OUT_DIR / "granger_summary.csv")
    
    # Load per-symbol data
    symbol_data = {}
    for sym in SYMBOLS:
        try:
            df = pd.read_csv(OUT_DIR / f"{sym}_var_series.csv")
            symbol_data[sym] = df
        except FileNotFoundError:
            print(f"[WARN] Missing data for {sym}, skipping...")
    
    generate_all_visualizations(symbol_data, var_df, gc_df)


if __name__ == "__main__":
    # Option 1: Load from saved data (faster)
    generate_visualizations_from_saved_data()
    
    # Option 2: Import and use fresh analysis results
    # from sentiment_analysis import run_analysis
    # var_df, gc_df, symbol_data = run_analysis()
    # generate_all_visualizations(symbol_data, var_df, gc_df)