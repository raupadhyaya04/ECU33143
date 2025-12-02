"""
sentiment_visualisation.py

Visualisations for crypto sentiment analysis:
  - Sentiment vs next-day returns (scatter + OLS fit)
  - Rolling correlation between sentiment and returns
  - Granger p-value heatmaps and dashboard

Relies on outputs from sentiment_analysis.py:
  - granger_summary.csv
  - {symbol}_sentiment_series.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter

# Try to import config from analysis module
try:
    from sentiment_analysis import SYMBOLS, OUT_DIR
except Exception:
    SYMBOLS = [
        "BTCUSDT","ETHUSDT","BNBUSDT","ADAUSDT","XRPUSDT",
        "SOLUSDT","DOGEUSDT","DOTUSDT","AVAXUSDT","LINKUSDT"
    ]
    OUT_DIR = Path("./Crypto-Sentiment/Results/Analysis")

# Style
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "figure.figsize": (12, 6),
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.titleweight": "bold"
})

VIZ_DIR = Path("./Crypto-Sentiment/Results/Visualisations/Sentiment")
VIZ_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------
# Helper: tidy date axis
# ---------------------
def tidy_date_axis(ax):
    locator = AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(ConciseDateFormatter(locator))
    for label in ax.get_xticklabels():
        label.set_rotation(0)
        label.set_horizontalalignment("center")


# ---------------------
# Per-symbol visuals
# ---------------------
def plot_sentiment_return_scatter(df: pd.DataFrame, symbol: str, save_path: Path):
    """Scatter: sentiment vs next-day return, with linear fit."""
    df = df.copy()
    df["ret1d_next"] = df["ret1d"].shift(-1)

    ok = df.dropna(subset=["reddit_sentiment", "ret1d_next"])
    if ok.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(
        ok["reddit_sentiment"], ok["ret1d_next"] * 100,
        alpha=0.35, s=18, color="steelblue", edgecolor="none"
    )

    # Add simple OLS fit
    if len(ok) > 5:
        z = np.polyfit(
            ok["reddit_sentiment"].values,
            (ok["ret1d_next"] * 100).values,
            1
        )
        x_line = np.linspace(
            ok["reddit_sentiment"].min(),
            ok["reddit_sentiment"].max(),
            100
        )
        y_line = z[0] * x_line + z[1]
        ax.plot(
            x_line, y_line,
            "r--", linewidth=2,
            label=f"Fit: y={z[0]:.3f}x+{z[1]:.3f}"
        )

    ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Reddit Sentiment Score")
    ax.set_ylabel("Next-Day Return (%)")
    ax.set_title(f"{symbol} — Sentiment vs Next-Day Returns")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_rolling_correlation(df: pd.DataFrame, symbol: str, save_path: Path, window: int = 30):
    """Rolling correlation between sentiment and returns (shaded by sign)."""
    df = df.copy()
    df["date_dt"] = pd.to_datetime(df["date"])
    df = df.set_index("date_dt").sort_index()
    roll = df["reddit_sentiment"].rolling(window).corr(df["ret1d"])

    if roll.dropna().empty:
        return

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(roll.index, roll, color="purple", linewidth=1.4, label=f"Rolling Corr ({window}d)")
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.fill_between(roll.index, 0, roll, where=(roll > 0), color="green", alpha=0.25, label="Positive")
    ax.fill_between(roll.index, 0, roll, where=(roll < 0), color="red", alpha=0.25, label="Negative")

    tidy_date_axis(ax)
    ax.set_ylabel(f"Correlation ({window}-day)")
    ax.set_xlabel("Date")
    ax.set_title(f"{symbol} — Rolling Correlation: Sentiment vs Returns")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# ---------------------
# Aggregate visuals
# ---------------------
def plot_granger_heatmap(gc_df: pd.DataFrame, save_path: Path):
    """Heatmaps of Granger p-values (sentiment→returns, returns→sentiment)."""
    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(16, 6),
        layout="constrained"
    )

    # Sentiment → Returns
    g1 = gc_df.pivot_table(index="symbol", values="p_senti_to_ret_min", aggfunc="first")
    sns.heatmap(
        g1, annot=True, fmt=".4f",
        cmap="RdYlGn_r", vmin=0, vmax=0.10,
        cbar_kws={"label": "p-value"},
        ax=ax1
    )
    ax1.set_title("Granger: Sentiment → Returns")
    ax1.set_ylabel("")

    # Returns → Sentiment
    g2 = gc_df.pivot_table(index="symbol", values="p_ret_to_senti_min", aggfunc="first")
    sns.heatmap(
        g2, annot=True, fmt=".4f",
        cmap="RdYlGn_r", vmin=0, vmax=0.10,
        cbar_kws={"label": "p-value"},
        ax=ax2
    )
    ax2.set_title("Granger: Returns → Sentiment")
    ax2.set_ylabel("")

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def create_sentiment_dashboard(gc_df: pd.DataFrame, save_path: Path):
    """Compact dashboard summarising Granger and VAR lags."""
    fig, axes = plt.subplots(
        1, 3,
        figsize=(18, 6),
        layout="constrained"
    )

    # 1) Sentiment → Returns p-values
    s = gc_df.sort_values("p_senti_to_ret_min")
    colors1 = ["green" if p < 0.05 else "gray" for p in s["p_senti_to_ret_min"]]
    axes[0].barh(s["symbol"], s["p_senti_to_ret_min"], color=colors1)
    axes[0].axvline(0.05, color="red", linestyle="--", linewidth=2)
    axes[0].set_xlabel("p-value")
    axes[0].set_title("Sentiment → Returns")
    axes[0].invert_yaxis()
    axes[0].grid(True, alpha=0.3, axis="x")

    # 2) Returns → Sentiment p-values
    r = gc_df.sort_values("p_ret_to_senti_min")
    colors2 = ["green" if p < 0.05 else "gray" for p in r["p_ret_to_senti_min"]]
    axes[1].barh(r["symbol"], r["p_ret_to_senti_min"], color=colors2)
    axes[1].axvline(0.05, color="red", linestyle="--", linewidth=2)
    axes[1].set_xlabel("p-value")
    axes[1].set_title("Returns → Sentiment")
    axes[1].invert_yaxis()
    axes[1].grid(True, alpha=0.3, axis="x")

    # 3) Selected VAR lags
    gl = gc_df.sort_values("selected_lag", ascending=False)
    axes[2].barh(gl["symbol"], gl["selected_lag"], color="purple")
    axes[2].set_xlabel("Optimal Lag (days)")
    axes[2].set_title("VAR Lag Selection")
    axes[2].invert_yaxis()
    axes[2].grid(True, alpha=0.3, axis="x")

    fig.suptitle(
        "Crypto Sentiment — Granger & Lag Summary",
        fontsize=16, fontweight="bold"
    )

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------
# Orchestrator
# ---------------------
def generate_visualizations_from_saved_data():
    """
    Load saved Granger results and per-symbol daily series,
    then render sentiment-focused figures.
    """
    gc_path = OUT_DIR / "granger_summary.csv"
    if not gc_path.exists():
        raise FileNotFoundError(f"Missing {gc_path}, run sentiment_analysis.run_sentiment_analysis() first.")

    gc_df = pd.read_csv(gc_path)

    # Load per-symbol series if available
    symbol_data = {}
    for sym in SYMBOLS:
        p = OUT_DIR / f"{sym}_sentiment_series.csv"
        if p.exists():
            try:
                df = pd.read_csv(p)
                symbol_data[sym] = df
            except Exception:
                pass

    print("\n[VIZ] Generating sentiment visualisations...")

    # Per-symbol
    for sym, df in symbol_data.items():
        try:
            print(f"[VIZ] {sym} ...")
            plot_sentiment_return_scatter(df, sym, VIZ_DIR / f"{sym}_sentiment_return_scatter.png")
            plot_rolling_correlation(df, sym, VIZ_DIR / f"{sym}_rolling_correlation.png")
        except Exception as e:
            print(f"[ERR] {sym}: {e}")

    # Aggregate
    plot_granger_heatmap(gc_df, VIZ_DIR / "granger_heatmap.png")
    create_sentiment_dashboard(gc_df, VIZ_DIR / "sentiment_dashboard.png")

    print(f"\n✅ Sentiment visuals saved to {VIZ_DIR}/")


if __name__ == "__main__":
    generate_visualizations_from_saved_data()