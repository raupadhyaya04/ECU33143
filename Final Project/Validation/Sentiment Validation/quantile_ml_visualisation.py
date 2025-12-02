import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Try to import config/paths from the ML module (optional)
try:
    from sentiment_ml_validation import CRYPTO_SYMBOLS, QUANTILES, RESULTS_DIR  # adjust name if different
except Exception:
    CRYPTO_SYMBOLS = ['BTC', 'ETH', 'BNB', 'SOL', 'ADA']
    QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]
    # Fallback: assume same folder structure as original file
    BASE_DIR = Path(__file__).parent
    RESULTS_DIR = BASE_DIR / "Results"

# Style
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.titleweight": "bold"
})

VIZ_DIR = RESULTS_DIR / "ML_Visualisations"
VIZ_DIR.mkdir(parents=True, exist_ok=True)


# ---------- Core plots ----------

def plot_improvement_heatmap(results_df: pd.DataFrame, save_path: Path):
    """
    Heatmap of Improvement_% by Crypto (rows) and Quantile (columns).
    """
    pivot = results_df.pivot_table(
        index="Crypto",
        columns="Quantile",
        values="Improvement_%",
        aggfunc="mean"
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        pivot,
        annot=True, fmt=".1f",
        cmap="RdYlGn",
        center=0.0,
        cbar_kws={"label": "Improvement in pinball loss (%)"}
    )
    ax.set_title("Sentiment Impact on Quantile Prediction\n(Improvement vs Historical Baseline)")
    ax.set_xlabel("Quantile")
    ax.set_ylabel("Crypto")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_avg_improvement_by_quantile(results_df: pd.DataFrame, save_path: Path):
    """
    Bar plot of average Improvement_% across all cryptos for each quantile.
    """
    avg_by_q = results_df.groupby("Quantile")["Improvement_%"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        avg_by_q["Quantile"].astype(str),
        avg_by_q["Improvement_%"],
        color="steelblue"
    )

    # Highlight tails (5% and 95%) if present
    for i, q in enumerate(avg_by_q["Quantile"]):
        if q in [0.05, 0.95]:
            bars[i].set_color("darkorange")

    ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_ylabel("Average Improvement in Pinball Loss (%)")
    ax.set_xlabel("Quantile")
    ax.set_title("Average Sentiment Contribution by Quantile")

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)

    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_crypto_profiles(results_df: pd.DataFrame, save_path: Path):
    """
    Line plot of Improvement_% across quantiles, one line per crypto.
    Shows whether sentiment helps more in tails or centre per asset.
    """
    # Ensure numeric sorting of quantiles
    df = results_df.copy()
    df["Quantile"] = df["Quantile"].astype(float)

    fig, ax = plt.subplots(figsize=(10, 6))
    for crypto in sorted(df["Crypto"].unique()):
        sub = df[df["Crypto"] == crypto].sort_values("Quantile")
        ax.plot(
            sub["Quantile"],
            sub["Improvement_%"],
            marker="o",
            label=crypto
        )

    ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xticks(sorted(df["Quantile"].unique()))
    ax.set_xticklabels([f"{q:.2f}" for q in sorted(df["Quantile"].unique())])

    ax.set_xlabel("Quantile")
    ax.set_ylabel("Improvement in Pinball Loss (%)")
    ax.set_title("Per-Crypto Improvement Profiles Across Quantiles")
    ax.legend(loc="best", fontsize=8, frameon=True)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def create_ml_summary_dashboard(results_df: pd.DataFrame, save_path: Path):
    """
    Small dashboard summarising:
      - Heatmap-like overview (text)
      - Average improvement by quantile
      - Tail vs centre improvement comparison
    """
    fig = plt.figure(figsize=(14, 8), layout="constrained")
    gs = fig.add_gridspec(2, 2)

    # 1) Average improvement by quantile (bar)
    ax1 = fig.add_subplot(gs[0, 0])
    avg_by_q = results_df.groupby("Quantile")["Improvement_%"].mean().reset_index()
    bars = ax1.bar(
        avg_by_q["Quantile"].astype(str),
        avg_by_q["Improvement_%"],
        color="steelblue"
    )
    for i, q in enumerate(avg_by_q["Quantile"]):
        if q in [0.05, 0.95]:
            bars[i].set_color("darkorange")
    ax1.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    ax1.set_ylabel("Average Improvement (%)")
    ax1.set_xlabel("Quantile")
    ax1.set_title("Avg Sentiment Contribution by Quantile")
    ax1.grid(True, axis="y", alpha=0.3)

    # 2) Boxplot of improvement distribution by quantile
    ax2 = fig.add_subplot(gs[0, 1])
    sns.boxplot(
        data=results_df,
        x="Quantile",
        y="Improvement_%",
        ax=ax2,
        palette="Blues"
    )
    ax2.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    ax2.set_ylabel("Improvement in Pinball Loss (%)")
    ax2.set_title("Distribution of Improvement by Quantile")

    # 3) Per-crypto average improvement (tails vs centre)
    ax3 = fig.add_subplot(gs[1, 0])
    df = results_df.copy()
    df["is_tail"] = df["Quantile"].isin([0.05, 0.95])
    avg_tail = df[df["is_tail"]].groupby("Crypto")["Improvement_%"].mean()
    avg_center = df[~df["is_tail"]].groupby("Crypto")["Improvement_%"].mean()

    idx = sorted(df["Crypto"].unique())
    tail_vals = [avg_tail.get(c, np.nan) for c in idx]
    center_vals = [avg_center.get(c, np.nan) for c in idx]

    x = np.arange(len(idx))
    width = 0.35
    ax3.bar(x - width/2, tail_vals, width, label="Tails (5%,95%)", color="darkorange")
    ax3.bar(x + width/2, center_vals, width, label="Centre", color="steelblue")
    ax3.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    ax3.set_xticks(x)
    ax3.set_xticklabels(idx, rotation=45, ha="right")
    ax3.set_ylabel("Avg Improvement (%)")
    ax3.set_title("Per-Crypto: Tail vs Centre Improvement")
    ax3.legend(loc="best", fontsize=8, frameon=True)
    ax3.grid(True, axis="y", alpha=0.3)

    # 4) Summary text
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")

    tail_improvement = df[df["is_tail"]]["Improvement_%"].mean()
    center_improvement = df[~df["is_tail"]]["Improvement_%"].mean()
    n_cryptos = df["Crypto"].nunique()

    txt = f"""
ML QUANTILE VALIDATION SUMMARY
=============================

Cryptos: {n_cryptos}
Total (crypto, quantile) combos: {len(df)}

Average improvement by quantile:
{avg_by_q.to_string(index=False, header=['Quantile', 'Avg %'])}

Tail vs centre:
  TAIL (5%, 95%):  {tail_improvement:6.1f}% average improvement
  CENTRE:          {center_improvement:6.1f}% average improvement

Interpretation:
  - Positive values → sentiment-based model beats historical baseline.
  - If tails > centre, evidence that sentiment helps more for risk management.
"""

    ax4.text(
        0.02, 0.5, txt,
        fontsize=9, family="monospace",
        va="center", ha="left",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    )

    fig.suptitle(
        "Crypto Sentiment — ML Quantile Validation Dashboard",
        fontsize=16, fontweight="bold"
    )

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------- Orchestrator ----------

def generate_ml_visualisations():
    """
    Load quantile_ml_validation.csv and create ML validation figures.
    """
    csv_path = RESULTS_DIR / "quantile_ml_validation.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing {csv_path}. Run the ML validation script first to generate it."
        )

    results_df = pd.read_csv(csv_path)

    print("\n[VIZ] Generating ML quantile validation visualisations...")

    plot_improvement_heatmap(results_df, VIZ_DIR / "ml_improvement_heatmap.png")
    plot_avg_improvement_by_quantile(results_df, VIZ_DIR / "ml_avg_improvement_by_quantile.png")
    plot_crypto_profiles(results_df, VIZ_DIR / "ml_crypto_profiles.png")
    create_ml_summary_dashboard(results_df, VIZ_DIR / "ml_summary_dashboard.png")

    print(f"\n✅ ML visuals saved to {VIZ_DIR}/")


if __name__ == "__main__":
    generate_ml_visualisations()