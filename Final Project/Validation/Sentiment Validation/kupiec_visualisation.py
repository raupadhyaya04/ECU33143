"""
kupiec_visualisation.py

Visualisations for VaR / Kupiec validation:
  - Violation rates vs expected by symbol & method
  - Kupiec p-values by symbol & method
  - Overall validation dashboard

Relies on outputs from kupiec_validation.py:
  - kupiec_validation_rolling.csv
"""

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Try to import config from validation module
try:
    from sentiment_kupiec_validation import SYMBOLS, ALPHAS, OUT_DIR
except Exception:
    SYMBOLS = [
        "BTCUSDT","ETHUSDT","BNBUSDT","ADAUSDT","XRPUSDT",
        "SOLUSDT","DOGEUSDT","DOTUSDT","AVAXUSDT","LINKUSDT"
    ]
    ALPHAS = [0.95, 0.99]
    OUT_DIR = Path("./Validation/Sentiment Validation/Results/Analysis")

# Style
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "figure.figsize": (12, 6),
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.titleweight": "bold"
})

VIZ_DIR = Path("./Validation/Sentiment Validation/Results/Visualisations/")
VIZ_DIR.mkdir(parents=True, exist_ok=True)


def plot_var_performance_comparison(kupiec_df: pd.DataFrame, save_path: Path):
    """
    Compare VaR performance across symbols using out-of-sample rolling windows:
    - Violation rate vs expected
    - Kupiec p-values
    """
    fig, axes = plt.subplots(
        2, 2,
        figsize=(16, 10),
        layout="constrained"
    )

    for idx, alpha in enumerate(ALPHAS):
        subset = kupiec_df[kupiec_df["alpha"] == alpha].copy()
        if subset.empty:
            continue

        # Aggregate by symbol & method (mean over windows)
        agg_rate = subset.groupby(["symbol", "method"])["violation_rate"].mean().unstack()
        agg_pval = subset.groupby(["symbol", "method"])["p_value"].mean().unstack()

        # Exception rates
        ax = axes[idx, 0]
        agg_rate.plot(kind="bar", ax=ax, color=["steelblue", "coral"])
        ax.axhline(
            1 - alpha, color="red", linestyle="--", linewidth=2,
            label=f"Expected ({1 - alpha:.2%})"
        )
        ax.set_ylabel("Average Violation Rate")
        ax.set_title(f"VaR Violation Rates (α={int(alpha*100)}%)")
        ax.legend(loc="upper left", fontsize=8, frameon=True)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

        # Kupiec p-values
        ax = axes[idx, 1]
        agg_pval.plot(kind="bar", ax=ax, color=["steelblue", "coral"])
        ax.axhline(
            0.05, color="red", linestyle="--", linewidth=2,
            label="5% Significance"
        )
        ax.set_ylabel("Average Kupiec p-value")
        ax.set_title(f"Kupiec Test (α={int(alpha*100)}%)")
        ax.legend(loc="upper left", fontsize=8, frameon=True)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def create_validation_dashboard(kupiec_df: pd.DataFrame, save_path: Path):
    """Dashboard with key metrics from rolling out-of-sample VaR validation."""
    fig = plt.figure(figsize=(18, 10), layout="constrained")
    gs = fig.add_gridspec(2, 3)

    # 1) Violation rates (95%)
    ax1 = fig.add_subplot(gs[0, 0])
    k95 = kupiec_df[kupiec_df["alpha"] == 0.95]
    if not k95.empty:
        agg95 = k95.groupby(["symbol", "method"])["violation_rate"].mean().unstack()
        agg95.plot(kind="bar", ax=ax1, color=["steelblue", "coral"], width=0.7)
        ax1.axhline(0.05, color="red", linestyle="--", linewidth=2, label="Expected (5%)")
        ax1.set_title("Violation Rates (95%)")
        ax1.set_ylabel("Average Violation Rate")
        ax1.legend(loc="upper left", fontsize=8, frameon=True)
        ax1.grid(True, alpha=0.3, axis="y")
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha="right")
    else:
        ax1.text(0.5, 0.5, "No 95% data", ha="center", va="center")
        ax1.axis("off")

    # 2) Violation rates (99%)
    ax2 = fig.add_subplot(gs[0, 1])
    k99 = kupiec_df[kupiec_df["alpha"] == 0.99]
    if not k99.empty:
        agg99 = k99.groupby(["symbol", "method"])["violation_rate"].mean().unstack()
        agg99.plot(kind="bar", ax=ax2, color=["steelblue", "coral"], width=0.7)
        ax2.axhline(0.01, color="red", linestyle="--", linewidth=2, label="Expected (1%)")
        ax2.set_title("Violation Rates (99%)")
        ax2.set_ylabel("Average Violation Rate")
        ax2.legend(loc="upper left", fontsize=8, frameon=True)
        ax2.grid(True, alpha=0.3, axis="y")
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha="right")
    else:
        ax2.text(0.5, 0.5, "No 99% data", ha="center", va="center")
        ax2.axis("off")

    # 3) Kupiec pass rate by method
    ax3 = fig.add_subplot(gs[0, 2])
    if not kupiec_df.empty:
        pass_rate = kupiec_df.groupby("method")["pass"].mean().sort_index() * 100
        ax3.bar(pass_rate.index, pass_rate.values, color=["steelblue", "coral"])
        ax3.set_ylabel("Pass Rate (%)")
        ax3.set_title("Kupiec Pass Rate (Out-of-Sample)")
        ax3.set_ylim([0, 100])
        ax3.grid(True, alpha=0.3, axis="y")
    else:
        ax3.text(0.5, 0.5, "No Kupiec data", ha="center", va="center")
        ax3.axis("off")

    # 4) Avg violation rate by alpha & method
    ax4 = fig.add_subplot(gs[1, :2])
    if not kupiec_df.empty:
        viol = kupiec_df.groupby(["alpha", "method"])["violation_rate"].mean().reset_index()
        for method, color in zip(["historical", "quantile_reg"], ["steelblue", "coral"]):
            sub = viol[viol["method"] == method]
            if not sub.empty:
                ax4.plot(
                    sub["alpha"] * 100,
                    sub["violation_rate"] * 100,
                    marker="o", color=color, label=method
                )
        ax4.set_xlabel("Confidence Level (%)")
        ax4.set_ylabel("Avg Violation Rate (%)")
        ax4.set_title("Average Violation Rate vs Confidence")
        ax4.grid(True, alpha=0.3)
        ax4.legend(loc="best", fontsize=9, frameon=True)
    else:
        ax4.text(0.5, 0.5, "No data", ha="center", va="center")
        ax4.axis("off")

    # 5) Summary text
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis("off")
    if not kupiec_df.empty:
        n_symbols = kupiec_df["symbol"].nunique()
        total_tests = len(kupiec_df)
        hist_pass = kupiec_df[kupiec_df["method"] == "historical"]["pass"].mean() * 100 if "historical" in kupiec_df["method"].values else 0
        cond_pass = kupiec_df[kupiec_df["method"] == "quantile_reg"]["pass"].mean() * 100 if "quantile_reg" in kupiec_df["method"].values else 0
        avg_95 = kupiec_df[kupiec_df["alpha"] == 0.95]["violation_rate"].mean() if 0.95 in kupiec_df["alpha"].values else float("nan")
        avg_99 = kupiec_df[kupiec_df["alpha"] == 0.99]["violation_rate"].mean() if 0.99 in kupiec_df["alpha"].values else float("nan")

        txt = f"""
VaR VALIDATION SUMMARY
======================

Symbols: {n_symbols}
Total tests: {total_tests:,}

Kupiec pass rate:
  Historical: {hist_pass:.1f}%
  Conditional: {cond_pass:.1f}%

Avg violation rate:
  95%: {avg_95:.2%} (expected 5%)
  99%: {avg_99:.2%} (expected 1%)
"""
    else:
        txt = "No Kupiec validation data available."

    ax5.text(
        0.05, 0.5, txt,
        fontsize=10, family="monospace",
        va="center",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    )

    fig.suptitle(
        "Cryptocurrency VaR — Kupiec Validation Dashboard",
        fontsize=16, fontweight="bold"
    )

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_visualizations_from_saved_data():
    """
    Load saved rolling Kupiec results and render VaR validation figures.
    """
    kupiec_path = OUT_DIR / "kupiec_validation_rolling.csv"
    if not kupiec_path.exists():
        raise FileNotFoundError(f"Missing {kupiec_path}, run kupiec_validation.run_kupiec_analysis() first.")

    kupiec_df = pd.read_csv(kupiec_path)

    print("\n[VIZ] Generating Kupiec / VaR visualisations...")

    plot_var_performance_comparison(kupiec_df, VIZ_DIR / "var_performance_comparison.png")
    create_validation_dashboard(kupiec_df, VIZ_DIR / "validation_dashboard.png")

    print(f"\n✅ Validation visuals saved to {VIZ_DIR}/")


if __name__ == "__main__":
    generate_visualizations_from_saved_data()
