import numpy as np
import pandas as pd
from pathlib import Path
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests
import warnings
warnings.filterwarnings('ignore')

# ---------------- CONFIG ----------------
SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT",
    "SOLUSDT", "DOGEUSDT", "DOTUSDT", "AVAXUSDT", "LINKUSDT"
]

PRICE_DIR = Path("./Data/Crypto Data")
SENTI_DIR = Path("./Data/Sentiment Data")
OUT_DIR   = Path("./Crypto-Sentiment/Results/Analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SMOOTH_DAY = 7
MAX_LAG    = 10
# ----------------------------------------


# ---------- DATA LOADING ----------

def load_symbol_df(symbol: str) -> pd.DataFrame:
    """
    Load OHLCV + sentiment and build aligned daily features:
      - ret1d: daily log return
      - reddit_sentiment: raw daily sentiment (filled with 0)
      - sent_7d_ma: 7-day moving average of sentiment
    """
    price_fp = PRICE_DIR / f"{symbol}.csv"
    senti_fp = SENTI_DIR / f"{symbol}_reddit_sentiment.csv"

    if not price_fp.exists() or not senti_fp.exists():
        raise FileNotFoundError(f"Missing files for {symbol}")

    # Price data
    px = pd.read_csv(price_fp)
    px["timestamp"] = pd.to_datetime(px["timestamp"], errors="coerce", utc=True)
    px["date"] = px["timestamp"].dt.tz_convert("UTC").dt.date
    px = px.sort_values("timestamp")
    px["ret1d"] = np.log(px["close"]).diff()

    # Sentiment data
    se = pd.read_csv(senti_fp)
    se["date"] = pd.to_datetime(se["date"]).dt.date
    se = se.sort_values("date")

    # Merge
    df = px.merge(se, on="date", how="left")
    df["reddit_sentiment"] = df["reddit_sentiment"].fillna(0.0)
    df["sent_7d_ma"] = pd.Series(df["reddit_sentiment"]).rolling(SMOOTH_DAY).mean()

    df = df.dropna(subset=["ret1d", "reddit_sentiment", "sent_7d_ma"]).copy()
    df = df.reset_index(drop=True)
    return df


# ---------- GRANGER CAUSALITY ----------

def pick_var_lag(df: pd.DataFrame, max_lag=10) -> int:
    """
    Choose VAR lag by AIC for the bivariate system [ret1d, reddit_sentiment].
    """
    z = df[["ret1d", "reddit_sentiment"]].dropna()
    from statsmodels.tsa.api import VAR
    model = VAR(z)
    sel = model.select_order(maxlags=max_lag)
    return int(sel.aic or 1)


def granger_summary(df: pd.DataFrame, max_lag=10) -> dict:
    """
    Granger tests both directions:
      - sentiment → returns
      - returns   → sentiment

    Stores the minimum p-value across lags for each direction.
    """
    z = df[["ret1d", "reddit_sentiment"]].dropna().copy()
    lag = pick_var_lag(df, max_lag=max_lag)

    # Sentiment → Returns
    res1 = grangercausalitytests(z[["ret1d", "reddit_sentiment"]], maxlag=lag, verbose=False)
    pvals_senti_to_ret = [res1[L][0]["ssr_ftest"][1] for L in range(1, lag+1)]
    p_senti_to_ret = float(np.min(pvals_senti_to_ret))

    # Returns → Sentiment
    res2 = grangercausalitytests(z[["reddit_sentiment", "ret1d"]], maxlag=lag, verbose=False)
    pvals_ret_to_senti = [res2[L][0]["ssr_ftest"][1] for L in range(1, lag+1)]
    p_ret_to_senti = float(np.min(pvals_ret_to_senti))

    return {
        "selected_lag": lag,
        "p_senti_to_ret_min": p_senti_to_ret,
        "p_ret_to_senti_min": p_ret_to_senti
    }


# ---------- MAIN SENTIMENT ANALYSIS RUNNER ----------

def run_sentiment_analysis():
    """
    Run sentiment–returns Granger analysis for all symbols and save results.

    Outputs:
      - granger_summary.csv with:
          symbol, selected_lag, p_senti_to_ret_min, p_ret_to_senti_min
      - optional per-symbol merged series if you want them for plotting later.
    """
    all_gc_rows = []

    print(f"\n{'='*60}")
    print("CRYPTO-SENTIMENT GRANGER ANALYSIS")
    print(f"{'='*60}")
    print(f"Max lag for VAR selection: {MAX_LAG}")
    print(f"{'='*60}\n")

    for sym in SYMBOLS:
        try:
            print(f"📊 Processing {sym}...")
            df = load_symbol_df(sym)
            print(f"  Data: {len(df)} days ({df.iloc[0]['date']} to {df.iloc[-1]['date']})")

            # Granger causality summary
            gc = granger_summary(df, max_lag=MAX_LAG)
            gc_row = {"symbol": sym, **gc}
            all_gc_rows.append(gc_row)

            # Optionally save per-symbol series for visualisation
            out_series = OUT_DIR / f"{sym}_sentiment_series.csv"
            df.to_csv(out_series, index=False)

            print(f"  ✅ Complete\n")

        except Exception as e:
            print(f"  ❌ Error for {sym}: {e}\n")

    # Save Granger results
    gc_df = pd.DataFrame(all_gc_rows)
    gc_df.to_csv(OUT_DIR / "granger_summary.csv", index=False)

    print(f"\n{'='*60}")
    print("GRANGER ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved:")
    print(f"  - {OUT_DIR / 'granger_summary.csv'}")
    print(f"  - {OUT_DIR}/*_sentiment_series.csv")
    print(f"{'='*60}\n")

    return gc_df


if __name__ == "__main__":
    gc_df = run_sentiment_analysis()