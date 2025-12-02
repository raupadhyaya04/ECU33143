"""
kupiec_validation.py

Out-of-sample Kupiec proportion-of-failures tests for:
  - Historical VaR (returns-only)
  - Conditional VaR via quantile regression (with sentiment)

Outputs:
  - CSV: ./Crypto-Sentiment/Results/Analysis/kupiec_validation_rolling.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import chi2
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# ---------------- CONFIG ----------------
SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT",
    "SOLUSDT", "DOGEUSDT", "DOTUSDT", "AVAXUSDT", "LINKUSDT"
]

PRICE_DIR = Path("./Data/Crypto Data")
SENTI_DIR = Path("./Data/Sentiment Data")
OUT_DIR   = Path("./Validation/Sentiment Validation/Results/Analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# VaR params
CALIB_WINDOW = 250   # Calibration window
TEST_WINDOW  = 250   # Test window size
STEP_SIZE    = 125   # Step between test windows (50% overlap)
ALPHAS       = [0.95, 0.99]
SMOOTH_DAY   = 7
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


# ---------- VAR CALCULATION (OUT-OF-SAMPLE) ----------

def calculate_historical_var(calib_returns: pd.Series, alpha: float) -> float:
    """Calculate historical VaR from calibration period (loss threshold as positive number)."""
    q = calib_returns.quantile(1 - alpha)
    return -float(q)


def calculate_conditional_var(
    calib_df: pd.DataFrame,
    test_df: pd.DataFrame,
    alpha: float
) -> np.ndarray:
    """
    Train quantile regression on calibration period,
    predict conditional VaR on test period using sentiment features.
    """
    # Train on calibration
    y_train = calib_df["ret1d"].values
    X_train = sm.add_constant(calib_df[["reddit_sentiment", "sent_7d_ma"]].values)

    model = sm.QuantReg(y_train, X_train)
    res = model.fit(q=1 - alpha)

    # Predict on test
    X_test = sm.add_constant(test_df[["reddit_sentiment", "sent_7d_ma"]].values)
    q_hat = res.predict(X_test)

    return -q_hat  # VaR as positive number


# ---------- KUPIEC TEST ----------

def kupiec_test(returns: np.ndarray, var_values: np.ndarray, alpha: float) -> dict:
    """
    Single Kupiec proportion-of-failures test on one test window.

    returns: array of realized returns
    var_values: array of VaR thresholds (positive, loss)
    alpha: VaR confidence level (e.g. 0.99)
    """
    exceptions = (returns < -var_values).sum()
    n = len(returns)
    rate = exceptions / n if n > 0 else 0.0
    pi0 = 1 - alpha

    if exceptions == 0 or exceptions == n:
        lr = 0.0
    else:
        pi = rate
        lr = -2 * (
            ((n - exceptions) * np.log(1 - pi0) + exceptions * np.log(pi0))
            - ((n - exceptions) * np.log(1 - pi) + exceptions * np.log(pi))
        )

    pval = 1 - chi2.cdf(lr, df=1)

    return {
        "n_days": int(n),
        "exceptions": int(exceptions),
        "violation_rate": float(rate),
        "expected_rate": float(1 - alpha),
        "lr_stat": float(lr),
        "p_value": float(pval),
        "pass": bool(pval > 0.05)
    }


# ---------- ROLLING OUT-OF-SAMPLE KUPIEC ----------

def rolling_kupiec_validation(df: pd.DataFrame, symbol: str, alpha: float) -> list:
    """
    Perform multiple out-of-sample Kupiec tests.

    For each window:
      1. Calibrate VaR on past CALIB_WINDOW days
      2. Test on next TEST_WINDOW days
      3. Step forward by STEP_SIZE days, repeat
    """
    results = []

    min_size = CALIB_WINDOW + TEST_WINDOW
    if len(df) < min_size:
        print(f"  ⚠️ Insufficient data for {symbol} (need {min_size}, have {len(df)})")
        return results

    start_idx = CALIB_WINDOW

    while start_idx + TEST_WINDOW <= len(df):
        # Calibration period
        calib_start = start_idx - CALIB_WINDOW
        calib_end = start_idx
        calib_df = df.iloc[calib_start:calib_end]

        # Test period
        test_start = start_idx
        test_end = start_idx + TEST_WINDOW
        test_df = df.iloc[test_start:test_end]

        # Historical VaR
        var_hist = calculate_historical_var(calib_df["ret1d"], alpha)
        var_hist_array = np.full(len(test_df), var_hist)
        kt_hist = kupiec_test(test_df["ret1d"].values, var_hist_array, alpha)

        results.append({
            "symbol": symbol,
            "alpha": alpha,
            "method": "historical",
            "window_start": test_df.iloc[0]["date"],
            "window_end": test_df.iloc[-1]["date"],
            **kt_hist
        })

        # Conditional VaR (quantile regression)
        try:
            var_cond = calculate_conditional_var(calib_df, test_df, alpha)
            kt_cond = kupiec_test(test_df["ret1d"].values, var_cond, alpha)

            results.append({
                "symbol": symbol,
                "alpha": alpha,
                "method": "quantile_reg",
                "window_start": test_df.iloc[0]["date"],
                "window_end": test_df.iloc[-1]["date"],
                **kt_cond
            })
        except Exception as e:
            print(f"  ⚠️ Quantile reg failed for {symbol} window {test_start}: {e}")

        # Step forward
        start_idx += STEP_SIZE

    return results


# ---------- MAIN KUPIEC RUNNER ----------

def run_kupiec_analysis():
    """
    Run Kupiec backtests for all symbols and save results.
    Returns: kupiec_df (DataFrame of all tests).
    """
    all_kupiec_rows = []

    print(f"\n{'='*60}")
    print("CRYPTO-SENTIMENT KUPIEC VALIDATION")
    print(f"{'='*60}")
    print(f"Calibration window: {CALIB_WINDOW} days")
    print(f"Test window:       {TEST_WINDOW} days")
    print(f"Step size:         {STEP_SIZE} days")
    print(f"Alphas:            {ALPHAS}")
    print(f"{'='*60}\n")

    for sym in SYMBOLS:
        try:
            print(f"📊 Processing {sym}...")
            df = load_symbol_df(sym)
            print(f"  Data: {len(df)} days ({df.iloc[0]['date']} to {df.iloc[-1]['date']})")

            # Rolling Kupiec tests
            for alpha in ALPHAS:
                kupiec_results = rolling_kupiec_validation(df, sym, alpha)
                all_kupiec_rows.extend(kupiec_results)

                if kupiec_results:
                    n_hist = len([r for r in kupiec_results if r["method"] == "historical"])
                    pass_hist = sum(r["pass"] for r in kupiec_results if r["method"] == "historical")
                    n_cond = len([r for r in kupiec_results if r["method"] == "quantile_reg"])
                    pass_cond = sum(r["pass"] for r in kupiec_results if r["method"] == "quantile_reg")

                    print(f"  α={alpha}: {n_hist} test windows")
                    print(f"    Historical: {pass_hist}/{n_hist} pass ({pass_hist / n_hist * 100:.0f}%)")
                    if n_cond > 0:
                        print(f"    Conditional: {pass_cond}/{n_cond} pass ({pass_cond / n_cond * 100:.0f}%)")

            print(f"  ✅ Complete\n")

        except Exception as e:
            print(f"  ❌ Error for {sym}: {e}\n")

    kupiec_df = pd.DataFrame(all_kupiec_rows)
    kupiec_df.to_csv(OUT_DIR / "kupiec_validation_rolling.csv", index=False)

    # Summary statistics
    print(f"\n{'='*60}")
    print("KUPIEC VALIDATION SUMMARY")
    print(f"{'='*60}")

    for method in ["historical", "quantile_reg"]:
        method_df = kupiec_df[kupiec_df["method"] == method]
        if len(method_df) > 0:
            pass_rate = method_df["pass"].mean()
            avg_viol = method_df["violation_rate"].mean()
            print(f"\n{method.upper()}:")
            print(f"  Total tests: {len(method_df)}")
            print(f"  Pass rate:   {pass_rate * 100:.1f}%")
            print(f"  Avg violation rate: {avg_viol * 100:.2f}%")

            for alpha in ALPHAS:
                alpha_df = method_df[method_df["alpha"] == alpha]
                if len(alpha_df) > 0:
                    pass_rate_a = alpha_df["pass"].mean()
                    print(f"    α={alpha}: {pass_rate_a * 100:.1f}% pass rate")

    print(f"\n{'='*60}")
    print("✅ KUPIEC ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved:")
    print(f"  - {OUT_DIR / 'kupiec_validation_rolling.csv'}")
    print(f"{'='*60}\n")

    return kupiec_df


if __name__ == "__main__":
    kupiec_df = run_kupiec_analysis()