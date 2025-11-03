import os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy.stats import chi2
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests

# ---------------- CONFIG ----------------
SYMBOLS = [
    "BTCUSDT","ETHUSDT","BNBUSDT","ADAUSDT","XRPUSDT",
    "SOLUSDT","DOGEUSDT","DOTUSDT","AVAXUSDT","LINKUSDT"
]

PRICE_DIR = Path("./Data/Crypto Data")                 # your OHLCV
SENTI_DIR = Path("../Data/Sentiment Data")             # your *_reddit_sentiment.csv
OUT_DIR   = SENTI_DIR / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ROLL_WIN   = 250      # trading days (~1y) for VaR
ALPHAS     = [0.95, 0.99]  # VaR confidence levels
MAX_LAG    = 10       # for Granger/VARR lag selection
SMOOTH_DAY = 7        # rolling mean for sentiment
# ----------------------------------------


# ---------- Utilities ----------
def load_symbol_df(symbol: str) -> pd.DataFrame:
    """Load OHLCV + sentiment and build aligned daily features."""
    price_fp = PRICE_DIR / f"{symbol}.csv"
    senti_fp = SENTI_DIR / f"{symbol}_reddit_sentiment.csv"

    if not price_fp.exists() or not senti_fp.exists():
        raise FileNotFoundError(f"Missing files for {symbol}: {price_fp} or {senti_fp}")

    px = pd.read_csv(price_fp)
    px["timestamp"] = pd.to_datetime(px["timestamp"], errors="coerce", utc=True)
    px["date"] = px["timestamp"].dt.tz_convert("UTC").dt.date
    px = px.sort_values("timestamp")

    # close-to-close log returns (stationary)
    px["ret1d"] = np.log(px["close"]).diff()

    se = pd.read_csv(senti_fp)
    # expect columns: date, reddit_sentiment
    # accept either string date or datetime-like
    se["date"] = pd.to_datetime(se["date"]).dt.date
    se = se.sort_values("date")

    df = px.merge(se, on="date", how="left")
    # fill sentiment gaps with 0 (or try forward-fill if you prefer continuity)
    df["reddit_sentiment"] = df["reddit_sentiment"].fillna(0.0)
    df["sent_7d_ma"] = (
        pd.Series(df["reddit_sentiment"].values, index=pd.to_datetime(df["date"]))
        .rolling(f"{SMOOTH_DAY}D").mean().values
    )
    # Alternatively: rolling by rows (assumes daily sampling)
    df["sent_7d_ma"] = pd.Series(df["reddit_sentiment"]).rolling(SMOOTH_DAY).mean()

    # drop first rows with NaNs from returns/smoothing
    df = df.dropna(subset=["ret1d", "reddit_sentiment", "sent_7d_ma"]).copy()
    return df


def rolling_historical_var(returns: pd.Series, alpha=0.99, window=250) -> pd.Series:
    """
    Historical VaR (left tail). Returns VaR as a positive number (loss threshold).
    Using log returns; VaR compares to negative tail.
    """
    # quantile of returns distribution at (1 - alpha)
    q = returns.rolling(window).quantile(1 - alpha)
    # Convert to a positive loss threshold: VaR = -q
    return -q


def kupiec_pof_test(returns: pd.Series, var_series: pd.Series, alpha=0.99) -> dict:
    """
    Kupiec Proportion-of-Failures test:
    H0: observed exception rate == expected (1 - alpha).
    """
    aligned = pd.concat([returns, var_series], axis=1).dropna()
    r = aligned.iloc[:, 0]
    v = aligned.iloc[:, 1]

    # exceptions when loss > VaR threshold
    # with log returns, a "loss" is -ret1d; exception if -ret > VaR  <=> ret < -VaR
    exceptions = (r < -v).astype(int)
    n = len(exceptions)
    x = exceptions.sum()
    # likelihood ratio
    pi = x / n if n > 0 else 0.0
    pi0 = 1 - alpha
    # guard for degenerate cases
    if x == 0 or x == n:
        lr = 0.0
    else:
        lr = -2 * ( ( (n-x)*np.log(1-pi0) + x*np.log(pi0) ) - ( (n-x)*np.log(1-pi) + x*np.log(pi) ) )
    pval = 1 - chi2.cdf(lr, df=1)
    return {"n": n, "exceptions": int(x), "rate": (x/n if n else np.nan), "lr": float(lr), "p_value": float(pval)}


def conditional_var_quantreg(df: pd.DataFrame, alpha=0.99, features=("reddit_sentiment","sent_7d_ma")) -> pd.Series:
    """
    Conditional VaR via quantile regression: q_alpha(ret | X) = X * beta
    Returns VaR as positive threshold (loss), so VaR = -q_alpha.
    """
    y = df["ret1d"].values
    X = sm.add_constant(df[list(features)].values)
    model = sm.QuantReg(y, X)
    res = model.fit(q=1 - alpha)
    q_hat = res.predict(X)
    # VaR as positive loss threshold:
    cvar = -pd.Series(q_hat, index=df.index)
    return cvar


def pick_var_lag(df: pd.DataFrame, max_lag=10) -> int:
    """
    Choose VAR lag by AIC on a bivariate system [ret1d, reddit_sentiment].
    """
    # Build the matrix with aligned columns
    z = df[["ret1d", "reddit_sentiment"]].dropna()
    # statsmodels VAR is in tsa.api
    from statsmodels.tsa.api import VAR
    model = VAR(z)
    sel = model.select_order(maxlags=max_lag)
    # fall back if None:
    lag = sel.aic
    return int(lag or 1)


def granger_summary(df: pd.DataFrame, max_lag=10) -> dict:
    """
    Granger tests both directions with lag chosen by AIC, also report p-values across lags.
    """
    # stationarity: returns are fine; sentiment often near-stationary; if needed, difference:
    z = df[["ret1d", "reddit_sentiment"]].dropna().copy()

    # Try AIC lag:
    lag = pick_var_lag(df, max_lag=max_lag)

    # granger test: does senti -> returns?
    # grangercausalitytests expects array with [tested_series, causing_series]
    # so for "sentiment causes returns": y=ret1d, x=sentiment
    res1 = grangercausalitytests(z[["ret1d", "reddit_sentiment"]], maxlag=lag, verbose=False)
    # collect smallest p-value across lags (SSRF F-test)
    pvals_senti_to_ret = [res1[L][0]["ssr_ftest"][1] for L in range(1, lag+1)]
    p_senti_to_ret = float(np.min(pvals_senti_to_ret))

    # returns -> sentiment
    res2 = grangercausalitytests(z[["reddit_sentiment", "ret1d"]], maxlag=lag, verbose=False)
    pvals_ret_to_senti = [res2[L][0]["ssr_ftest"][1] for L in range(1, lag+1)]
    p_ret_to_senti = float(np.min(pvals_ret_to_senti))

    return {
        "selected_lag": lag,
        "p_senti_to_ret_min": p_senti_to_ret,
        "p_ret_to_senti_min": p_ret_to_senti
    }


# -------------- Main runner --------------
all_var_rows = []
all_gc_rows  = []

for sym in SYMBOLS:
    try:
        df = load_symbol_df(sym)

        # ----- Historical VaR + backtest -----
        for a in ALPHAS:
            var_hist = rolling_historical_var(df["ret1d"], alpha=a, window=ROLL_WIN)
            kt_hist = kupiec_pof_test(df["ret1d"], var_hist, alpha=a)
            all_var_rows.append({
                "symbol": sym, "alpha": a, "method": "historical",
                **kt_hist
            })

        # ----- Conditional VaR (Quantile Regression on sentiment) + backtest -----
        for a in ALPHAS:
            var_qr = conditional_var_quantreg(df, alpha=a, features=("reddit_sentiment","sent_7d_ma"))
            kt_qr = kupiec_pof_test(df["ret1d"], var_qr, alpha=a)
            all_var_rows.append({
                "symbol": sym, "alpha": a, "method": "quantile_reg",
                **kt_qr
            })

        # Save per-symbol daily VaR series (optional)
        var_out = OUT_DIR / f"{sym}_var_series.csv"
        var_frame = pd.DataFrame({
            "date": df["date"].values,
            "ret1d": df["ret1d"].values
        })
        for a in ALPHAS:
            var_frame[f"VaR_hist_{int(a*100)}"] = rolling_historical_var(df["ret1d"], alpha=a, window=ROLL_WIN).values
            var_frame[f"VaR_qr_{int(a*100)}"]   = conditional_var_quantreg(df, alpha=a, features=("reddit_sentiment","sent_7d_ma")).values
        var_frame.to_csv(var_out, index=False)

        # ----- Granger causality -----
        gc = granger_summary(df, max_lag=MAX_LAG)
        all_gc_rows.append({
            "symbol": sym,
            **gc
        })

        print(f"[OK] {sym}: VaR + Granger done.")

    except Exception as e:
        print(f"[ERR] {sym}: {e}")

# ---- Save summaries ----
pd.DataFrame(all_var_rows).to_csv(OUT_DIR / "var_backtests_summary.csv", index=False)
pd.DataFrame(all_gc_rows ).to_csv(OUT_DIR / "granger_summary.csv", index=False)

print(f"\nSaved:\n- {OUT_DIR/'var_backtests_summary.csv'}\n- {OUT_DIR/'granger_summary.csv'}")