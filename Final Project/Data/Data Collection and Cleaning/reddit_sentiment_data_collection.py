import praw
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# =============== CONFIG ===============
symbols = {
    "BTCUSDT": "bitcoin OR btc",
    "ETHUSDT": "ethereum OR eth",
    "BNBUSDT": "binance OR bnb",
    "ADAUSDT": "cardano OR ada",
    "XRPUSDT": "xrp OR ripple",
    "SOLUSDT": "solana OR sol",
    "DOGEUSDT": "dogecoin OR doge",
    "DOTUSDT": "polkadot OR dot",
    "AVAXUSDT": "avalanche OR avax",
    "LINKUSDT": "chainlink OR link"
}

SUBREDDITS = [
    # General
    "CryptoCurrency", "CryptoMarkets", "CryptoMoonShots", "CryptoCurrencyTrading",
    "CryptoTechnology", "Crypto_General", "cryptotrading", "cryptoinvesting",
    # Specific Assets
    "Bitcoin", "Ethereum", "Binance", "cardano", "Ripple", "solana", "dogecoin",
    "polkadot", "chainlink", "avalancheavax", "litecoin", "tronix", "stellar",
    # Analysis / News
    "CryptoAnalysis", "CryptoNews", "defi", "defistarter", "CryptoQuant",
    # Meta / Speculative
    "SatoshiStreetBets", "AltcoinDiscussion", "cryptomoon", "CryptoCurrencyClassic"
]

DATA_PATH = "./Data/Sentiment Data"
os.makedirs(DATA_PATH, exist_ok=True)

# =============== REDDIT AUTH ===============
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent="crypto-sentiment-script"
)
# ==========================================

analyzer = SentimentIntensityAnalyzer()

# --- Sentiment helper ---
def get_sentiment(text):
    if not isinstance(text, str):
        return 0
    return analyzer.polarity_scores(text)["compound"]

# --- Reddit fetcher (PRAW version) ---
def fetch_reddit_data(keyword, subreddits, limit=1000):
    """Fetch Reddit submissions (title + body) for given keyword from multiple subreddits."""
    results = []
    for sub in subreddits:
        try:
            for submission in reddit.subreddit(sub).search(keyword, sort="new", time_filter="all", limit=limit):
                results.append({
                    "created_utc": int(submission.created_utc),
                    "body": (submission.title or "") + " " + (submission.selftext or ""),
                    "subreddit": sub
                })
            time.sleep(0.3)  # gentle rate limit
        except Exception as e:
            print(f"⚠️ Error fetching {keyword} in r/{sub}: {e}")
    return results

# --- Core symbol processing ---
def process_symbol(symbol, keyword):
    print(f"\n🚀 Processing {symbol} ({keyword})")
    src_path = f"./Data/Crypto Data/{symbol}.csv"
    if not os.path.exists(src_path):
        print(f"⚠️ Skipping {symbol} — {src_path} not found.")
        return

    all_data = fetch_reddit_data(keyword, SUBREDDITS, limit=1000)
    if not all_data:
        print(f"⚠️ No Reddit data collected for {symbol}")
        return

    reddit_df = pd.DataFrame(all_data)
    reddit_df["sentiment"] = reddit_df["body"].apply(get_sentiment)
    reddit_df["timestamp"] = pd.to_datetime(reddit_df["created_utc"], unit="s")
    reddit_df["date"] = reddit_df["timestamp"].dt.date

    daily_sent = reddit_df.groupby("date")["sentiment"].mean().reset_index()
    daily_sent.rename(columns={"sentiment": "reddit_sentiment"}, inplace=True)

    reddit_out = os.path.join(DATA_PATH, f"{symbol}_reddit_sentiment.csv")
    merged_out = os.path.join(DATA_PATH, f"{symbol}_with_sentiment.csv")

    daily_sent.to_csv(reddit_out, index=False)

    df = pd.read_csv(src_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["date"] = df["timestamp"].dt.date
    merged = pd.merge(df, daily_sent, on="date", how="left")
    merged["reddit_sentiment"].fillna(0, inplace=True)
    merged.to_csv(merged_out, index=False)

    print(f"✅ Done: {symbol} → saved to {merged_out}")

# ========== RUN IN PARALLEL ==========
with ThreadPoolExecutor(max_workers=9) as executor:
    futures = [executor.submit(process_symbol, s, k) for s, k in symbols.items()]
    for future in as_completed(futures):
        _ = future.result()

print("\n🎉 All symbols processed successfully using Reddit API (PRAW)!")
