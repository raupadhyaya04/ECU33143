"""
Reddit Sentiment Data Collection
Fetches Reddit posts/comments for cryptocurrency keywords and saves raw data
"""

import praw
import pandas as pd
import time
import os
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


# =============== CONFIG ===============
SYMBOLS = {
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

# Setup paths
BASE_DIR = Path(__file__).parent.parent.parent
RAW_DATA_DIR = BASE_DIR / "Data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

SEARCH_LIMIT = 1000  # Posts per subreddit
RATE_LIMIT_DELAY = 0.3  # seconds between requests


# =============== REDDIT AUTH ===============
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent="crypto-sentiment-collection-script/1.0"
)
print(f"✅ Authenticated as: {reddit.user.me() if reddit.read_only else 'Read-only mode'}")


# =============== COLLECTION FUNCTIONS ===============

def fetch_reddit_data(keyword, subreddits, limit=1000):
    """
    Fetch Reddit submissions for given keyword from multiple subreddits.
    
    Args:
        keyword: Search query (e.g., "bitcoin OR btc")
        subreddits: List of subreddit names to search
        limit: Max submissions per subreddit
    
    Returns:
        List of dicts with raw Reddit data
    """
    results = []
    
    for sub in subreddits:
        try:
            print(f"  → Searching r/{sub}...", end=" ")
            count = 0
            
            for submission in reddit.subreddit(sub).search(
                keyword, 
                sort="new", 
                time_filter="all", 
                limit=limit
            ):
                results.append({
                    "created_utc": int(submission.created_utc),
                    "title": submission.title or "",
                    "body": submission.selftext or "",
                    "score": submission.score,
                    "num_comments": submission.num_comments,
                    "subreddit": sub,
                    "submission_id": submission.id,
                    "author": str(submission.author) if submission.author else "[deleted]"
                })
                count += 1
            
            print(f"{count} posts")
            time.sleep(RATE_LIMIT_DELAY)  # Rate limiting
            
        except Exception as e:
            print(f"\n⚠️ Error fetching {keyword} in r/{sub}: {e}")
    
    return results


def collect_symbol_data(symbol, keyword):
    """
    Collect raw Reddit data for a single cryptocurrency symbol.
    
    Args:
        symbol: Ticker symbol (e.g., "BTCUSDT")
        keyword: Search query string
    
    Returns:
        Path to saved CSV file, or None if failed
    """
    print(f"\n{'='*60}")
    print(f"🚀 Collecting data for {symbol}")
    print(f"   Keyword: {keyword}")
    print(f"{'='*60}")
    
    # Fetch data from Reddit
    all_data = fetch_reddit_data(keyword, SUBREDDITS, limit=SEARCH_LIMIT)
    
    if not all_data:
        print(f"⚠️ No Reddit data collected for {symbol}")
        return None
    
    # Convert to DataFrame
    reddit_df = pd.DataFrame(all_data)
    
    # Add timestamp conversion
    reddit_df["timestamp"] = pd.to_datetime(reddit_df["created_utc"], unit="s")
    reddit_df["date"] = reddit_df["timestamp"].dt.date
    
    # Sort by date
    reddit_df = reddit_df.sort_values("timestamp")
    
    # Save raw data
    output_path = RAW_DATA_DIR / f"{symbol}_reddit_raw.csv"
    reddit_df.to_csv(output_path, index=False)
    
    print(f"✅ Saved {len(reddit_df)} posts to {output_path}")
    print(f"   Date range: {reddit_df['date'].min()} to {reddit_df['date'].max()}")
    
    return output_path


# =============== MAIN COLLECTION RUNNER ===============

def collect_all_symbols(parallel=True, max_workers=5):
    """
    Collect Reddit data for all cryptocurrency symbols.
    
    Args:
        parallel: Whether to use parallel processing
        max_workers: Number of parallel workers (if parallel=True)
    """
    print(f"\n{'='*60}")
    print(f"REDDIT SENTIMENT DATA COLLECTION")
    print(f"{'='*60}")
    print(f"Symbols: {len(SYMBOLS)}")
    print(f"Subreddits: {len(SUBREDDITS)}")
    print(f"Output directory: {RAW_DATA_DIR}")
    print(f"Mode: {'Parallel' if parallel else 'Sequential'}")
    print(f"{'='*60}\n")
    
    start_time = datetime.now()
    
    if parallel:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(collect_symbol_data, symbol, keyword)
                for symbol, keyword in SYMBOLS.items()
            ]
            
            results = []
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
    else:
        results = []
        for symbol, keyword in SYMBOLS.items():
            result = collect_symbol_data(symbol, keyword)
            results.append(result)
    
    # Summary
    successful = [r for r in results if r is not None]
    failed = len(results) - len(successful)
    
    elapsed = datetime.now() - start_time
    
    print(f"\n{'='*60}")
    print(f"COLLECTION COMPLETE")
    print(f"{'='*60}")
    print(f"✅ Successful: {len(successful)}/{len(SYMBOLS)}")
    if failed > 0:
        print(f"❌ Failed: {failed}")
    print(f"⏱️  Time elapsed: {elapsed}")
    print(f"📂 Data saved to: {RAW_DATA_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    collect_all_symbols(parallel=True, max_workers=5)