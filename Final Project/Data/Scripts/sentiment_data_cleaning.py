"""
Reddit Sentiment Data Cleaning and Processing
Applies VADER sentiment analysis and aggregates to daily frequency
"""

import pandas as pd
import os
from pathlib import Path
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime


# =============== CONFIG ===============
SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT",
    "SOLUSDT", "DOGEUSDT", "DOTUSDT", "AVAXUSDT", "LINKUSDT"
]

# Setup paths
BASE_DIR = Path(__file__).parent.parent.parent
RAW_DATA_DIR = BASE_DIR / "Data" / "Raw Data"
SENTIMENT_DATA_DIR = BASE_DIR / "Data" / "Sentiment Data"
SENTIMENT_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Initialize VADER
analyzer = SentimentIntensityAnalyzer()


# =============== CLEANING FUNCTIONS ===============

def get_sentiment(text):
    """
    Calculate VADER sentiment score for text.
    
    Args:
        text: Input text string
    
    Returns:
        Compound sentiment score (-1 to +1)
    """
    if not isinstance(text, str) or not text.strip():
        return 0.0
    return analyzer.polarity_scores(text)["compound"]


def clean_and_process_symbol(symbol):
    """
    Clean raw Reddit data and compute sentiment scores.
    
    Args:
        symbol: Ticker symbol (e.g., "BTCUSDT")
    
    Returns:
        Path to cleaned CSV, or None if failed
    """
    print(f"\n{'='*60}")
    print(f"🧹 Cleaning {symbol}")
    print(f"{'='*60}")
    
    # Load raw data
    raw_path = RAW_DATA_DIR / f"{symbol}_reddit_raw.csv"
    if not raw_path.exists():
        print(f"⚠️ Raw data not found: {raw_path}")
        return None
    
    df = pd.read_csv(raw_path)
    print(f"  Loaded {len(df)} raw posts")
    
    # Combine title and body for sentiment analysis
    df["combined_text"] = df["title"].fillna("") + " " + df["body"].fillna("")
    
    # Remove empty posts
    df = df[df["combined_text"].str.strip() != ""]
    print(f"  After removing empty: {len(df)} posts")
    
    # Compute sentiment scores
    print(f"  Computing sentiment scores...")
    df["sentiment"] = df["combined_text"].apply(get_sentiment)
    
    # Convert dates
    df["date"] = pd.to_datetime(df["date"])
    
    # Aggregate to daily level
    print(f"  Aggregating to daily frequency...")
    daily_sentiment = df.groupby("date").agg({
        "sentiment": "mean",
        "submission_id": "count",  # Number of posts per day
        "score": "sum",  # Total upvotes
        "num_comments": "sum"  # Total comments
    }).reset_index()
    
    daily_sentiment.rename(columns={
        "sentiment": "reddit_sentiment",
        "submission_id": "post_count",
        "score": "total_upvotes",
        "num_comments": "total_comments"
    }, inplace=True)
    
    # Save cleaned data
    output_path = SENTIMENT_DATA_DIR / f"{symbol}_reddit_sentiment.csv"
    daily_sentiment.to_csv(output_path, index=False)
    
    print(f"✅ Saved daily sentiment data: {output_path}")
    print(f"   Date range: {daily_sentiment['date'].min()} to {daily_sentiment['date'].max()}")
    print(f"   Days with data: {len(daily_sentiment)}")
    print(f"   Avg posts/day: {daily_sentiment['post_count'].mean():.1f}")
    print(f"   Avg sentiment: {daily_sentiment['reddit_sentiment'].mean():.3f}")
    
    return output_path


def merge_with_price_data(symbol):
    """
    Merge sentiment data with cryptocurrency price data.
    
    Args:
        symbol: Ticker symbol (e.g., "BTCUSDT")
    
    Returns:
        Path to merged CSV, or None if failed
    """
    print(f"\n  💰 Merging with price data for {symbol}...")
    
    # Load sentiment data
    sentiment_path = SENTIMENT_DATA_DIR / f"{symbol}_reddit_sentiment.csv"
    if not sentiment_path.exists():
        print(f"  ⚠️ Sentiment data not found: {sentiment_path}")
        return None
    
    sentiment_df = pd.read_csv(sentiment_path)
    sentiment_df["date"] = pd.to_datetime(sentiment_df["date"]).dt.date
    
    # Load price data
    price_path = BASE_DIR / "Data" / "Crypto Data" / f"{symbol}.csv"
    if not price_path.exists():
        print(f"  ⚠️ Price data not found: {price_path}")
        return None
    
    price_df = pd.read_csv(price_path)
    price_df["timestamp"] = pd.to_datetime(price_df["timestamp"], unit="ms")
    price_df["date"] = price_df["timestamp"].dt.date
    
    # Merge on date
    merged = pd.merge(price_df, sentiment_df, on="date", how="left")
    
    # Fill missing sentiment with 0
    merged["reddit_sentiment"].fillna(0, inplace=True)
    merged["post_count"].fillna(0, inplace=True)
    
    # Save merged data
    output_path = SENTIMENT_DATA_DIR / f"{symbol}_with_sentiment.csv"
    merged.to_csv(output_path, index=False)
    
    print(f"  ✅ Merged data saved: {output_path}")
    
    return output_path


# =============== MAIN CLEANING RUNNER ===============

def clean_all_symbols():
    """Clean and process Reddit data for all cryptocurrency symbols."""
    print(f"\n{'='*60}")
    print(f"REDDIT SENTIMENT DATA CLEANING")
    print(f"{'='*60}")
    print(f"Symbols: {len(SYMBOLS)}")
    print(f"Input directory: {RAW_DATA_DIR}")
    print(f"Output directory: {SENTIMENT_DATA_DIR}")
    print(f"{'='*60}\n")
    
    start_time = datetime.now()
    
    results = []
    for symbol in SYMBOLS:
        # Clean and compute sentiment
        cleaned_path = clean_and_process_symbol(symbol)
        results.append(cleaned_path)
        
        # Merge with price data
        if cleaned_path:
            merge_with_price_data(symbol)
    
    # Summary
    successful = [r for r in results if r is not None]
    failed = len(results) - len(successful)
    
    elapsed = datetime.now() - start_time
    
    print(f"\n{'='*60}")
    print(f"CLEANING COMPLETE")
    print(f"{'='*60}")
    print(f"✅ Successful: {len(successful)}/{len(SYMBOLS)}")
    if failed > 0:
        print(f"❌ Failed: {failed}")
    print(f"⏱️  Time elapsed: {elapsed}")
    print(f"📂 Data saved to: {SENTIMENT_DATA_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    clean_all_symbols()