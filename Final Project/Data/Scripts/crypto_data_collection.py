from binance import Client
import pandas as pd
import os
from pathlib import Path
from datetime import datetime

# Setup paths
BASE_DIR = Path(__file__).parent.parent.parent  # Go up to project root
CRYPTO_DATA_DIR = BASE_DIR / "Data" / "Crypto Data"
CRYPTO_DATA_DIR.mkdir(parents=True, exist_ok=True)  # Create if doesn't exist

# Binance API credentials
api_key = os.getenv('BINANCE_KEY')
api_secret = os.getenv('BINANCE_SECRET')
client = Client(api_key, api_secret)

# Configuration
interval = Client.KLINE_INTERVAL_1HOUR  # 1m, 5m, 1h, 1d, etc.

symbols = [
    "BTCUSDT",  # Bitcoin (add this if missing)
    "ETHUSDT",  # Ethereum
    "BNBUSDT",  # Binance Coin
    "ADAUSDT",  # Cardano
    "XRPUSDT",  # XRP
    "SOLUSDT",  # Solana
    "DOGEUSDT", # Dogecoin
    "DOTUSDT",  # Polkadot
    "AVAXUSDT", # Avalanche
    "LINKUSDT"  # Chainlink
]

print(f"{'='*60}")
print(f"BINANCE CRYPTOCURRENCY DATA COLLECTION")
print(f"{'='*60}")
print(f"Symbols: {len(symbols)}")
print(f"Interval: {interval}")
print(f"Start date: 1 Jan 2016")
print(f"Output directory: {CRYPTO_DATA_DIR}")
print(f"{'='*60}\n")

start_time = datetime.now()
successful = 0
failed = 0

# Fetch data for each symbol
for symbol in symbols:
    try:
        print(f"📊 Fetching {symbol}...", end=" ")
        
        # Fetch historical klines from Binance
        klines = client.get_historical_klines(symbol, interval, "1 Jan, 2016")
        
        if not klines:
            print(f"⚠️ No data returned")
            failed += 1
            continue
        
        # Convert to DataFrame
        cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'num_trades',
                'taker_base_vol', 'taker_quote_vol', 'ignore']
        df = pd.DataFrame(klines, columns=cols)
        
        # Convert timestamps to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Convert price/volume columns to float
        numeric_cols = ['open', 'high', 'low', 'close', 'volume',
                       'quote_asset_volume', 'taker_base_vol', 'taker_quote_vol']
        df[numeric_cols] = df[numeric_cols].astype(float)
        
        # Save to Crypto Data folder
        output_path = CRYPTO_DATA_DIR / f"{symbol}.csv"
        df.to_csv(output_path, index=False)
        
        # Summary stats
        date_range = f"{df['timestamp'].min().date()} to {df['timestamp'].max().date()}"
        print(f"✅ {len(df)} candles | {date_range}")
        successful += 1
        
    except Exception as e:
        print(f"❌ Error: {e}")
        failed += 1

# Final summary
elapsed = datetime.now() - start_time

print(f"\n{'='*60}")
print(f"COLLECTION COMPLETE")
print(f"{'='*60}")
print(f"✅ Successful: {successful}/{len(symbols)}")
if failed > 0:
    print(f"❌ Failed: {failed}")
print(f"⏱️  Time elapsed: {elapsed}")
print(f"📂 Data saved to: {CRYPTO_DATA_DIR}")
print(f"{'='*60}\n")