from binance import Client, ThreadedWebsocketManager, ThreadedDepthCacheManager
import pandas as pd
import os

api_key = os.getenv('BINANCE_KEY')
api_secret = os.getenv('BINANCE_SECRET')
client = Client(api_key, api_secret)

interval = Client.KLINE_INTERVAL_1HOUR  # 1m, 5m, 1h, 1d, etc.

symbols = [
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


for symbol in symbols:
    print(f"Fetching data for {symbol}...")
    # Fetch from 1 Jan 2016 until now
    klines = client.get_historical_klines(symbol, interval, "1 Jan, 2016")

    # Convert to DataFrame
    cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'num_trades',
            'taker_base_vol', 'taker_quote_vol', 'ignore']
    df = pd.DataFrame(klines, columns=cols)

    # Convert timestamps to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df.astype(float)

    df.to_csv(f'{symbol}.csv')
    print(df.head())
    print(f"Data saved to {symbol}.csv")
