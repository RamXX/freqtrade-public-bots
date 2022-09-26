# Converts Freqtrade download data from JSON to CSV with datetime structures. 
# Makes it easier to integrate with backtesting.py or other libraries.

import pandas as pd
from user import user # Users will need to create a user.py file with your UNIX username

freqtrade_dir = f'/home/{user}/freqtrade'
exchange = 'kucoin'
# exchange_commission = 0.001
base_currency = "USDT"
timeframe="1d"
initial_cash = 50000
assets = ["BTC", "ETH", "BNB", "XRP", "ADA", "SOL", "DOGE", "DOT", "MATIC", "SHIB"]

for i in assets:
    df = pd.read_json(f'{freqtrade_dir}/user_data/data/{exchange}/{i}_{base_currency}-{timeframe}.json')
    df.columns = ['UnixDate', 'Open', 'High', 'Low', 'Close', 'Volume']
    df['Date'] = pd.to_datetime(df['UnixDate'], unit='ms', origin='unix')
    df.set_index(df['Date'], inplace=True)
    df.drop('UnixDate', axis=1, inplace=True)
    df.to_csv(f'{i}_{base_currency}-{timeframe}.csv', index=False)