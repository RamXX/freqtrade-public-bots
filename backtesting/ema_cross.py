import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import SignalStrategy, TrailingStrategy
from backtesting.test import GOOG
import talib as ta
from user import user

### Data sources from Freqtrade ###
freqtrade_dir = f'/home/{user}/freqtrade'
exchange = 'kucoin'
exchange_commission = 0.001
base_currency = "USDT"
timeframe="1d"
initial_cash = 50000
# Top-10 crypto assets by market cap
# assets = ["BTC", "ETH", "BNB", "XRP", "ADA", "SOL", "DOGE", "DOT", "MATIC", "SHIB"]
assets = ["ETH"]

### Strategy ###
class EmaCross_trailing(SignalStrategy,
               TrailingStrategy):
    n1 = 8
    n2 = 21
    
    def init(self):
        # In init() and in next() it is important to call the
        # super method to properly initialize the parent classes
        super().init()
        
        # Precompute the two moving averages
        ema1 = self.I(ta.EMA, self.data.Close, self.n1)
        ema2 = self.I(ta.EMA, self.data.Close, self.n2)
        
        # Where ema1 crosses ema2 upwards. Diff gives us [-1,0, *1*]
        signal = (pd.Series(ema1) > ema2).astype(int).diff().fillna(0)
        signal = signal.replace(-1, 0)  # Upwards/long only
        
        # Use 95% of available liquidity (at the time) on each order.
        # (Leaving a value of 1. would instead buy a single share.)
        entry_size = signal * .95
                
        # Set order entry sizes using the method provided by 
        # `SignalStrategy`. See the docs.
        self.set_signal(entry_size=entry_size)
        
        # Set trailing stop-loss to 2x ATR using
        # the method provided by `TrailingStrategy`
        self.set_trailing_sl(2)

### Backtesting each asset separately ###

for i in assets:
    df = pd.read_json(f'{freqtrade_dir}/user_data/data/{exchange}/{i}_{base_currency}-{timeframe}.json')
    df.columns = ['UnixDate', 'Open', 'High', 'Low', 'Close', 'Volume']
    df['Date'] = pd.to_datetime(df['UnixDate'], unit='ms', origin='unix')
    df.set_index(df['Date'], inplace=True)
    df.drop('UnixDate', axis=1, inplace=True)
    
    bt = Backtest(df, EmaCross_trailing, cash=initial_cash, commission=exchange_commission,
                exclusive_orders=True)
    stats = bt.run()
    bt.plot()
    print (stats)