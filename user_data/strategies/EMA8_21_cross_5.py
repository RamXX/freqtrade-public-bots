# EMA8_21_cross_5 - Designed to backtest the 8/21 strategy on the daily timeframe.
#
# --- Required -- do not remove these libs ---
from freqtrade.strategy import IStrategy
from pandas import DataFrame
# --------------------------------

# Imports used by individual strategies.
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class EMA8_21_cross_5(IStrategy):
    """
    EMA8_21_cross_5 - Designed to backtest the 8/21 strategy on the daily.
    In essence, it buys when EMA 8 crosses EMA 21 upwards, and sells in the reverse situation.
    It has ROI and stoploss disabled in order to ONLY use signals for entries and exits.
    This strategy buys only when the 4% threshold has been crossed.
    """
    # Daily timeframe
    timeframe = "1d"

    INTERFACE_VERSION = 3
    
    minimal_roi = {
        "0": 100 # ROI disabled. We'll only use signals.
    }

    # Stoploss:
    stoploss = -0.99 # Stop loss disabled. We'll only use signals.

    # Trailing stop:
    trailing_stop = False # Not used
    trailing_stop_positive = 0.0
    trailing_stop_positive_offset = 0.0
    trailing_only_offset_is_reached = False

    # run "populate_indicators" for all candles
    process_only_new_candles = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 100
    
    # Experimental settings (configuration will overide these if set)
    use_exit_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = True

    # Strategy variables
    fast_length = 8
    slow_length = 21
    limit = 4.0

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:  
        
        fastema = ta.EMA(dataframe['close'], timeperiod=self.fast_length)
        slowema = ta.EMA(dataframe['close'], timeperiod=self.slow_length)
        dataframe['ppo'] = (fastema - slowema) / slowema * 100
       
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (qtpylib.crossed_above(dataframe['ppo'], self.limit)) &
                (dataframe['volume'] > 0)
            ),
            'enter_long'] = 1

        dataframe.loc[
            (
                (qtpylib.crossed_below(dataframe['ppo'], -self.limit)) &
                (dataframe['volume'] > 0)
            ),
            'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (qtpylib.crossed_below(dataframe['ppo'],  self.limit)) &
                (dataframe['volume'] > 0)
            ),
            'exit_long'] = 1

        dataframe.loc[
            (
                (qtpylib.crossed_above(dataframe['ppo'], -self.limit)) &
                (dataframe['volume'] > 0)
            ),
            'exit_long'] = 1
        return dataframe



