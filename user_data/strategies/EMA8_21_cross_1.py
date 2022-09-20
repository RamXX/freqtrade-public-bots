# EMA8_21_cross_1 - Designed to backtest the 8/21 strategy on the weekly timeframe.
#
# --- Required -- do not remove these libs ---
from freqtrade.strategy import IStrategy
from pandas import DataFrame
# --------------------------------

# Imports used by individual strategies.
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class EMA8_21_cross_1(IStrategy):
    """
    EMA8_21_cross_1 - Designed to backtest the 8/21 strategy on the weekly.
    In essence, it buys when EMA 8 crosses EMA 21 upwards, and sells in the reverse situation.
    It has ROI and stoploss disabled in order to ONLY use signals for entries and exits.
    This is a basic, barebone strategy created to assess the effectiveness of the 8/21 cross.
    Other versions in this repo add additional constraints.
    """
    # Weekly timeframe
    timeframe = "1w"

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
    startup_candle_count: int = 42 
    
    # Experimental settings (configuration will overide these if set)
    use_exit_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = True


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:  
        
        dataframe['ema8'] = ta.EMA(dataframe['close'], timeperiod=8)
        dataframe['ema21'] = ta.EMA(dataframe['close'], timeperiod=21)

       
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (qtpylib.crossed_above(dataframe['ema8'], dataframe['ema21'])) &
                (dataframe['volume'] > 0)
            ),
            'enter_long'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (qtpylib.crossed_below(dataframe['ema8'],  dataframe['ema21'])) &
                (dataframe['volume'] > 0)
            ),
            'exit_long'] = 1
        return dataframe



