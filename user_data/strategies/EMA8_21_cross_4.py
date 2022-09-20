# EMA8_21_cross_4 - Designed to backtest the 8/21 strategy on the daily timeframe.
# Additional constraints on top of the base strategy.
#
# --- Required -- do not remove these libs ---
from freqtrade.strategy import IStrategy
from pandas import DataFrame
# --------------------------------

# Imports used by individual strategies.
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class EMA8_21_cross_4(IStrategy):
    """
    EMA8_21_cross_4 - Designed to backtest the 8/21 strategy on the daily.
    In essence, it buys when EMA 8 crosses EMA 21 upwards, and sells in the reverse situation.
    It has ROI and stoploss disabled in order to ONLY use signals for entries and exits.

    This version triggers a buy only if the EMA 8 is over the 21 AND if the current and previous
    candles close above the EMA 8.
    For selling, we also need the current and previous candle close below the EMA 21.
    This version adds an additiona constraing: buys only happen over the 200 SMA. Sales happen if the close gets 
    below that line also.
    """
    # Weekly timeframe
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
    startup_candle_count: int = 400

    # Experimental settings (configuration will overide these if set)
    use_exit_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = True


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:  
        
        dataframe['ema8'] = ta.EMA(dataframe['close'], timeperiod=8)
        dataframe['ema21'] = ta.EMA(dataframe['close'], timeperiod=21)
        dataframe['sma200'] = ta.SMA(dataframe['close'], timeperiod=200)


       
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['close'] >= dataframe['sma200']) &
                (dataframe['ema8'] >= dataframe['ema21']) &
                (dataframe['close'] >= dataframe['ema8']) &
                (dataframe['close'].shift(1) >= dataframe['ema8'].shift(1)) &
                (dataframe['volume'] > 0)
            ),
            'enter_long'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (
                    (dataframe['ema8'] < dataframe['ema21']) &
                    (dataframe['close'] < dataframe['ema21']) &
                    (dataframe['close'].shift(1) < dataframe['ema21'].shift(1))
                )
                |
                (
                    (qtpylib.crossed_below(dataframe['close'], dataframe['sma200']))
                ) &
                (dataframe['volume'] > 0)
            ),
            'exit_long'] = 1
        return dataframe



