# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement

from pandas import DataFrame

from freqtrade.strategy import IStrategy
from typing import Optional
from datetime import datetime
from freqtrade.persistence import Trade
from datetime import timedelta
from freqtrade.strategy import stoploss_from_open, DecimalParameter, IntParameter


import talib.abstract as ta
import numpy as np
import pandas_ta as pta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import logging
import math

log = logging.getLogger(__name__)

# This strategy leverages the high volatility in the lower timeframes for tokens with high volatility
# ---
# Designed and written by @hextropian (Twitter), a.k.a. as DrWho?#8511 (Discord)
# Use at your own risk - no warranties of success whatsoever.

# Hyperopted parameters for 5m

class VolatilityCatcher(IStrategy):
  
    INTERFACE_VERSION = 3

    minimal_roi = {
        "0": 0.063,
        "34": 0.029,
        "83": 0.016,
        "182": 0
    }

    stoploss = -0.312
    use_custom_stoploss = False 

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.061
    trailing_stop_positive_offset = 0.082
    trailing_only_offset_is_reached = False

    # Optimal timeframe for the strategy.
    timeframe = '5m'

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the "ask_strategy" section in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = True

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 60

    rsi_threshold = IntParameter(low=3, high=60, default=50, space='buy', optimize=True, load=True)
    rsi_overbought = IntParameter(low=50, high=98, default=75, space='sell', optimize=True, load=True)
    reversal_threshold = DecimalParameter(0.01, 0.5, default=0.1, decimals=2, space='buy', optimize=True, load=True)

    # trailing stoploss hyperopt parameters
    opt_HSL = False # Optimize?
    # hard stoploss profit
    pHSL = DecimalParameter(-0.200, -0.040, default=-0.08, decimals=3, space='sell', optimize=opt_HSL, load=True)
    # # profit threshold 1, trigger point, SL_1 is used
    pPF_1 = DecimalParameter(0.008, 0.020, default=0.016, decimals=3, space='sell', optimize=opt_HSL, load=True)
    pSL_1 = DecimalParameter(0.008, 0.020, default=0.011, decimals=3, space='sell', optimize=opt_HSL, load=True)

    # profit threshold 2, SL_2 is used
    pPF_2 = DecimalParameter(0.040, 0.100, default=0.080, decimals=3, space='sell', optimize=opt_HSL, load=True)
    pSL_2 = DecimalParameter(0.020, 0.070, default=0.040, decimals=3, space='sell', optimize=opt_HSL, load=True)

    buy_params = {
        "rsi_threshold": 50,
        "reversal_threshold": 0.45
    }

    sell_params = {
        "rsi_overbought": 96,
        "pHSL": -0.141,
        "pPF_1": 0.019,
        "pPF_2": 0.079,
        "pSL_1": 0.017,
        "pSL_2": 0.069,
    }


    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:

        # hard stoploss profit
        HSL = self.pHSL.value
        PF_1 = self.pPF_1.value
        SL_1 = self.pSL_1.value
        PF_2 = self.pPF_2.value
        SL_2 = self.pSL_2.value

        # For profits between PF_1 and PF_2 the stoploss (sl_profit) used is linearly interpolated
        # between the values of SL_1 and SL_2. For all profits above PL_2 the sl_profit value 
        # rises linearly with current profit, for profits below PF_1 the hard stoploss profit is used.

        if (current_profit > PF_2):
            sl_profit = SL_2 + (current_profit - PF_2)
        elif (current_profit > PF_1):
            sl_profit = SL_1 + ((current_profit - PF_1)*(SL_2 - SL_1)/(PF_2 - PF_1))
        else:
            sl_profit = HSL
        
        return stoploss_from_open(sl_profit, current_profit)


    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime, entry_tag: Optional[str],
                            side: str, **kwargs) -> bool:
        """
        From NostalgiaForInfinityX by iterativ 
        https://github.com/iterativv/NostalgiaForInfinity
        allow force entries and protects against slippage.
        """

        if (entry_tag == 'force_entry'):
            return True

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        if(len(dataframe) < 1):
            return True

        dataframe = dataframe.iloc[-1].squeeze()

        if ((rate > dataframe['close'])):
            slippage = ((rate / dataframe['close']) - 1.0)

            if slippage < 0.044:
                return True
            else:
                log.warning(
                    "Cancelling buy for %s due to slippage %s",
                    pair, slippage
                )
                return False

        return True


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        # Bollinger bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid'] 
        dataframe['bb_upperband'] = bollinger['upper']

        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['volma'] = ta.SMA(dataframe['volume'], timeperiod=60)

        dataframe['ema5c'] = ta.EMA(dataframe['close'], timeperiod=5)
        dataframe['ema4o'] = ta.EMA(dataframe['open'], timeperiod=4)

        return dataframe


    #################################################################################
    ##                                                                             ## 
    ##                            BUY (Enter) conditions                           ##
    ##                                                                             ## 
    ################################################################################# 
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    
        dataframe.loc[
            (
                (
                    (dataframe['close'] > dataframe['open']) & # Current candle is green
                    (dataframe['close'] < dataframe['bb_middleband']) # and closes below the middle BB
                )
                &
                (
                    (dataframe['close'].shift(1) < dataframe['open'].shift(1)) # Previous candle is red
                   &(dataframe['close'].shift(1) <= dataframe['bb_lowerband'].shift(1)) # and closes below the BB
                )      
                &
                (   # Volume is above its MA for any of the last 3 candles
                      (dataframe['volume'] >= dataframe['volma'])
                    | (dataframe['volume'].shift(1) >= dataframe['volma'].shift(1))
                    | (dataframe['volume'].shift(2) >= dataframe['volma'].shift(2))
                )
                &
                (   # Candle marks a reversal on [virtually] the same place
                    (dataframe['open'] - dataframe['close'].shift(1)).abs() <= self.reversal_threshold.value
                )
                &
                (   # Oversold RSI on previous candle
                    (dataframe['rsi'].shift(1) <= self.rsi_threshold.value)
                )
                & (dataframe['volume'] > 0)
            ),
            'enter_long'] = 1

        return dataframe
    
    #################################################################################
    ##                                                                             ## 
    ##                            SELL (Exit) conditions                           ##
    ##                                                                             ## 
    ################################################################################# 
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                   (dataframe['close'].shift(1) > dataframe['open'].shift(1)) & # Previous candle is green
                   (dataframe['close'].shift(1) >= dataframe['bb_upperband']) & # Above its upperband
                   (dataframe['rsi'].shift(1) >= self.rsi_overbought.value) &
                   # Now we look at the current candle:
                   (dataframe['ema5c'] <= dataframe['ema4o']) & # Lost momentum
                   (dataframe['volume'] > 0)
            ),
            'exit_long'] = 1

        return dataframe


# Helper function 
def to_minutes(**timdelta_kwargs):
    return int(timedelta(**timdelta_kwargs).total_seconds() / 60)

