# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement

from pandas import DataFrame

from freqtrade.strategy import IStrategy
from typing import Optional
from datetime import datetime
from freqtrade.persistence import Trade
from datetime import timedelta

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import logging

log = logging.getLogger(__name__)

# This strategy leverages the high volatility in the lower timeframes for the CIX100 token
# ---
# Designed and written by @hextropian (Twitter), a.k.a. as DrWho?#8511 (Discord)
# Use at your own risk - no warranties of success whatsoever.

# Hyperopted parameters:
# freqtrade hyperopt -c ./config-single.json --strategy BB_CIX --hyperopt-loss SharpeHyperOptLoss --timerange=20220101- -e 1000 --spaces stoploss roi trailing
# 243/1000:    274 trades. 264/0/10 Wins/Draws/Losses. 
# Avg profit   1.15%. 
# Median profit   1.51%. 
# Total profit 16268.95796529 USDT (  32.54%). 
# Avg duration 13:35:00 min. 
# Objective: -4.31843

class BB_CIX(IStrategy):
  
    INTERFACE_VERSION = 3

    minimal_roi = {
        "0": 0.085,
        "104": 0.032,
        "282": 0.015,
        "539": 0
    }

    # stoploss = -0.99 # Effectively disabled
    stoploss = -0.212 # Hyperopted
    use_custom_stoploss = False 

    # Hyperopted numbers
    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.14
    trailing_stop_positive_offset = 0.158
    trailing_only_offset_is_reached = True

    # Optimal timeframe for the strategy.
    timeframe = '15m'

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the "ask_strategy" section in the config.
    use_exit_signal = False
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 60


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
        # dataframe['bb_middleband'] = bollinger['mid'] # (Currently unused)
        dataframe['bb_upperband'] = bollinger['upper']

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
                    (dataframe['close'] <= dataframe['bb_lowerband'])
                    & (dataframe['volume'] > 0)
                )
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
                 (dataframe['close'] > dataframe['bb_upperband'])
               & (dataframe['volume'] > 0)
            ),
            'exit_long'] = 1

        return dataframe


# Helper function 
def to_minutes(**timdelta_kwargs):
    return int(timedelta(**timdelta_kwargs).total_seconds() / 60)

