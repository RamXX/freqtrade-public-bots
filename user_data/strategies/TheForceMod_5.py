# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement

import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame, Series

from freqtrade.strategy import IStrategy

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class TheForceMod_5(IStrategy):
    """
    Originally based in the ideas of TheForce, particularly using crosses on EMA5 close and open as triggers
    along with Stochastic fast.
    https://github.com/StephaneTurquay/freqtrade-strategies-crypto-trading-bot

    This version adds tweaks the use of the MACD for entries and exits.
    ----
    Mods made by @hextropian (Twitter), a.k.a. as DrWho?#8511 (Discord)
    Use at your own risk - no warranties of success whatsoever.
    """
    INTERFACE_VERSION = 3

    minimal_roi = {
        # These parameters below were generated via 1000X hyperoptimization for a selected portfolio of
        # 48 tokens/coins, fitted from 8/1/22 to 9/12/22
        # 891/1000:    393 trades. 368/0/25 Wins/Draws/Losses. Avg profit   
        # 0.33%. Median profit   
        # 0.01%. Total profit 2390.28582140 USDT (  11.12%). 
        # Avg duration 1 day, 0:35:00 min. Objective: -12.78700
        "0": 0.264,
        "102": 0.068,
        "270": 0.021,
        "536": 0
    }

    stoploss = -0.166

    # Trailing stoploss (hyperopted)
    trailing_stop = True
    trailing_stop_positive = 0.246
    trailing_stop_positive_offset = 0.309
    trailing_only_offset_is_reached = True

    # Optimal timeframe for the strategy.
    timeframe = '15m'

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False

    # These values can be overridden in the "ask_strategy" section in the config.
    use_exit_signal = False
    # exit_profit_only = True is dangerous. You need to keep a close eye in case of a strong
    # downtrend and control your exits manually. Only recommended for testing.
    exit_profit_only = False
    ignore_roi_if_entry_signal = True

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 30

    custom_info = {}
    
    plot_config = {
        # Main plot indicators (Moving averages, ...)
        'main_plot': {
            'ema5c': {'color': 'green'},
            'ema5o': {'color': 'yellow'},
            'mavm': {'color': 'white'},
        },
        'subplots': {
            # Subplots - each dict defines one additional plot
            "MACD": {
                'macd': {'color': 'blue'},
                'macdsignal': {'color': 'orange'},
            }
        }
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) :
        """
        Indicators we need for the selected timeframe (15m)
        """
        
        # Momentum Indicators
        # ------------------------------------

        # Stochastic Fast
        stoch_fast = ta.STOCHF(dataframe,5,3,3)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']

        # MACD
        macd = ta.MACD(dataframe,12,26,1)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        # EMA - Exponential Moving Average for open and close

        dataframe['ema5c'] = ta.EMA(dataframe['close'], timeperiod=5)
        dataframe['ema5o'] = ta.EMA(dataframe['open'], timeperiod=5)

        # MAVW indicator
        dataframe['mavw'] = MAVW(dataframe)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) :
        """
        Entry (buy) rules
        """
        dataframe.loc[
            (
                (dataframe['low'] >= dataframe['mavw'])
                &
                (dataframe['ema5c'] >= dataframe['ema5o'])
                &
                (dataframe['fastk'] >= dataframe['fastd'])
                &
                (
                    (dataframe['fastk'] >= 20) & (dataframe['fastk'] <= 80)
                    &
                    (dataframe['fastd'] >= 20) & (dataframe['fastd'] <= 80)
                )
                &
                (
                    (dataframe['macdhist'] >= dataframe['macdhist'].shift(1))
                    |
                    (
                        (dataframe['macd'] > dataframe['macd'].shift(1))
                        &
                        (dataframe['macdsignal'] > dataframe['macdsignal'].shift(1))
                    )
                )
                &
                (
                    (dataframe['close'] > dataframe['close'].shift(1))
                )
            ),
            'enter_long'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) :
        """
        Exit (sell) rules
        """
        dataframe.loc[
            (
                (
                    (
                        (dataframe['fastk'] < dataframe['fastd'])
                        &
                        (dataframe['fastk'] < 80)
                        &
                        (
                            (dataframe['macd'] < dataframe['macd'].shift(1))
                            &
                            (dataframe['macdsignal'] < dataframe['macdsignal'].shift(1))
                            &
                            (dataframe['macdhist'] < dataframe['macdhist'].shift(1))
                        )
                        &
                        (
                            (dataframe['ema5c'] < dataframe['ema5o'])
                        )
                        &
                        (
                            (dataframe['close'] < dataframe['open'])
                        )
                        &
                        (
                            (dataframe['close'] < dataframe['ema5c'])
                        )
                    )
                )
            ),
            'exit_long'] = 1
        return dataframe

# MavilimW indicator
def MAVW(dataframe, fmal=3, smal=5):
    """
    Python implementation of the MavilimW indicator by KivancOzbilgic
    https://www.tradingview.com/v/IAssyObN/
    """
    tmal = fmal + smal
    Fmal = smal + tmal
    Ftmal = tmal + Fmal
    Smal = Fmal + Ftmal
    M1 = ta.WMA(dataframe['close'], timeperiod=fmal)
    M2 = ta.WMA(M1, timeperiod=smal)
    M3 = ta.WMA(M2, timeperiod=tmal)
    M4 = ta.WMA(M3, timeperiod=Fmal)
    M5 = ta.WMA(M4, timeperiod=Ftmal)
    return Series(ta.WMA(M5, timeperiod=Smal))