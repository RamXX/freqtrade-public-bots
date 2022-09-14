# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement

import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame, Series

from freqtrade.strategy import IStrategy

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class TheForceMod_1(IStrategy):
    """
    Originally based in the ideas of TheForce, particularly using crosses on EMA5 close and open as triggers
    along with Stochastic fast.
    https://github.com/StephaneTurquay/freqtrade-strategies-crypto-trading-bot

    This version adds additional constraints for more precise entries and exits, checks BTC and ETH status,
    and implements a circuit breaker that sells the entire portfolio if a major drop happens to BTC and ETH
    due to the high correlation with other coins.
    ----
    Mods made by @hextropian (Twitter), a.k.a. as DrWho?#8511 (Discord)
    Use at your own risk - no warranties of success whatsoever.
    """
    INTERFACE_VERSION = 3

    minimal_roi = {
        # These parameters below were generated via 1000X hyperoptimization for a selected portfolio of
        # 48 tokens/coins, fitted from 8/1/22 to 9/6/22
        "0": 0.35,
        "92": 0.062,
        "270": 0.015,
        "329": 0
    }

    stoploss = -0.030

    # Trailing stoploss (hyperopted)
    trailing_stop = True
    trailing_stop_positive = 0.111
    trailing_stop_positive_offset = 0.205
    trailing_only_offset_is_reached = False

    # Optimal timeframe for the strategy.
    timeframe = '15m'

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False

    # These values can be overridden in the "ask_strategy" section in the config.
    use_exit_signal = True
    # exit_profit_only = True is dangerous. You need to keep a close eye in case of a strong
    # downtrend and control your exits manually. Only recommended for testing.
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 30

    custom_info = {
        "sell_all_threshold": 0.038, # 3.8% decline in BTC and ETH in x number of candles.
        "no_candles": 2 # Half an hour.
    }
    
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

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.timeframe) for pair in pairs]
        # Explicitly adding BTC and ETH in case they're not already in the whitelist.
        informative_pairs += [("ETH/USDT", self.timeframe),
                                  ("BTC/USDT", self.timeframe),
                                 ]
        return informative_pairs

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

        # MAVM - MavilimW indicator (see at the end)
        dataframe['mavm'] = MAVW(dataframe)

        # Keltner Channel
        kc = qtpylib.keltner_channel(dataframe, window=20, atrs=2)
        dataframe['kc_upper'] = kc['upper']
        dataframe['kc_lower'] = kc['lower']
        dataframe['kc_mid']   = kc['mid']

        # Get informative entries BTC and ETH
        informative_BTC = self.dp.get_pair_dataframe(pair="BTC/USDT", timeframe=self.timeframe)
        informative_ETH = self.dp.get_pair_dataframe(pair="ETH/USDT", timeframe=self.timeframe)

        t_btc = 1-(informative_BTC['close']/informative_BTC['close'].shift(self.custom_info['no_candles']))
        t_eth = 1-(informative_ETH['close']/informative_ETH['close'].shift(self.custom_info['no_candles']))

        kc_btc = qtpylib.keltner_channel(informative_BTC, window=20, atrs=2)
        kc_eth = qtpylib.keltner_channel(informative_ETH, window=20, atrs=2)
        mavm_btc = MAVW(informative_BTC)
        mavm_eth = MAVW(informative_ETH)


        dataframe['trade_ok'] = (
            (informative_BTC['close'] >= kc_btc['mid']) & # trading above the middle band of its Keltner Channel
            (informative_BTC['close'] >= mavm_btc) & # trading above its MAVM line
            (informative_BTC['close'] >= informative_BTC['open']) & # green current candle
            (informative_BTC['close'].shift(1) >= informative_BTC['open'].shift(1)) & # green previous candle
            (informative_ETH['close'] >= kc_eth['mid']) & # trading above the middle band of its Keltner Channel
            (informative_ETH['close'] >= mavm_eth) & # trading above its MAVM line
            (informative_ETH['close'] >= informative_ETH['open']) & # green current candle
            (informative_ETH['close'].shift(1) >= informative_ETH['open'].shift(1)) # green previous candle
        )


        # Sell everything if we have a sudden large drop in BTC and ETH
        dataframe['emergency_sell'] = (
            (t_btc >= self.custom_info['sell_all_threshold']) & 
            (t_eth >= self.custom_info['sell_all_threshold'])
        )

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) :
        """
        Entry (buy) rules
        """
        dataframe.loc[
            (
                (
                    (dataframe['trade_ok'])
                )
                &
                (
                    (dataframe['low'] >= dataframe['mavm'])
                )
                &
                (
                    (dataframe['fastk'] >= 20) & (dataframe['fastk'] <= 80)
                    &
                    (dataframe['fastd'] >= 20) & (dataframe['fastd'] <= 80)
                )
                &
                (
                    (dataframe['macd'] > dataframe['macd'].shift(1))
                    &
                    (dataframe['macdsignal'] > dataframe['macdsignal'].shift(1))
                )
                &
                (
                    (dataframe['close'] > dataframe['close'].shift(1))
                )
                &
                (
                    (dataframe['ema5c'] >= dataframe['ema5o'])
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
                        (dataframe['high'] < dataframe['mavm'])
                    )
                    |
                    (
                        (
                            dataframe['close'] < dataframe['kc_mid']
                        )
                        &
                        (
                            (dataframe['fastk'] <= 80)
                            &
                            (dataframe['fastd'] <= 80)
                        )
                        &
                        (
                            (dataframe['macd'] < dataframe['macd'].shift(1))
                            &
                            (dataframe['macdsignal'] < dataframe['macdsignal'].shift(1))
                        )
                        &
                        (
                            (dataframe['ema5c'] < dataframe['ema5o'])
                        )
                    )
                )
                |
                (
                    (dataframe['emergency_sell'])
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