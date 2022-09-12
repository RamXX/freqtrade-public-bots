# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement

import pandas_ta as pta
import numpy as np

from pandas import DataFrame

from freqtrade.strategy import IStrategy, merge_informative_pair, CategoricalParameter, IntParameter, DecimalParameter
from typing import Optional
from datetime import datetime
from freqtrade.persistence import Trade
from functools import reduce
from datetime import timedelta

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import logging

log = logging.getLogger(__name__)

# Originally based in the ideas of TheForce, particularly using crosses on EMA5 close and open as triggers
# along with Stochastic fast.
# https://github.com/StephaneTurquay/freqtrade-strategies-crypto-trading-bot
# 
# ----
# Mods made by @hextropian (Twitter), a.k.a. as DrWho?#8511 (Discord)
# Use at your own risk - no warranties of success whatsoever.
class TheForce3(IStrategy):
  
    INTERFACE_VERSION = 3

    minimal_roi = {
        "0": 100 # Effectively disabled
        # Hyperopted on top10 pairs
        # "0": 0.276,
        # "36": 0.077,
        # "90": 0.031,
        # "145": 0
    }

    custom_info = {
        # You definitely want to adjust this to better reflect your risk and portfolio
        'risk_reward_ratio': 1.5,
        'sl_multiplier': 3.5,
        'set_to_break_even_at_profit': 1.01,
    }

    stoploss = -0.99 # Effectively disabled
    # Hyperopted on top10 pairs
    # stoploss = -0.217
    use_custom_stoploss = True # We use a custom function for fixed risk/reward.

    # Hyperopted numbers
    # Trailing stop:
    trailing_stop = False
    trailing_stop_positive = 0.341
    trailing_stop_positive_offset = 0.359
    trailing_only_offset_is_reached = False

    # Optimal timeframe for the strategy.
    timeframe = '5m'
    inf_timeframe='4h'

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the "ask_strategy" section in the config.
    use_exit_signal = True
    # exit_profit_only = True is dangerous. You need to keep a close eye in case of a strong
    # downtrend and control your exits manually. Only recommended for testing.
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 200

    # Protection parameters from Apollo11 strategy
    @property
    def protections(self):
        return [
            {
                # Don't enter a trade right after selling a trade.
                "method": "CooldownPeriod",
                "stop_duration": to_minutes(minutes=5),
            },
            {
                # Stop trading if max-drawdown is reached.
                "method": "MaxDrawdown",
                "lookback_period": to_minutes(hours=12),
                "trade_limit": 20,  # Considering all pairs that have a minimum of 20 trades
                "stop_duration": to_minutes(hours=1),
                "max_allowed_drawdown": 0.2,  # If max-drawdown is > 20% this will activate
            },
            {
                # Stop trading if a certain amount of stoploss occurred within a certain time window.
                "method": "StoplossGuard",
                "lookback_period": to_minutes(hours=6),
                "trade_limit": 4,  # Considering all pairs that have a minimum of 4 trades
                "stop_duration": to_minutes(minutes=30),
                "only_per_pair": False,  # Looks at all pairs
            },
            {
                # Lock pairs with low profits
                "method": "LowProfitPairs",
                "lookback_period": to_minutes(hours=1, minutes=30),
                "trade_limit": 2,  # Considering all pairs that have a minimum of 2 trades
                "stop_duration": to_minutes(hours=2),
                "required_profit": 0.02,  # If profit < 2% this will activate for a pair
            },
            {
                # Lock pairs with low profits
                "method": "LowProfitPairs",
                "lookback_period": to_minutes(hours=6),
                "trade_limit": 4,  # Considering all pairs that have a minimum of 4 trades
                "stop_duration": to_minutes(minutes=30),
                "required_profit": 0.01,  # If profit < 1% this will activate for a pair
            },
        ]


    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        """
            custom_stoploss using a fixed risk/reward ratio.
            Based on: https://github.com/freqtrade/freqtrade-strategies/blob/main/user_data/strategies/FixedRiskRewardLoss.py
        """
        result = break_even_sl = takeprofit_sl = -1
        custom_info_pair = self.custom_info.get(pair)
        if custom_info_pair is not None:
            # using current_time/open_date directly via custom_info_pair[trade.open_daten]
            # would only work in backtesting/hyperopt.
            # in live/dry-run, we have to search for nearest row before it
            open_date_mask = custom_info_pair.index.unique().get_loc(trade.open_date_utc, method='ffill')
            open_df = custom_info_pair.iloc[open_date_mask]

            # trade might be open too long for us to find opening candle
            if(len(open_df) != 1):
                return -1 # won't update current stoploss

            initial_sl_abs = open_df['stoploss_rate']

            # calculate initial stoploss at open_date
            initial_sl = initial_sl_abs/current_rate-1

            # calculate take profit treshold
            # by using the initial risk and multiplying it
            risk_distance = trade.open_rate-initial_sl_abs
            reward_distance = risk_distance*self.custom_info['risk_reward_ratio']
            # take_profit tries to lock in profit once price gets over
            # risk/reward ratio treshold
            take_profit_price_abs = trade.open_rate+reward_distance
            # take_profit gets triggerd at this profit
            take_profit_pct = take_profit_price_abs/trade.open_rate-1

            # break_even tries to set sl at open_rate+fees (0 loss)
            break_even_profit_distance = risk_distance*self.custom_info['set_to_break_even_at_profit']
            # break_even gets triggerd at this profit
            break_even_profit_pct = (break_even_profit_distance+current_rate)/current_rate-1

            result = initial_sl
            if(current_profit >= break_even_profit_pct):
                break_even_sl = (trade.open_rate*(1+trade.fee_open+trade.fee_close) / current_rate)-1
                result = break_even_sl

            if(current_profit >= take_profit_pct):
                takeprofit_sl = take_profit_price_abs/current_rate-1
                result = takeprofit_sl

        return result

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

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs_4h = [(pair, self.inf_timeframe) for pair in pairs]
        # Explicitly adding BTC and ETH in case they're not already in the whitelist.
        informative_pairs_4h += [("ETH/USDT", self.inf_timeframe),
                                  ("BTC/USDT", self.inf_timeframe),
                                 ]
        return informative_pairs_4h

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame: 
        # assert self.dp, 'DataProvider is required for multiple timeframes.'
        # Get the informative pairs
        informative_4h = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.inf_timeframe)

        # Get informative BTC and ETH
        informative_BTC = self.dp.get_pair_dataframe(pair="BTC/USDT", timeframe=self.inf_timeframe)
        informative_ETH = self.dp.get_pair_dataframe(pair="ETH/USDT", timeframe=self.inf_timeframe)

        btc_t3 = ta.T3(informative_BTC, timeperiod=11, vfactor=0.7)
        eth_t3 = ta.T3(informative_ETH, timeperiod=11, vfactor=0.7)
        current_t3 = ta.T3(informative_4h, timeperiod=11, vfactor=0.7)

        # Trend directions for BTC, ETH and the current pair, used as a reference.
        dataframe['macro_ok'] = (
              # Candles are green and above T3
              # Hoghly correlated assets will likely follow the trend. 
              (informative_BTC['close'] >= informative_BTC['open'])
            & (informative_BTC['close'] >= btc_t3)
            & (informative_ETH['close'] >= informative_ETH['open'])
            & (informative_ETH['close'] >= eth_t3)
            & (informative_4h['close'] >= informative_4h['open'])
            & (informative_4h['close'] >= current_t3)
        )

        # Decreasing size candles and trading below T3.
        dataframe['macro_downtrend'] = (
              (informative_BTC['close'] < btc_t3) 
            & (informative_BTC['close'] < informative_BTC['close'].shift(1))
            & (informative_ETH['close'] < eth_t3) 
            & (informative_ETH['close'] < informative_ETH['close'].shift(1))
            & (informative_4h['close'] < current_t3)
            & (informative_4h['close'] < informative_4h['close'].shift(1))
        )

        # Stochastic Fast
        stoch_fast = ta.STOCHF(dataframe,5,3,3)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']

        # MACD
        macd = ta.MACD(dataframe,12,26,1)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        # EMA 5 for close and open
        dataframe['ema5c'] = ta.EMA(dataframe['close'], timeperiod=5)
        dataframe['ema5o'] = ta.EMA(dataframe['open'], timeperiod=5)

        # ADX - great to validate actual trends
        dataframe['adx'] = ta.ADX(dataframe)

        # t3
        dataframe['t3'] = ta.T3(dataframe, timeperiod=11, vfactor=0.7)

        # Stop loss
        dataframe['stoploss_rate'] = dataframe['close'] - (ta.ATR(dataframe, timeperiod=32) * self.custom_info['sl_multiplier'])
        
        self.custom_info[metadata['pair']] = dataframe[['date', 'stoploss_rate']].copy().set_index('date')

        return dataframe


    #################################################################################
    ##                                                                             ## 
    ##                            BUY (Enter) conditions                           ##
    ##                                                                             ## 
    ################################################################################# 
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dont_buy_conditions = []

        dont_buy_conditions.append(
            (
                # don't buy if there isn't 2% profit to be made
                (dataframe['close'].rolling(48).max() < (dataframe['close'] * 1.008))
            )
        )
        dont_buy_conditions.append(
            (
                # don't buy if we're in a macro downtrend
                (dataframe['macro_downtrend'])
            )
        )

        dataframe.loc[
            (
                 (
                    (qtpylib.crossed_above(dataframe['ema5c'],  dataframe['ema5o'])) 
                    & (dataframe['macro_ok'])
                    & (dataframe['fastk'] >= 20) & (dataframe['fastk'] <= 80)
                    & (dataframe['fastd'] >= 20) & (dataframe['fastd'] <= 80)
                    & (dataframe['fastk'] >= dataframe['fastd'])
                    & (dataframe['macd'] > dataframe['macd'].shift(1))
                    & (dataframe['macdsignal'] > dataframe['macdsignal'].shift(1))
                    & (dataframe['close'] > dataframe['close'].shift(1))
                    & (dataframe['close'] >= dataframe['t3'])
                    & (dataframe['volume'] > 0)
                )
            ),
            'enter_long'] = 1

        if dont_buy_conditions:
            for condition in dont_buy_conditions:
                dataframe.loc[condition, 'enter_long'] = 0

        return dataframe
    
    #################################################################################
    ##                                                                             ## 
    ##                            SELL (Exit) conditions                           ##
    ##                                                                             ## 
    ################################################################################# 
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        dataframe.loc[
            (
                 (
                      (qtpylib.crossed_below(dataframe['ema5c'],  dataframe['ema5o'])) 
                    & (dataframe['fastk'] <= 80)
                    & (dataframe['fastd'] <= 80)
                    & (dataframe['fastk'] < dataframe['fastd'])
                    & (dataframe['macd'] < dataframe['macd'].shift(1))
                    & (dataframe['macdsignal'] < dataframe['macdsignal'].shift(1))
                    & (dataframe['adx'] >= 18)
                    & (dataframe['volume'] > 0)
                ) 
                |
                (   # Sudden strong downturn in BTC and ETH and we're also trading below T3
                    (dataframe['macro_downtrend'])
                   & (dataframe['close'] < dataframe['t3'])
                )
            ),
            'exit_long'] = 1

        return dataframe

# Helper function 
def to_minutes(**timdelta_kwargs):
    return int(timedelta(**timdelta_kwargs).total_seconds() / 60)
