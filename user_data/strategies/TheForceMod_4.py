# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement

from pandas import DataFrame

from freqtrade.strategy import IStrategy, merge_informative_pair, CategoricalParameter
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
# This version adds additional constraints, such as following uptrends for BTC and ETH in 
# 15m timeframes, as well as using stochastic crosses for K and D in addition to ranges.
# It also implements a fixed risk-reward model. Currently performing best at 1:1.31 ratio,
# but users may want to tweak based on asset portfolio.
# Mods made by @hextropian (Twitter), a.k.a. as DrWho?#8511 (Discord)
# Use at your own risk - no warranties of success whatsoever.
class TheForceMod_4(IStrategy):
  
    INTERFACE_VERSION = 3

    minimal_roi = {
        #"0": 100 # Effectively disabled
         "0": 0.147,
        "34": 0.094,
        "90": 0.028,
        "205": 0
    }

    custom_info = {
        'risk_reward_ratio': 1.31,
        'sl_multiplier': 4.0,
        'set_to_break_even_at_profit': 1.01,
    }

    # stoploss = -0.99 # Effectively disabled
    stoploss = -0.345
    use_custom_stoploss = True # We use a custom function for fixed risk/reward.

    # Hyperopted numbers
    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.018
    trailing_stop_positive_offset = 0.055
    trailing_only_offset_is_reached = True

    # Optimal timeframe for the strategy.
    timeframe = '5m'

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False

    # These values can be overridden in the "ask_strategy" section in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 60


    # Hyperoptable parameters
    # Enter conditions
    enter_condition_1_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=True, load=True)
    enter_condition_2_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=True, load=True)
    enter_condition_3_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=True, load=True)
    enter_condition_4_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=True, load=True)
    enter_condition_5_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=True, load=True)
    enter_condition_6_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=True, load=True)
    
    # Exit conditions
    exit_condition_1_enable = CategoricalParameter([True, False], default=True, space='sell', optimize=True, load=True)
    exit_condition_2_enable = CategoricalParameter([True, False], default=True, space='sell', optimize=True, load=True)
    exit_condition_3_enable = CategoricalParameter([True, False], default=True, space='sell', optimize=True, load=True)
    exit_condition_4_enable = CategoricalParameter([True, False], default=True, space='sell', optimize=True, load=True)
   
    # Buy hyperspace params:
    buy_params = {
        "enter_condition_1_enable": True,
        "enter_condition_2_enable": True,
        "enter_condition_3_enable": True,
        "enter_condition_4_enable": True,
        "enter_condition_5_enable": True,
        "enter_condition_6_enable": True,
        "enter_condition_7_enable": True,
        "enter_condition_8_enable": True
    }
    # Sell hyperspace params:
    sell_params = {
        "exit_condition_1_enable": True,
        "exit_condition_2_enable": True,
        "exit_condition_3_enable": True,
        "exit_condition_4_enable": True
    }

    # Protection parameters from Apollo11 strategy
    @property
    def protections(self):
        return [
            {
                # Don't enter a trade right after selling a trade.
                "method": "CooldownPeriod",
                "stop_duration": to_minutes(minutes=0),
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
                "stop_duration": to_minutes(hours=15),
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

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs_15m = [(pair, '15m') for pair in pairs]
        # Explicitly adding BTC and ETH in case they're not already in the whitelist.
        informative_pairs_15m += [("ETH/USDT", "15m"),
                                  ("BTC/USDT", "15m"),
                                 ]
        return informative_pairs_15m

    def informative_15m_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        assert self.dp, 'DataProvider is required for multiple timeframes.'
        # Get the informative pairs
        informative_15m = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe='15m')

        return informative_15m


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

    # From NostalgiaForInfinityX by iterativ 
    # https://github.com/iterativv/NostalgiaForInfinity
    # allow force entries and protects against slippage.
    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime, entry_tag: Optional[str],
                            side: str, **kwargs) -> bool:

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

    def normal_tf_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame: 
        
        # Stochastic Fast
        stoch_fast = ta.STOCHF(dataframe,5,3,3)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']

        # MACD
        macd = ta.MACD(dataframe,12,26,1)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        # EMAs 8 and 21
        dataframe['ema8'] = ta.EMA(dataframe['close'], timeperiod=8)
        dataframe['ema21'] = ta.EMA(dataframe['close'], timeperiod=21)

        # T3
        dataframe['t3'] = ta.T3(dataframe['low'], timeperiod=11, vfactor=0.7)

        # ADX
        dataframe['adx'] = ta.ADX(dataframe)

        # Stop loss
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['stoploss_rate'] = dataframe['close'] - (dataframe['atr'] * self.custom_info['sl_multiplier'])
        
        self.custom_info[metadata['pair']] = dataframe[['date', 'stoploss_rate']].copy().set_index('date')

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # The indicators for the 15m informative timeframe
        informative_15m = self.informative_15m_indicators(dataframe, metadata)
        dataframe = merge_informative_pair(dataframe, informative_15m, self.timeframe, '15m', ffill=True)

        # The indicators for the normal (5m) timeframe
        dataframe = self.normal_tf_indicators(dataframe, metadata)

        return dataframe

    #################################################################################
    ##                                                                             ## 
    ##                            BUY (Enter) conditions                           ##
    ##                                                                             ## 
    ################################################################################# 
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        
        conditions.append(
            (   # Condition 1: EMA8 close over EMA21 open
                self.enter_condition_1_enable.value &
                (qtpylib.crossed_above(dataframe['ema8'],  dataframe['ema21']))
            )
        )
        conditions.append(
            (   # Condition 2: Close above its T3
                self.enter_condition_2_enable.value &
                (
                    (dataframe['close'] >= dataframe['t3'])
                )
            )
        )
        conditions.append(
            (   # Condition 3: Stochastic fast is in a normal range
                self.enter_condition_3_enable.value &
                (
                    (dataframe['fastk'] >= 20) & (dataframe['fastk'] <= 80) &
                    (dataframe['fastd'] >= 20) & (dataframe['fastd'] <= 80)
                )
            )
        )
        conditions.append(
            (   # Condition 4: Stochastic K is over D
                self.enter_condition_4_enable.value &
                (dataframe['fastk'] >= dataframe['fastd'])
            )
        )
        conditions.append(
            (   # Condition 5: MACD is moving upwards
                self.enter_condition_5_enable.value &
                (
                    (dataframe['macd'] > dataframe['macd'].shift(1)) &
                    (dataframe['macdsignal'] > dataframe['macdsignal'].shift(1))
                )
            )
        )
        conditions.append(
            (   # Condition 6: Candles increasing in value
                self.enter_condition_6_enable.value &
                (dataframe['close'] > dataframe['close'].shift(1))
            )
        )
        conditions.append(
            (   # Condition 6: Candles increasing in value
                self.enter_condition_6_enable.value &
                (dataframe['adx'] >= 20)
            )
        )
        conditions.append(
            (   # Always-on condition - ensuring we have volume
                (dataframe['volume'] > 0)
            )
        )

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'enter_long'
            ] = 1
        
        return dataframe
    
    #################################################################################
    ##                                                                             ## 
    ##                            SELL (Exit) conditions                           ##
    ##                                                                             ## 
    ################################################################################# 
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        
        conditions.append(
            (   # Condition 1: EMA8 close falls under EMA21 open
                self.exit_condition_1_enable.value &
                (qtpylib.crossed_below(dataframe['ema8'],  dataframe['ema21'])) 
            )
        )
        conditions.append(
            (   # Condition 2:  Stochastic K and D are both below oversold range
                self.exit_condition_2_enable.value &
                (
                    (dataframe['fastk'] <= 80) &
                    (dataframe['fastd'] <= 80)
                )
            )
        )
        conditions.append(
            (   # Condition 3: Stochastic K is under D
                self.exit_condition_3_enable.value &
                (dataframe['fastk'] < dataframe['fastd'])
            )
        )
        conditions.append(
            (   # Condition 4: MACD decreasing momentum
                self.exit_condition_4_enable.value &
                (
                    (dataframe['macd'] < dataframe['macd'].shift(1)) &
                    (dataframe['macdsignal'] < dataframe['macdsignal'].shift(1))
                )
            )
        )
        conditions.append(
            (   # Always-on condition - ensure we have some volume
                (dataframe['volume'] > 0)
            )
        )
        conditions.append(
            (   # Condition 3: 
                self.exit_condition_3_enable.value &
                (dataframe['adx'] >= 20)
            )
        )
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'exit_long'
            ] = 1
        return dataframe

# Helper function 
def to_minutes(**timdelta_kwargs):
    return int(timedelta(**timdelta_kwargs).total_seconds() / 60)

