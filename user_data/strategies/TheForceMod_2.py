# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement

import numpy as np  # noqa
from pandas import DataFrame

from freqtrade.strategy import IStrategy

import talib.abstract as ta
import pandas_ta as pta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from technical.qtpylib import hma
import logging
from typing import Optional
from freqtrade.persistence import Trade
from datetime import datetime
from freqtrade.strategy import merge_informative_pair

log = logging.getLogger(__name__)


class TheForceMod_2(IStrategy):
  
    INTERFACE_VERSION = 2

    minimal_roi = {
        "0": 0.07,
        "40": 0.06,
        "95": 0.035,
        "215": 0
    }

    stoploss = -0.34
    # stoploss = -0.99

    # Trailing stoploss
    trailing_stop = True
    trailing_stop_positive = 0.103
    trailing_stop_positive_offset = 0.109
    trailing_only_offset_is_reached = True

    # Optimal timeframe for the strategy.
    timeframe = '5m'
    inf_15m = '15m'
    inf_1h = '1h'

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False

    # use_custom_stoploss = True
    use_custom_stoploss = False

    # These values can be overridden in the "ask_strategy" section in the config.
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = True

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 200
 
    custom_info = {
        'risk_reward_ratio': 1.36,
        'sl_multiplier': 4.5,
        'set_to_break_even_at_profit': 1.01,
    }

    plot_config = {
        # Main plot indicators (Moving averages, ...)
        'main_plot': {
            'tema': {},
            'sar': {'color': 'white'},
        },
        'subplots': {
            # Subplots - each dict defines one additional plot
            "MACD": {
                'macd': {'color': 'blue'},
                'macdsignal': {'color': 'orange'},
            }
        }
    }

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:

        """
            custom_stoploss using a risk/reward ratio
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
        # allow force entries

        if (entry_tag == 'force_entry'):
            return True

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        if(len(dataframe) < 1):
            return True

        dataframe = dataframe.iloc[-1].squeeze()

        # Show exact parameters in log
        initial_sl_abs = dataframe['stoploss_rate']
        initial_sl = initial_sl_abs/rate-1
        risk_distance = rate-initial_sl_abs
        reward_distance = risk_distance*self.custom_info['risk_reward_ratio']
        take_profit_price_abs = rate+reward_distance
        take_profit_pct = take_profit_price_abs/rate-1
        log.info(f"Purchasing: {pair}")
        log.info(f"Price: {rate}")
        log.info(f"Take profit: {take_profit_price_abs} (rr: {self.custom_info['risk_reward_ratio']})")
        log.info(f"Take profit %: {take_profit_pct:.2%} ")
        log.info(f"Stoploss: {initial_sl_abs}")
        log.info(f"Stoploss %: {initial_sl:.2%}")
        # ----

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
        informative_pairs_15m = [(pair, '15m') for pair in pairs]
        return informative_pairs_15m

    def informative_15m_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        assert self.dp, 'DataProvider is required for multiple timeframes.'
        # Get the informative pair
        informative_15m = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.inf_15m)

        informative_15m['zlma'], _ = ZLMA(informative_15m['close'], length=50, mamode="linreg", offset=7)
        informative_15m['hma8'] = hma(informative_15m['close'], window=8)
        informative_15m['hma21'] = hma(informative_15m['close'], window=21)

        return informative_15m

    def informative_1h_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        assert self.dp, 'DataProvider is required for multiple timeframes.'
        # Get the informative pair
        informative_1h = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.inf_15m)

        informative_1h['zlma'], _ = ZLMA(informative_1h['close'], length=50, mamode="linreg", offset=7)

        return informative_1h

    def normal_tf_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Indicators for normal trading timeframe
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

        # EMA - Exponential Moving Average

        dataframe['ema5c'] = ta.EMA(dataframe['close'], timeperiod=5)
        dataframe['ema5o'] = ta.EMA(dataframe['open'], timeperiod=5)

        # Volume-weighted EMA
        dataframe['vwma200'] = pta.vwma(dataframe['close'], dataframe['volume'] ,length=200)

        # Stop loss
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['stoploss_rate'] = dataframe['close'] - (dataframe['atr'] * self.custom_info['sl_multiplier'])
        
        self.custom_info[metadata['pair']] = dataframe[['date', 'stoploss_rate']].copy().set_index('date')

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # The indicators for the 15m informative timeframe
        informative_15m = self.informative_15m_indicators(dataframe, metadata)
        dataframe = merge_informative_pair(dataframe, informative_15m, self.timeframe, self.inf_15m, ffill=True)

        # The indicators for the normal (5m) timeframe
        dataframe = self.normal_tf_indicators(dataframe, metadata)

        # dataframe['maxline'] = np.maximum(np.maximum(dataframe['vwma200'], dataframe['zlma_15m']), np.maximum(dataframe['hma8_15m'], dataframe['hma21_15m']))
        dataframe['maxline'] = np.maximum(dataframe['vwma200'], dataframe['zlma_15m'])
        dataframe['minline'] = np.minimum(dataframe['vwma200'], dataframe['zlma_15m'])

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) :
        dataframe.loc[
            (
                (   # Candle crosses up whichever is lower, the ZLMA or VWMA200 lines
                    (qtpylib.crossed_above(dataframe['close'],  dataframe['minline']))
                )
                &
                (   # Candle is trending up
                    (candle_uptrend(dataframe))
                )
                &
                (
                    (is_green(dataframe))
                )
                &
                (   # Candle closes above whichever is larger of ZLMA or VWMA200 (means it's higher than both)
                    (dataframe['close'] >= dataframe['minline'])
                ) 
                &
                (
                    (dataframe['close'] >= dataframe['zlma_15m'])
                ) 
                &
                (
                    (dataframe['zlma_15m'] >= dataframe['vwma200'])
                ) 
                &
                (
                    (dataframe['hma8_15m'] >= dataframe['hma21_15m'])
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
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) :
        dataframe.loc[
            (
                 (
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
                |
                (   # Momentum is over
                    (qtpylib.crossed_below(dataframe['hma8_15m'],  dataframe['maxline']))
                )
            ),
            'sell'] = 1
        return dataframe


def ZLMA(close, length=32, mamode='ema', offset=0, **kwargs):
    from pandas_ta.utils import get_offset, verify_series
    """
    Indicator: Zero Lag Moving Average (ZLMA)
    The Pandas implementation seems broken for anything other than mamode='ema'
    so implementing this workaround.
    Source: https://github.com/twopirllc/pandas-ta/issues/394 
    """
    # Validate Arguments
    length = int(length) if length and length > 0 else 10
    mamode = mamode.lower() if isinstance(mamode, str) else "ema"
    close = verify_series(close, length)
    offset = get_offset(offset)

    if close is None: return
    # Calculate Result
    lag = int(0.5 * length - 1)
    close_ = 2 * close  - close.shift(lag)
    if   mamode == "dema":   zlma = pta.dema(close_, length=length, **kwargs)
    elif mamode == "fwma":   zlma = pta.fwma(close_, length=length, **kwargs)
    elif mamode == "hma":    zlma = pta.hma(close_, length=length, **kwargs)
    elif mamode == "linreg": zlma = pta.linreg(close_, length=length, **kwargs)
    elif mamode == "pwma":   zlma = pta.pwma(close_, length=length, **kwargs)
    elif mamode == "rma":    zlma = pta.rma(close_, length=length, **kwargs)
    elif mamode == "sma":    zlma = pta.sma(close_, length=length, **kwargs)
    elif mamode == "swma":   zlma = pta.swma(close_, length=length, **kwargs)
    elif mamode == "sinwma": zlma = pta.sinwma(close_, length=length, **kwargs)
    elif mamode == "t3":     zlma = pta.t3(close_, length=length, **kwargs)
    elif mamode == "tema":   zlma = pta.tema(close_, length=length, **kwargs)
    elif mamode == "trima":  zlma = pta.trima(close_, length=length, **kwargs)
    elif mamode == "vidya":  zlma = pta.vidya(close_, length=length, **kwargs)
    elif mamode == "wma":    zlma = pta.wma(close_, length=length, **kwargs)
    else:                    zlma = pta.ema(close_, length=length, **kwargs) # "ema"

    # Offset
    if offset != 0:
        zlma = zlma.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        zlma.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        zlma.fillna(method=kwargs["fill_method"], inplace=True)
    # Name & Category
    zlma.name = f"ZL_{zlma.name}"
    zlma.category = "overlap"

    return zlma, zlma.name

def is_green(dataframe):
    """
    Boolean series determining if the candle is green or not.
    if the value of open and close are equal, then the candle is considered green
    """
    return dataframe['close'] >= dataframe['open']

def candle_uptrend(dataframe, consider_size=False):
    """
    Returns boolean series. Uptrend is true if the current candle is green and 
    the last two before were also green. Size of the candle is disabled by default
    but can be enabled as consider_size=True. In that case, the size of the candles
    need to be increasing in magnitude. Volume always need to be > 0 regardless.
    """
    c1 = dataframe['close'].fillna(0) - dataframe['open'].fillna(0)
    v0 = (c1 > 0) & dataframe['volume'] > 0

    if consider_size:
        c2 = dataframe['close'].shift(1).fillna(0) - dataframe['open'].shift(1).fillna(0)
        c3 = dataframe['close'].shift(2).fillna(0) - dataframe['open'].shift(2).fillna(0)
        
        v1 = (dataframe['close'].fillna(0) > dataframe['close'].shift(1).fillna(0)) & (c1 > c2)
        v2 = (dataframe['close'].fillna(0) > dataframe['close'].shift(2).fillna(0)) & (c1 > c3)
        v3 = (dataframe['close'].shift(1).fillna(0) > dataframe['close'].shift(2).fillna(0)) & (c2 > c3)
    else:
        v1 = dataframe['close'].fillna(0) > dataframe['close'].shift(1).fillna(0) 
        v2 = dataframe['close'].fillna(0) > dataframe['close'].shift(2).fillna(0)
        v3 = dataframe['close'].shift(1).fillna(0) > dataframe['close'].shift(2).fillna(0)
        
    return (v0 & v1 & v2 & v3)