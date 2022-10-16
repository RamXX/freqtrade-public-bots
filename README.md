# Hextropian Freqtrade bots
Crypto Trading Bots for the [Freqtrade](https://freqtrade.io) framework

# Structure for this Repository
* All strategies are in the `user_data/strategies` directory.
* There is only one strategy per file.
* The file name is the same as the strategy name (minus the `.py` extension of course).
* Each strategy *type* has a JSON config file located here in the main root directory with the same name minus the "_n" portion. These files are sanitized, so you need to add your own credentials and modify the config to your needs.
* Some of these strategies have different variations. The names for each variation use the `_n` notation, so `mystrat_1` will be in `mystrat_1.py` file, and so on.
* All configs and strategies are tailored to work on [KuCoin](https://www.kucoin.com/ucenter/signup?rcode=rBSTQD7), although they can be adapted to work on other supported exchanges.

# How to use this Repository
* You need to have a fully functional `freqtrade` instance, proven to be running well.
* Clone the repo using:

 ```git clone https://github.com/hextropian/hextropian-freqtrade-bots.git```

 or download and decompress the files.
 * Copy the strategy file to your own `user_data/strategies` directory and the config file to your own root directory for `freqtrade`.
* Execute the proper commands, such as `freqtrade trade -c <path-to-config> --strategy <strategy-name>`.

If you want to run the `all_kc_pairs.py` utility, you will need to pip install the dependencies first by executing:
`pip3 install -r requirements.txt`
# List of Strategies
1. `onem_wavecatcher`
Scalper strategy for the 1m timeframe. Attempts to detect 'pumps' or unusually large candles with increased volume, and rides the wave with a trailing stoploss and an exit strategy based on the T3 curve cross.
2. `TheForce`
15m scalper strategy based on the EMA5 close crossing EMA5 open, as well as Stochastic fast indicator. I'm not certain about the origin of this strategy, but I originally found it [here](https://github.com/StephaneTurquay/freqtrade-strategies-crypto-trading-bot). This one is unmodified, with the exeption of commenting out the Stochastic RSI indicator which is not used.
3. `TheForceMod_1`
This version adds additional constraints for more precise entries and exits, checks BTC and ETH status,
and implements a circuit breaker that sells the entire portfolio if a major drop happens to BTC and ETH
due to the high correlation with other coins.
4. `TheForceMod_2`
This is a 5m version that checks indicators for the 15m and 1h timeframes. It uses additional indicators such as ZLMA and Volume-weighted EMA.
5. `TheForceMod_3`
5m version that check indicators in the 4h timeframe. It also checks BTC and ETH, and uses T3 and ADX to measure thresholds and trends.
6. `TheForceMod_4`
This version replaces the core EMA5c/o for an EMA 8/21 cross on close.
7. `TheForceMod_5`
This version does not use exit signals but increases the accuracy of entries and relies on ROI/stoploss only for exits. Hyperopted for the config file portfolio.
8. `HyperStra_GSN_SMAOnly`
By @Farhad#0318. Idea from GodStraNew_SMAOnly. Performs well in sideways markets.
9. `EMA_8_21_cross_1`
Basic EMA 8/21 cross strategy on weekly. Barebones. Just to test the idea.
10. `EMA_8_21_cross_2`
Based on [10], this version triggers a buy only if the EMA 8 is over the 21 AND if the current and previous candles close above the EMA 8.
For selling, we're still assuming a downward cross. 
11. `EMA_8_21_cross_3`
Based on [10], this version triggers a buy only if the EMA 8 is over the 21 AND if the current and previous candles close above the EMA 8.
For selling, we also need the current and previous candle close below the EMA 21.
12. `EMA_8_21_cross_4`
Based on [10],  this version triggers a buy only if the EMA 8 is over the 21 AND if the current and previous
candles close above the EMA 8.
For selling, we also need the current and previous candle close below the EMA 21.
This version adds an additiona constraing: buys only happen over the 200 SMA. Sales happen if the close gets below that line also.
13. `EMA_8_21_cross_5`
This strategy uses both, shorts and longs, at a 4% distance of the cross entry point. Originally discussed by Discord users `EverForwardTech1972#1558` and `ChinaMatt#2982` (who coded it into Pine script) in the `.786 unlimited` [Discord](https://discord.gg/Sa8DxXdV).
14. `BB_-_CIX`
It leverages the volatility of the [CIX100](https://cix100.com/) token in the lower timeframes to scalp around the Bollinger bands in the 15m timeframe. Use with the `config-bb-cix.json` configuration file.
15. `BB_CIX_2`
1m-version of the one above with fixed trailing R:R and no exit signals.

# Disclaimer
* These strategies come with no warranties whatsoever. Use at your own risk.
* Nothing in this repository can be construed as financial advice. These strategies are for experimentation and learning purposes only.
* Always understand the strategy you are running.
* Always backtest and run in dry-run mode until you are satisfied with the results.