# Hextropian Freqtrade bots
Crypto Trading Bots for the [Freqtrade](https://freqtrade.io) framework

# Structure for this Repository
* All strategies are in the `user_data/strategies` directory.
* There is only one strategy per file.
* The file name is the same as the strategy name (minus the `.py` extension of course).
* Each strategy has a JSON config file located here in the main root directory with the same name. These files are sanitized, so you need to add your own credentials and modify the config to your needs.
* Some of these strategies have different variations. The names for each variation use the `_n` notation, so `mystrat_1` will be in `mystrat_1.py` file, and so on.
* All configs and strategies are tailored to work on [KuCoin](https://www.kucoin.com/ucenter/signup?rcode=rBSTQD7), although they can be adapted to work on other supported exchanges.

# How to use this Repository
* You need to have a fully functional `freqtrade` instance, proven to be running well.
* Clone the repo using:

 ```git clone https://github.com/hextropian/hextropian-freqtrade-bots.git```

 or download and decompress the files.
 * Copy the strategy file to your own `user_data/strategies` directory and the config file to your own root directory for `freqtrade`.
* Execute the proper commands, such as `freqtrade trade -c <path-to-config> --strategy <strategy-name>`.

# List of Strategies
1. `onem_wavecatcher`
Scalper strategy for the 1m timeframe. Attempts to detect 'pumps' or unusually large candles with increased volume, and rides the wave with a trailing stoploss and an exit strategy based on the T3 curve cross.

# Disclaimer
* These strategies come with no warranties whatsoever. Use at your own risk.
* Nothing in this repository can be construed as financial advice. These strategies are for experimentation and learning purposes only.
* Always understand the strategy you are running.
* Always backtest and run in dry-run mode until you are satisfied with the results.