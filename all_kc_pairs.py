# ========================================================= #
# usage: python3 ./all_kc_pairs.py                          #
# purpose: prints all coins and tokens currently traded in  #
# KuCoin against USDT, except leveraged tokens.             #
# The output is in JSON format and is ready to be pasted    #
# into freqtrade config files for 'pair_whitelist' entries  #
# You need to populate the all_kc_pairs_keys.py file with   #
# your own credentials.                                     # 
# ========================================================= #
import ccxt
import pandas as pd
import re
import warnings

# Local config. Includes API keys. Consider refactoring to retreive from
# environmental variables.
import all_kc_pairs_keys as c

# Won't truncate the output
pd.set_option('display.max_rows', None)

# Won't display warnings
warnings.filterwarnings('ignore')

exchange = ccxt.kucoin ({
    'apiKey': c.KEY_ID,
    'secret': c.KEY_SECRET,
    'password': c.PASSWORD
})

exchange.load_markets()
coin_pairs = exchange.symbols
valid_coin_pairs = []
regex = '^.*/USDT'
print ("[")
for coin_pair in coin_pairs:
  if re.match(regex, coin_pair) and exchange.markets[coin_pair]['active']:
    if not ('3L' in coin_pair) and not ('3S' in coin_pair):
        valid_coin_pairs.append(coin_pair)
        print (f'\"{coin_pair}\",')
print ("]")
