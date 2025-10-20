import sys

sys.path.append("/core/trading")

from alpaca_connector import submit_buy_order

submit_buy_order("AAPL", 1)
