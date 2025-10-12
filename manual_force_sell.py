import sys
sys.path.append("/core/trading")

from alpaca_connector import submit_sell_order

submit_sell_order("AAPL", 1)
